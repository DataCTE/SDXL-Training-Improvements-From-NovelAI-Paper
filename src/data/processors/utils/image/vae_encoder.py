import torch
import logging
from typing import Optional
import time
import gc
from weakref import WeakValueDictionary
from src.config.config import VAEEncoderConfig
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
from src.data.processors.utils.batch_utils import get_gpu_memory_usage
from diffusers import AutoencoderKL
import asyncio
from collections import OrderedDict

logger = logging.getLogger(__name__)

def is_xformers_available():
    """Check if xformers is available."""
    try:
        import xformers
        return True
    except ImportError:
        return False

class LRUCache:
    """An LRU cache with a max size."""
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

class VAEEncoder:
    """
    Advanced VAEEncoder with dynamic chunk sizing and optional torch.compile.
    Intended for large-scale batch encoding.

    Key Updates:
    - Dynamic chunk size in encode_images()
    - Optional torch.compile for encode call
    - LRU caching behavior for latents
    - xformers / slicing usage
    """

    @staticmethod
    def enable_torch_backend_optimizations(config):
        if torch.cuda.is_available():
            # Enable TF32 on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    def __init__(self, vae: AutoencoderKL, config: VAEEncoderConfig):
        self.logger = logging.getLogger(__name__)
        self.vae = vae
        self.config = config

        # Global optimizations
        self.enable_torch_backend_optimizations(config)

        # Freeze & prepare VAE
        self._setup_vae()
        self._enable_vae_specific_optimizations()
        self._setup_flash_attention()
        self._setup_tensor_parallel()

        # Use this if torch.compile is stable for your environment
        if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
            self.logger.info("Wrapping VAE encode() with torch.compile() for potential speed-ups.")
            self.vae.encode = torch.compile(self.vae.encode, mode="default")

        # LRU for latents
        self.latent_cache = LRUCache(getattr(config, "latent_cache_size", 64))

        # Try to do a warm-up pass to compile kernels
        self._warmup_vae()

    def _setup_vae(self):
        self.vae.requires_grad_(False)
        self.vae.eval()
        dtype_str = getattr(self.config, "forced_dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)
        self.vae.to(device=self.config.device, dtype=dtype)

    def _enable_vae_specific_optimizations(self):
        if torch.cuda.is_available():
            # Slicing can help reduce memory usage
            if hasattr(self.vae, "enable_slicing"):
                self.vae.enable_slicing()
        if getattr(self.config, "enable_xformers_attention", False) and is_xformers_available():
            # xformers memory-efficient attention
            try:
                self.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                self.logger.warning(f"xformers not enabled in vae: {e}")

    def _setup_flash_attention(self):
        try:
            from flash_attn import flash_attn_func
            self.use_flash_attention = True
            # Potentially iterate submodules to apply custom attention
        except ImportError:
            self.use_flash_attention = False

    def _setup_tensor_parallel(self):
        # If you have multiple GPUs, you can wrap with DDP
        if torch.cuda.device_count() > 1:
            self.vae = torch.nn.parallel.DistributedDataParallel(
                self.vae,
                device_ids=[self.config.device],
                output_device=self.config.device,
                find_unused_parameters=False
            )

    def _warmup_vae(self):
        """Optional warm-up pass to compile CUDA kernels if shapes are consistent."""
        if torch.cuda.is_available():
            dummy_h = getattr(self.config, "warmup_height", 64)
            dummy_w = getattr(self.config, "warmup_width", 64)
            dummy_input = torch.zeros((1, 3, dummy_h, dummy_w), device=self.config.device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.vae.encode(dummy_input).latent_dist.sample()
            torch.cuda.synchronize()
            self.logger.info("VAE warm-up pass complete.")

    async def cleanup(self):
        try:
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            del self.vae
            self.logger.info("VAE encoder cleanup done.")
        except Exception as e:
            self.logger.error(f"VAE cleanup error: {str(e)}")

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during VAE encoder destructor: {e}")

    @staticmethod
    def _make_cache_key(images: torch.Tensor) -> str:
        device_str = str(images.device)
        shape_str = "x".join(str(s) for s in images.shape)
        # Quick partial hash
        data_hash = 0
        if images.numel() > 0:
            data_hash = hash(images[0, 0, 0, 0].item())
        return f"{shape_str}_{device_str}_{data_hash}"

    def _get_auto_chunk_size(self, images: torch.Tensor) -> int:
        """
        Dynamically guess the largest sub-batch that fits in GPU memory, or use config.inference_chunk_size.
        """
        try:
            config_chunk_size = getattr(self.config, "inference_chunk_size", 64)
            total_mem = torch.cuda.get_device_properties(self.config.device).total_memory
            used_mem = torch.cuda.memory_allocated(self.config.device)
            available_mem = total_mem - used_mem
            buffer_factor = 0.8

            # Estimate with one sample
            sample = images[0:1].to(self.config.device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.half):
                example_latent = self.vae.encode(sample).latent_dist.sample()
            bytes_per_image = example_latent.element_size() * example_latent.numel()

            max_images = int((available_mem * buffer_factor) // bytes_per_image)
            auto_chunk_size = max(1, min(config_chunk_size, max_images))
            return auto_chunk_size
        except Exception as e:
            self.logger.warning(f"Dynamic chunk sizing failed: {e}")
            return getattr(self.config, "inference_chunk_size", 64)

    @torch.inference_mode()
    async def encode_images(self, images: torch.Tensor, keep_on_gpu=False) -> torch.Tensor:
        """
        Encode a batch of images into latents. 
        Dynamically picks chunk size for maximum speed/compatibility.
        """

        # Attempt cache
        cache_key = self._make_cache_key(images)
        cached = self.latent_cache.get(cache_key)
        if cached is not None:
            return cached.to(self.config.device) if keep_on_gpu else cached.cpu()

        chunk_size = self._get_auto_chunk_size(images)
        latents_list = []

        start_idx = 0
        while start_idx < images.shape[0]:
            end_idx = start_idx + chunk_size
            chunk = images[start_idx:end_idx]

            if chunk.device != self.config.device:
                chunk = chunk.to(self.config.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                lat_dist = self.vae.encode(chunk).latent_dist
                lat_chunk = lat_dist.sample()

            if not keep_on_gpu:
                lat_chunk = lat_chunk.cpu()

            latents_list.append(lat_chunk)
            start_idx = end_idx

        latents = torch.cat(latents_list, dim=0)
        self.latent_cache[cache_key] = latents  # store in LRU

        # If size is over limit, pop oldest
        if len(self.latent_cache.cache) > self.latent_cache.maxsize:
            self.latent_cache.cache.popitem(last=False)

        return latents
