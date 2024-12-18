import torch
import logging
from typing import Optional
import time
import gc
from weakref import WeakValueDictionary
from src.config.config import VAEEncoderConfig  # Import the new config class
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
    """LRU cache with a max size."""
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end to show recently used
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
    Handles VAE encoding with memory optimization and async support.
    Includes staticmethods to unify Torch backend and cleanup logic.

    Modified to reduce overhead from repeated torch.compile calls
    and small-chunk processing. Large sub-batches usually
    yield much faster throughput for big datasets.
    """

    @staticmethod
    def enable_torch_backend_optimizations(config):
        """
        Enable TF32 on Ampere GPUs and set cudnn options.
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    @staticmethod
    def perform_basic_cleanup():
        """ Centralized GPU memory cleanup (empty cache, garbage collect). """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def __init__(self, vae, config):
        """
        Initialize VAE encoder with configuration, enabling large
        sub-batches to reduce iteration overhead for big datasets.
        """
        self.logger = logging.getLogger(__name__)
        self.vae = vae
        self.config = config

        # Use shared Torch optimizations
        self.enable_torch_backend_optimizations(config)

        # Freeze and relocate VAE
        self._setup_vae()

        # VAE-specific optimizations
        self._enable_vae_specific_optimizations()

        # Try flash attention
        self._setup_flash_attention()

        # Setup multi-GPU if needed
        self._setup_tensor_parallel()

        # Larger chunk size for big datasets
        # Increase this if your GPU has enough memory; e.g., 128 or 256
        self.inference_chunk_size = getattr(config, "inference_chunk_size", 64)

        # Simple LRU cache
        self.latent_cache = {}
        self.max_cache_size = 64  # Keep it small to avoid overhead

        # Optional warm-up to compile CUDA kernels:
        self._warmup_vae()

    def _warmup_vae(self):
        """Optional warm-up pass for the VAE."""
        if torch.cuda.is_available():
            dummy_size = getattr(self.config, "test_warmup_size", 128)
            dummy_input = torch.zeros((1, 3, dummy_size, dummy_size), device=self.config.device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.half):
                _ = self.vae.encode(dummy_input).latent_dist.sample()
            torch.cuda.synchronize()
            self.logger.info("Warm-up pass complete.")

    def _setup_vae(self):
        """Freeze and move VAE to proper device & dtype."""
        self.vae.requires_grad_(False)
        self.vae.eval()
        forced_dtype = getattr(self.config, "forced_dtype", "float32")
        dtype = getattr(torch, forced_dtype, torch.float32)
        self.vae.to(device=self.config.device, dtype=dtype)

    def _enable_vae_specific_optimizations(self):
        """Enable VAE slicing, xformers, etc. for memory or speed gains."""
        if torch.cuda.is_available():
            self.vae.enable_slicing()

        if getattr(self.config, "enable_xformers_attention", False):
            try:
                import xformers
                self.vae.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass

    def _setup_flash_attention(self):
        """Try enabling flash attention at model level if possible."""
        try:
            from flash_attn import flash_attn_func
            self.use_flash_attention = True
            for module in self.vae.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    module._flash_attention_func = flash_attn_func
        except ImportError:
            self.use_flash_attention = False

    def _setup_tensor_parallel(self):
        """Wrap the VAE in DDP if multiple GPUs are available."""
        if torch.cuda.device_count() > 1:
            self.vae = torch.nn.parallel.DistributedDataParallel(
                self.vae,
                device_ids=[self.config.device],
                output_device=self.config.device,
                find_unused_parameters=True
            )

    async def cleanup(self):
        """Clean up VAE resources."""
        try:
            if self.config.device.type == 'cuda':
                self.perform_basic_cleanup()
            del self.vae
            self.logger.info("VAE encoder cleanup done.")
        except Exception as e:
            self.logger.error(f"VAE cleanup error: {str(e)}")

    def __del__(self):
        """Cleanup when encoder is deleted."""
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
    def _make_cache_key(images):
        # Something simple: shape + device + small data hash
        device_str = str(images.device)
        shape_str = "x".join(str(s) for s in images.shape)
        # Very cheap partial hash:
        data_hash = hash(images[0, 0, 0, 0].item()) if images.numel() > 0 else 0
        return f"{shape_str}_{device_str}_{data_hash}"

    @torch.inference_mode()
    async def encode_images(self, images, keep_on_gpu=False):
        """
        Encode images into latents. Uses bigger chunking to reduce overhead.
        """
        # Attempt to retrieve from cache
        key = self._make_cache_key(images)
        if key in self.latent_cache:
            latents_cached = self.latent_cache[key]
            return latents_cached.to(self.config.device) if keep_on_gpu else latents_cached.cpu()

        # Process in chunks:
        latents_list = []
        for start in range(0, images.shape[0], self.inference_chunk_size):
            end = start + self.inference_chunk_size
            chunk = images[start:end]

            # Move chunk to device if not there
            if chunk.device != self.config.device:
                chunk = chunk.to(self.config.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.half):
                latent_dist = self.vae.encode(chunk).latent_dist
                latents_chunk = latent_dist.sample()
            # Possibly move off GPU if we won't keep it
            if not keep_on_gpu:
                latents_chunk = latents_chunk.cpu()

            latents_list.append(latents_chunk)

        latents = torch.cat(latents_list, dim=0)
        # Simple cache, remove oldest if over capacity
        if len(self.latent_cache) >= self.max_cache_size:
            self.latent_cache.pop(next(iter(self.latent_cache)))
        self.latent_cache[key] = latents
        return latents if keep_on_gpu else latents.cpu()
