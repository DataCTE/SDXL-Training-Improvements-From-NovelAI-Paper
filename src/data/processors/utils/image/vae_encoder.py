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
    """

    @staticmethod
    def enable_torch_backend_optimizations(config: VAEEncoderConfig):
        """
        Centralized Torch backend optimizations.
        This replaces duplication in other modules.
        """
        if torch.cuda.is_available():
            # Enable TF32 on Ampere GPUs and set cudnn options
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    @staticmethod
    def perform_basic_cleanup():
        """
        Centralized GPU memory cleanup (empty cache, garbage collect).
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def __init__(self, vae: AutoencoderKL, config: VAEEncoderConfig):
        """Initialize VAE encoder with configuration."""
        self.vae = vae
        self.config = config

        # Apply shared Torch backend optimizations
        VAEEncoder.enable_torch_backend_optimizations(config)

        # Freeze VAE and move to device
        self._setup_vae()

        # Enable VAE-related optimizations (slicing, xformers, etc.)
        self._enable_vae_specific_optimizations()

        # Setup optional flash attention and tensor parallel
        self._setup_flash_attention()
        self._setup_tensor_parallel()

        # Initialize caching
        self.latent_cache = LRUCache(maxsize=1024)
        self._static_shapes = set()

    def _setup_vae(self):
        """Setup VAE model with proper configuration."""
        self.vae.requires_grad_(False)
        self.vae.eval()
        dtype = getattr(torch, self.config.forced_dtype, torch.float32)
        self.vae.to(device=self.config.device, dtype=dtype)

    def _enable_vae_specific_optimizations(self):
        """Enable VAE-specific optimizations for faster processing."""
        if torch.cuda.is_available():
            # Always enable VAE slicing for memory efficiency
            self.vae.enable_slicing()

        # Enable xformers if configured and available
        if self.config.enable_xformers_attention and is_xformers_available():
            self.vae.enable_xformers_memory_efficient_attention()

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear CUDA cache if using GPU
            if self.config.device.type == 'cuda':
                VAEEncoder.perform_basic_cleanup()

            # Clear references to the VAE
            del self.vae

            logger.info("Successfully cleaned up VAE encoder resources.")

        except Exception as e:
            logger.error(f"Error during VAE encoder cleanup: {str(e)}")
            try:
                VAEEncoder.perform_basic_cleanup()
            except:
                pass

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

    def _enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention if possible."""
        try:
            import xformers
            self.vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention for VAE")
        except ImportError:
            logger.warning("xformers not available, using standard attention")

    @torch.inference_mode()
    async def encode_images(self, images: torch.Tensor, keep_on_gpu: bool = False) -> torch.Tensor:
        """Encode images to latent space."""
        try:
            # Check cache
            cache_key = self._get_cache_key(images)
            if cached_latents := self.latent_cache.get(cache_key):
                return cached_latents

            # Process in optimal chunks
            chunks = self._get_optimal_chunks(images)
            latents = await self._process_chunks_parallel(chunks, keep_on_gpu)

            # Cache results
            self.latent_cache[cache_key] = latents
            return latents

        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)}")
            return None

    async def _process_chunks_parallel(self, chunks, keep_on_gpu):
        """
        Process chunks in parallel with advanced optimizations.
        """
        try:
            results = []
            # Process chunks sequentially but allow other async operations to run
            for chunk in chunks:
                # Wrap the synchronous processing in an executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_chunk, chunk, keep_on_gpu
                )
                results.append(result)
            return self._combine_results(results)

        except Exception as e:
            logger.error(f"Error in parallel chunk processing: {e}")
            raise

    def _combine_results(self, results):
        """Combine processed chunks back into a single tensor."""
        if not results:
            return None
        return torch.cat(results, dim=0)

    @torch.compile(fullgraph=True, dynamic=False)
    def _process_chunk(self, chunk, keep_on_gpu):
        """Optimized chunk processing with TorchDynamo."""
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                latents = self.vae.encode(chunk).latent_dist.sample()
                if not keep_on_gpu:
                    latents = latents.cpu()
                return latents
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    def _get_optimal_chunks(self, images):
        """Get optimal chunk sizes based on hardware and model."""
        if torch.cuda.is_available():
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            return self._split_for_hardware(images, sm_count)
        return [images]

    def _split_for_hardware(self, images, sm_count):
        """Split batch optimally for hardware."""
        chunk_size = max(1, (sm_count * 128) // images.shape[-1])
        return torch.chunk(images, chunk_size)

    def _setup_flash_attention(self):
        """Enable flash attention for faster processing."""
        try:
            from flash_attn import flash_attn_func
            self.use_flash_attention = True
            # Patch VAE attention layers to use flash attention
            for module in self.vae.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    module._flash_attention_func = flash_attn_func
        except ImportError:
            self.use_flash_attention = False

    def _setup_tensor_parallel(self):
        """Setup tensor parallelism for multi-GPU processing."""
        if torch.cuda.device_count() > 1:
            self.vae = torch.nn.parallel.DistributedDataParallel(
                self.vae,
                device_ids=[self.config.device],
                output_device=self.config.device,
                find_unused_parameters=True
            )

    def _get_optimal_batch_size(self, images: torch.Tensor) -> int:
        """Determine optimal batch size based on available memory and image size."""
        try:
            if not torch.cuda.is_available():
                return self.config.vae_batch_size

            total_memory = torch.cuda.get_device_properties(self.vae.device).total_memory
            used_memory = torch.cuda.memory_allocated(self.vae.device)
            # Calculate actual available memory
            available_memory = total_memory - used_memory
            safe_memory = int(available_memory * 0.8)  # Leave 20% buffer

            # Calculate memory per image
            sample_latent = self.vae.encode(images[0:1]).latent_dist.sample()
            bytes_per_image = sample_latent.element_size() * sample_latent.nelement()

            optimal_size = max(1, safe_memory // bytes_per_image)
            return min(optimal_size, self.config.vae_batch_size)

        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}")
            return self.config.vae_batch_size

    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=self.vae.device)
            self._tensor_cache[key] = tensor
        return tensor

    def _get_latent_shape(self, images: torch.Tensor) -> tuple:
        """Calculate latent shape without encoding."""
        b, c, h, w = images.shape
        return (b, 4, h // 8, w // 8)

    def _get_cache_key(self, images: torch.Tensor) -> str:
        """
        Generate a simple cache key based on shape, dtype, device, and possibly data hash.
        Note: This is just a sample approach - ensure the key is robust enough for your use case.
        """
        shape_str = "_".join([str(dim) for dim in images.shape])
        dtype_str = str(images.dtype)
        device_str = str(images.device)
        return f"{shape_str}_{dtype_str}_{device_str}"
