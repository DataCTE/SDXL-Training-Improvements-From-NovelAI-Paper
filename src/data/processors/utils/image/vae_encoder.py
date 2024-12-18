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

# Add this function
def is_xformers_available():
    """Check if xformers is available."""
    try:
        import xformers
        return True
    except ImportError:
        return False

logger = logging.getLogger(__name__)

class VAEEncoder:
    """Handles VAE encoding with memory optimization and async support."""
    
    def __init__(self, vae: AutoencoderKL, config: VAEEncoderConfig):
        """Initialize VAE encoder with configuration."""
        self.vae = vae
        self.config = config
        
        # Freeze VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Move to device and set dtype from config with fallback
        dtype = getattr(torch, self.config.forced_dtype, torch.float32)
        self.vae.to(device=self.config.device, dtype=dtype)
        
        # Conditionally enable slicing or xformers
        if self.config.enable_vae_slicing:
            self.vae.enable_slicing()
        
        if self.config.enable_xformers_attention and is_xformers_available():
            self.vae.enable_xformers_memory_efficient_attention()
        
        logging.debug(
            f"Initialized VAEEncoder with device={self.config.device}, dtype={dtype}, "
            f"slicing={self.config.enable_vae_slicing}, xformers={self.config.enable_xformers_attention}"
        )

        # Add tensor cache for frequently used sizes
        self._tensor_cache = WeakValueDictionary()
        
        # Add batch processing queue
        self.batch_queue_size = getattr(config, 'batch_queue_size', 32)
        self.processing_queue = []
        
        # Enable memory optimizations
        self._enable_optimizations()

    def _enable_optimizations(self):
        """Enable all available optimizations for faster processing."""
        if torch.cuda.is_available():
            # Enable TF32 for faster processing on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal cudnn algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Always enable VAE slicing for memory efficiency
        self.vae.enable_slicing()
        
        if self.config.enable_xformers_attention and is_xformers_available():
            self.vae.enable_xformers_memory_efficient_attention()

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear any tensor cache if you plan to use caching
            # if hasattr(self, '_tensor_cache'):
            #     self._tensor_cache.clear()

            # Clear CUDA cache if using GPU
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Clear references to the VAE
            del self.vae
            gc.collect()

            logger.info("Successfully cleaned up VAE encoder resources.")

        except Exception as e:
            logger.error(f"Error during VAE encoder cleanup: {str(e)}")
            try:
                torch.cuda.empty_cache()
                gc.collect()
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

    # Optional caching function (commented out usage in encode_images)
    # def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    #     """Get or create tensor from cache."""
    #     tensor = self._tensor_cache.get(key)
    #     if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
    #         tensor = torch.empty(shape, dtype=dtype, device=self.config.device)
    #         self._tensor_cache[key] = tensor
    #     return tensor

    @torch.inference_mode()
    async def encode_images(self, images: "torch.Tensor", keep_on_gpu: bool = False) -> "torch.Tensor":
        """Optimized batch encoding with dynamic batching and prefetching."""
        try:
            if images.dim() == 3:
                images = images.unsqueeze(0)

            batch_size = images.shape[0]
            if batch_size == 0:
                return None

            # Use pinned memory for faster transfers
            if images.device.type == 'cpu':
                images = images.pin_memory()

            # Pre-allocate output tensor using cache
            latent_shape = self._get_latent_shape(images)
            out_device = self.vae.device if keep_on_gpu else "cpu"
            latents_out = self._get_cached_tensor(
                f"latents_{latent_shape}_{out_device}",
                latent_shape,
                self.vae.dtype
            )

            # Process in optimal sub-batches
            sub_batch_size = self._get_optimal_batch_size(images)
            
            async def process_sub_batch(start_idx):
                end_idx = min(start_idx + sub_batch_size, batch_size)
                chunk = images[start_idx:end_idx].to(
                    self.vae.device,
                    non_blocking=True
                )
                
                with torch.cuda.amp.autocast(dtype=self.vae.dtype):
                    dist = self.vae.encode(chunk).latent_dist
                    chunk_latents = dist.sample() * self.vae.config.scaling_factor

                # Efficient copy to output
                if keep_on_gpu:
                    latents_out[start_idx:end_idx].copy_(
                        chunk_latents,
                        non_blocking=True
                    )
                else:
                    latents_out[start_idx:end_idx].copy_(
                        chunk_latents.to("cpu", non_blocking=True),
                        non_blocking=True
                    )

            # Process sub-batches concurrently
            tasks = [
                process_sub_batch(i)
                for i in range(0, batch_size, sub_batch_size)
            ]
            await asyncio.gather(*tasks)

            if batch_size == 1:
                return latents_out.squeeze(0)
            return latents_out

        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)[:200]}")
            torch.cuda.empty_cache()
            return None

    def _get_optimal_batch_size(self, images: torch.Tensor) -> int:
        """Determine optimal batch size based on available memory and image size."""
        try:
            if not torch.cuda.is_available():
                return self.config.vae_batch_size

            total_memory = torch.cuda.get_device_properties(self.vae.device).total_memory
            free_memory = torch.cuda.memory_reserved(self.vae.device)
            
            # Calculate memory per image
            sample_latent = self.vae.encode(images[0:1]).latent_dist.sample()
            bytes_per_image = sample_latent.element_size() * sample_latent.nelement()
            
            # Leave 20% memory buffer
            safe_memory = int(free_memory * 0.8)
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
