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
        """
        Unified chunk-based method to encode images to latents with adaptive sub-batching.
        Uses inference_mode for reduced overhead and pre-allocates the output tensor.
        Fixes the issue with cpu(non_blocking=True) by using to('cpu', non_blocking=True).
        """
        import torch

        try:
            if images.dim() == 3:
                images = images.unsqueeze(0)

            batch_size = images.shape[0]
            if batch_size == 0:
                return None

            # Pin memory for faster CPU→GPU transfers
            if images.device.type == 'cpu':
                images = images.pin_memory()

            # Ensure contiguous
            images = images.contiguous(memory_format=torch.contiguous_format)

            # Get an example latent for shape inference
            sample_chunk = images[0:1].to(self.vae.device)
            with torch.cuda.amp.autocast(dtype=self.vae.dtype):
                sample_dist = self.vae.encode(sample_chunk).latent_dist
                sample_latent = sample_dist.sample() * self.vae.config.scaling_factor

            latent_shape = [batch_size] + list(sample_latent.shape[1:])

            # Pre-allocate latents on GPU or CPU depending on user preference
            out_device = self.vae.device if keep_on_gpu else "cpu"
            latents_out = torch.empty(size=latent_shape, dtype=sample_latent.dtype, device=out_device)

            # Adaptive sub-batch start
            sub_batch_size = getattr(self.config, 'vae_batch_size', 8)
            index = 0

            def reduce_sub_batch_size():
                nonlocal sub_batch_size
                sub_batch_size = max(sub_batch_size // 2, 1)
                self.logger.warning(f"Reduced sub-batch size to {sub_batch_size} due to OOM")

            with torch.cuda.amp.autocast(dtype=self.vae.dtype):
                while index < batch_size:
                    end = min(index + sub_batch_size, batch_size)
                    chunk = images[index:end].to(self.vae.device, non_blocking=True)
                    try:
                        dist = self.vae.encode(chunk).latent_dist
                        chunk_latents = dist.sample() * self.vae.config.scaling_factor

                        # Copy into pre-allocated output
                        if keep_on_gpu:
                            latents_out[index:end].copy_(chunk_latents, non_blocking=True)
                        else:
                            latents_out[index:end].copy_(
                                chunk_latents.to("cpu", non_blocking=True),
                                non_blocking=True
                            )

                        index = end

                    except RuntimeError as re:
                        if "out of memory" in str(re).lower():
                            torch.cuda.empty_cache()
                            reduce_sub_batch_size()
                            # If there's no more to reduce, raise
                            if sub_batch_size == 1:
                                raise
                        else:
                            self.logger.error(f"Unexpected VAE encoding error: {str(re)[:200]}")
                            raise

            # If only 1 image, squeeze batch dimension
            if batch_size == 1:
                return latents_out.squeeze(0)

            return latents_out

        except Exception as e:
            self.logger.error(f"VAE encoding error: {str(e)[:200]}...")
            torch.cuda.empty_cache()
            return None
