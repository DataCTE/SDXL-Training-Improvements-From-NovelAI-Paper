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
        Unified chunk-based method to encode one or more images to latents.

        Args:
            images: A 4D tensor (B, C, H, W) or 3D (C, H, W).
            keep_on_gpu: If True, keeps latents on GPU; otherwise moves them to CPU.

        Returns:
            Latents as torch.Tensor on GPU or CPU, or None on failure.
        """
        try:
            if images.dim() == 3:
                images = images.unsqueeze(0)

            # Optional pinned memory for faster GPU transfers
            if images.device.type == 'cpu':
                images = images.pin_memory()

            # Force contiguous for better memory access
            images = images.contiguous(memory_format=torch.contiguous_format)

            latents_list = []
            batch_size = images.shape[0]

            # Potential dynamic sub-batching idea (pseudo-code):
            # sub_batch_size = min(self.config.vae_batch_size, self._available_gpu_capacity() // chunk_size_estimate)
            sub_batch_size = self.config.vae_batch_size

            with torch.cuda.amp.autocast(dtype=self.vae.dtype):
                for i in range(0, batch_size, sub_batch_size):
                    chunk = images[i : i + sub_batch_size]
                    # Move chunk to GPU in half precision
                    chunk = chunk.to(self.vae.device, non_blocking=True)
                    
                    # Encode
                    latent_dist = self.vae.encode(chunk).latent_dist
                    chunk_latents = latent_dist.sample() * self.vae.config.scaling_factor
                    
                    if keep_on_gpu:
                        latents_list.append(chunk_latents)
                    else:
                        latents_list.append(chunk_latents.cpu(non_blocking=True))
            
            latents = torch.cat(latents_list, dim=0)
            
            # If we have a single image, squeeze batch dimension
            if latents.shape[0] == 1:
                return latents.squeeze(0)
            
            return latents
        
        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)[:200]}...")
            torch.cuda.empty_cache()
            return None
