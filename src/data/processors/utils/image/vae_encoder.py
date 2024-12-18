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

    @torch.no_grad()
    async def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        try:
            # Ensure image is in correct format
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(memory_format=torch.contiguous_format).float()
            
            # Process through VAE in chunks if needed
            latents = []
            for i in range(0, image.shape[0], self.config.vae_batch_size):
                chunk = image[i:i + self.config.vae_batch_size]
                chunk = chunk.to(self.vae.device, dtype=self.vae.dtype)
                
                # Encode and scale
                latent_dist = self.vae.encode(chunk).latent_dist
                chunk_latents = latent_dist.sample()
                chunk_latents = chunk_latents * self.vae.config.scaling_factor
                
                latents.append(chunk_latents.cpu())
                
            # Combine chunks
            latents = torch.cat(latents, dim=0)
            
            return latents
            
        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)[:200]}...")
            return None

    def __del__(self):
        """Cleanup when encoder is deleted."""
        self.cleanup()

    def _enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention if possible."""
        try:
            import xformers
            self.vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention for VAE")
        except ImportError:
            logger.warning("xformers not available, using standard attention")

    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=self.config.device)
            self._tensor_cache[key] = tensor
        return tensor

    @torch.no_grad()
    async def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        images = images.to(self.config.device)
        latents = self.vae.encode(images).latent_dist.sample()
        # Additional logging
        logging.debug(
            f"Encoded batch of shape {images.shape} into latents of shape {latents.shape} "
            f"in {time.time() - start_time:.3f}s"
        )
        return latents * self.vae.config.scaling_factor

    async def encode_image(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode a single image through VAE asynchronously."""
        try:
            # Add batch dimension if needed
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Encode through VAE
            latents = await self.encode_batch(image_tensor)
            
            # Remove batch dimension
            result = latents.squeeze(0)
            del image_tensor
            del latents
            
            return result
            
        except Exception as e:
            logger.error(f"Error encoding through VAE: {str(e)[:200]}...")
            if 'image_tensor' in locals():
                del image_tensor
            if 'latents' in locals():
                del latents
            torch.cuda.empty_cache()
            return None

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear tensor cache
            if hasattr(self, '_tensor_cache'):
                self._tensor_cache.clear()
            
            # Clear CUDA cache if using GPU
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Clear any references to the VAE
            if hasattr(self, 'vae'):
                del self.vae
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Successfully cleaned up VAE encoder resources")
            
        except Exception as e:
            logger.error(f"Error during VAE encoder cleanup: {str(e)}")
            # Try one last time to clear memory
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
