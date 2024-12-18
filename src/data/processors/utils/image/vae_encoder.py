import torch
import logging
from typing import Optional
import time
import gc
from weakref import WeakValueDictionary
from src.config.config import VAEEncoderConfig  # Import the new config class
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
from src.data.processors.utils.batch_utils import get_gpu_memory_usage
from transformers import AutoencoderKL

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
        
        # Move to device and set dtype
        self.vae.to(
            device=config.device,
            dtype=torch.float32  # VAE always uses float32 to avoid NaN losses
        )
        
        # Enable memory optimizations
        if config.enable_vae_slicing:
            self.vae.enable_slicing()
            
        if config.enable_memory_efficient_attention:
            if is_xformers_available():
                self.vae.enable_xformers_memory_efficient_attention()
            else:
                logger.warning("xformers not available, using standard attention")
                
        logger.info(
            f"Initialized VAE Encoder:\n"
            f"- Device: {config.device}\n"
            f"- Dtype: float32\n" 
            f"- Slicing: {config.enable_vae_slicing}\n"
            f"- Memory efficient attention: {config.enable_memory_efficient_attention}"
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
    async def encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE with optimized batch processing."""
        try:
            batch_stats = {
                'batch_size': pixel_values.shape[0],
                'start_memory': get_gpu_memory_usage(self.config.device),
                'start_time': time.time()
            }
            
            batch_size = pixel_values.shape[0]
            latents_list = []
            
            sub_batch_size = min(128, batch_size)
            
            compute_stream = torch.cuda.Stream()
            
            for i in range(0, batch_size, sub_batch_size):
                end_idx = min(i + sub_batch_size, batch_size)
                sub_batch = pixel_values[i:end_idx]
                
                with torch.cuda.stream(compute_stream):
                    latents = self.vae.encode(sub_batch).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents_list.append(latents)
                    
                # Update stats
                batch_stats.update({
                    'sub_batches_processed': i // sub_batch_size + 1,
                    'total_sub_batches': (batch_size + sub_batch_size - 1) // sub_batch_size
                })
                
            result = torch.cat(latents_list, dim=0)
            
            # Log final stats
            batch_stats.update({
                'end_memory': get_gpu_memory_usage(self.config.device),
                'duration': time.time() - batch_stats['start_time'],
                'memory_change': (
                    get_gpu_memory_usage(self.config.device) - 
                    batch_stats['start_memory']
                )
            })
            
            log_metrics(batch_stats, step=batch_stats['batch_size'], step_type="vae_encode")
            
            return result
            
        except Exception as e:
            log_error_with_context(e, "Error in VAE batch encoding")
            raise

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
