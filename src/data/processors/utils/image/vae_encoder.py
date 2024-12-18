import torch
import logging
from typing import Optional
import time
import gc
from weakref import WeakValueDictionary
from src.config.config import VAEEncoderConfig  # Import the new config class
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
from src.data.processors.utils.batch_utils import get_gpu_memory_usage

logger = logging.getLogger(__name__)

class VAEEncoder:
    """Handles VAE encoding with memory optimization and async support."""
    
    def __init__(self, vae, config: VAEEncoderConfig):
        """Initialize VAE encoder with consolidated config."""
        try:
            self.vae = vae
            self.config = config
            self._tensor_cache = WeakValueDictionary()
            
            # Configure VAE
            self.vae = self.vae.to(self.config.device, self.config.dtype)
            self.vae.eval()
            
            # Enable optimizations
            if self.config.enable_memory_efficient_attention:
                self._enable_memory_efficient_attention()
                
            if self.config.enable_vae_slicing:
                self.vae.enable_slicing()
                
            # Log initialization
            logger.info(
                f"Initialized VAEEncoder:\n"
                f"- Device: {config.device}\n"
                f"- Dtype: {config.dtype}\n"
                f"- Batch size: {config.vae_batch_size}\n"
                f"- Memory efficient attention: {config.enable_memory_efficient_attention}\n"
                f"- VAE slicing: {config.enable_vae_slicing}"
            )
            log_system_metrics(prefix="VAE Encoder initialization: ")
            
        except Exception as e:
            log_error_with_context(e, "Error initializing VAE encoder")
            raise

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
