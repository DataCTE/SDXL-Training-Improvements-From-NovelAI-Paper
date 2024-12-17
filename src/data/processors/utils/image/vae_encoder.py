import torch
import logging
from typing import Optional
from dataclasses import dataclass
import asyncio
import gc
from weakref import WeakValueDictionary

from src.data.processors.utils.batch_utils import adjust_batch_size
from src.data.processors.utils.system_utils import get_gpu_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class VAEEncoderConfig:
    """Configuration for VAE encoding."""
    device: torch.device = torch.device('cuda')
    dtype: torch.dtype = torch.float16
    enable_memory_efficient_attention: bool = True
    enable_vae_slicing: bool = True
    vae_batch_size: int = 8
    max_memory_usage: float = 0.9

class VAEEncoder:
    """Handles VAE encoding with memory optimization and async support."""
    
    def __init__(self, vae, config: VAEEncoderConfig):
        """Initialize VAE encoder."""
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
            
        logger.info(
            f"Initialized VAEEncoder:\n"
            f"- Device: {config.device}\n"
            f"- Dtype: {config.dtype}\n"
            f"- Batch size: {config.vae_batch_size}\n"
            f"- Memory efficient attention: {config.enable_memory_efficient_attention}\n"
            f"- VAE slicing: {config.enable_vae_slicing}"
        )
        
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
    def encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE with optimized batch processing."""
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        try:
            # Increased sub-batch size but still memory efficient
            sub_batch_size = min(8, batch_size)  # Increased from 4 to 8
            
            # Pre-allocate CUDA stream for better GPU utilization
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for i in range(0, batch_size, sub_batch_size):
                    sub_batch = pixel_values[i:i+sub_batch_size]
                    
                    try:
                        # Overlap CPU->GPU transfer with computation
                        sub_batch = sub_batch.to(
                            self.config.device, 
                            non_blocking=True
                        )
                        
                        # Encode with automatic mixed precision
                        with torch.cuda.amp.autocast(dtype=self.config.dtype):
                            latents = self.vae.encode(sub_batch).latent_dist.sample()
                            latents = latents * self.vae.config.scaling_factor
                            
                        # Asynchronous CPU transfer
                        latents_list.append(latents.cpu(non_blocking=True))
                        
                        # Cleanup without blocking
                        del latents
                        del sub_batch
                        
                    except RuntimeError as e:
                        logger.error(f"Error encoding sub-batch {i}: {str(e)[:200]}...")
                        for _ in range(len(sub_batch)):
                            latents_list.append(torch.zeros(
                                (1, 4, pixel_values.shape[2]//8, pixel_values.shape[3]//8),
                                dtype=self.config.dtype,
                                device='cpu'
                            ))
                    
                    # Less frequent but more thorough cleanup
                    if i % (sub_batch_size * 4) == 0:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
            
            # Wait for all operations to complete
            torch.cuda.synchronize()
            
            # Combine results efficiently
            result = torch.cat(latents_list, dim=0)
            del latents_list
            
            # Final result to GPU
            return result.to(self.config.device, non_blocking=True)
            
        except Exception as e:
            logger.error(f"Error in batch encoding: {str(e)[:200]}...")
            if 'latents_list' in locals():
                del latents_list
            torch.cuda.empty_cache()
            raise

    async def encode_image(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode a single image through VAE asynchronously."""
        try:
            # Add batch dimension and ensure on CPU initially
            image_tensor = image_tensor.unsqueeze(0).cpu()
            
            # Encode through VAE
            latents = await asyncio.to_thread(
                self.encode_batch,
                image_tensor
            )
            
            # Remove batch dimension and cleanup input
            result = latents.squeeze(0)
            del image_tensor
            del latents
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error encoding through VAE: {str(e)[:200]}...")
            # Cleanup on error
            if 'image_tensor' in locals():
                del image_tensor
            if 'latents' in locals():
                del latents
            gc.collect()
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
