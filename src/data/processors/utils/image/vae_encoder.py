import torch
import logging
from typing import Optional
from dataclasses import dataclass
import asyncio

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
        """Initialize VAE encoder.
        
        Args:
            vae: VAE model instance
            config: Configuration for VAE encoding
        """
        self.vae = vae
        self.config = config
        
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

    def _enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention if possible."""
        try:
            import xformers
            self.vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention for VAE")
        except ImportError:
            logger.warning("xformers not available, using standard attention")

    @torch.no_grad()
    def encode_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images through VAE.
        
        Args:
            pixel_values: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            VAE encoded latents
        """
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        # Process in smaller batches to manage memory
        for i in range(0, batch_size, self.config.vae_batch_size):
            try:
                batch_slice = slice(i, min(i + self.config.vae_batch_size, batch_size))
                current_batch = pixel_values[batch_slice]
                
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    latents = self.vae.encode(current_batch).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents_list.append(latents)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Reduce batch size and retry
                    self.config.vae_batch_size = adjust_batch_size(
                        current_batch_size=self.config.vae_batch_size,
                        max_batch_size=32,
                        min_batch_size=1,
                        current_memory_usage=get_gpu_memory_usage(self.config.device),
                        max_memory_usage=self.config.max_memory_usage
                    )
                    logger.warning(f"GPU OOM, reducing VAE batch size to {self.config.vae_batch_size}")
                    torch.cuda.empty_cache()
                    # Retry with smaller batch size
                    return self.encode_batch(pixel_values)
                raise
        
        # Combine all batches
        return torch.cat(latents_list, dim=0)

    async def encode_image(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode a single image through VAE asynchronously.
        
        Args:
            image_tensor: Tensor of shape (channels, height, width)
            
        Returns:
            VAE encoded latent or None if encoding fails
        """
        try:
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Encode through VAE
            latents = await asyncio.to_thread(
                self.encode_batch,
                image_tensor
            )
            
            # Remove batch dimension
            return latents.squeeze(0)
            
        except Exception as e:
            logger.error(f"Error encoding through VAE: {str(e)[:200]}...")
            return None

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear CUDA cache
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up VAE encoder resources")
            
        except Exception as e:
            logger.error(f"Error during VAE encoder cleanup: {str(e)}")
