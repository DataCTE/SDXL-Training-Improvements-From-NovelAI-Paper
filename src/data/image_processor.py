# src/data/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List, Optional
from torchvision import transforms
from dataclasses import dataclass
import numpy as np
import logging
import torch.nn.functional as F
from src.data.utils import (
    get_gpu_memory_usage,
    adjust_batch_size,
    create_thread_pool,
    get_optimal_workers
)
from src.data.bucket import BucketManager

logger = logging.getLogger(__name__)

@dataclass
class ImageProcessorConfig:
    dtype: torch.dtype = torch.float16
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5)
    device: torch.device = torch.device('cuda')
    enable_memory_efficient_attention: bool = True
    enable_vae_slicing: bool = True
    vae_batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    max_memory_usage: float = 0.9
    max_image_size: Tuple[int, int] = (8192, 8192)
    min_image_size: Tuple[int, int] = (256, 256)

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig, bucket_manager: Optional[BucketManager] = None):
        """Initialize with optional bucket manager for resolution-aware processing."""
        self.config = config
        self.bucket_manager = bucket_manager
        
        # Initialize thread pool for parallel processing
        self.num_workers = min(
            self.config.num_workers,
            get_optimal_workers(memory_per_worker_gb=1.0)  # 1GB per worker for image processing
        )
        self.executor = create_thread_pool(self.num_workers)
        
        # Pre-allocate reusable tensors on GPU
        self.buffer_size = (32, 3, 8192, 8192)  # Adjustable based on max expected size
        self.tensor_buffer = torch.empty(self.buffer_size, 
                                      dtype=self.config.dtype, 
                                      device=self.config.device)
        
        logger.info(
            f"Initialized ImageProcessor:\n"
            f"- Device: {config.device}\n"
            f"- Dtype: {config.dtype}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Buffer size: {self.buffer_size}"
        )

    def _build_transform(self):
        """Build optimized transform pipeline."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3]),  # Ensure 3 channels
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std),
            transforms.Lambda(lambda x: x.to(self.config.dtype))
        ])

    @staticmethod
    def _resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
        """Optimized resize operation."""
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img

    def _process_single_image(self, img: Image.Image, width: int, height: int) -> torch.Tensor:
        """Process a single image with GPU acceleration."""
        img = self._resize_image(img, width, height)
        
        # Transform directly to GPU with mixed precision
        with torch.cuda.amp.autocast(dtype=self.config.dtype):
            img_tensor = self.transform(img).to(self.config.device, non_blocking=True)
        
        return img_tensor

    def _adjust_buffer_size(self, batch_size: int, width: int, height: int):
        """Adjust tensor buffer size if needed."""
        if (batch_size > self.buffer_size[0] or 
            width > self.buffer_size[2] or 
            height > self.buffer_size[3]):
            
            self.buffer_size = (
                max(batch_size, self.buffer_size[0]), 
                3,
                max(width, self.buffer_size[2]), 
                max(height, self.buffer_size[3])
            )
            
            # Reallocate buffer with new size
            self.tensor_buffer = torch.empty(
                self.buffer_size,
                dtype=self.config.dtype,
                device=self.config.device
            )
            logger.debug(f"Resized tensor buffer to {self.buffer_size}")

    @torch.no_grad()
    def process_batch(self, images: List[Image.Image], width: int, height: int) -> torch.Tensor:
        """Process a batch of images with optimized parallel processing and GPU utilization."""
        batch_size = len(images)
        if batch_size == 0:
            return torch.empty(0, dtype=self.config.dtype, device=self.config.device)
        
        # Adjust buffer size if needed
        self._adjust_buffer_size(batch_size, width, height)
        
        # Get output tensor view
        output = self.tensor_buffer[:batch_size, :, :height, :width]
        
        # Process images in parallel with GPU acceleration
        futures = []
        for i, img in enumerate(images):
            future = self.executor.submit(self._process_single_image, img, width, height)
            futures.append((i, future))
        
        # Collect results efficiently
        for i, future in futures:
            try:
                img_tensor = future.result()
                output[i].copy_(img_tensor, non_blocking=True)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Fill with zeros on error
                output[i].zero_()
        
        return output

    @torch.no_grad()
    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE with optimized memory usage."""
        if self.config.enable_memory_efficient_attention:
            vae.enable_xformers_memory_efficient_attention()
        
        if self.config.enable_vae_slicing:
            vae.enable_slicing()
        
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        # Process in smaller batches to manage memory
        for i in range(0, batch_size, self.config.vae_batch_size):
            try:
                batch_slice = slice(i, min(i + self.config.vae_batch_size, batch_size))
                current_batch = pixel_values[batch_slice]
                
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    latents = vae.encode(current_batch).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
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
                    return self.encode_vae(vae, pixel_values)
                raise
        
        # Combine all batches
        return torch.cat(latents_list, dim=0)

    def _get_optimal_size(self, width: int, height: int) -> Tuple[int, int]:
        """Get optimal size considering bucket constraints if available."""
        if self.bucket_manager is not None:
            bucket = self.bucket_manager.find_bucket(width, height)
            if bucket is not None:
                return bucket.width, bucket.height
        
        # Default sizing logic
        return self._default_resize(width, height)