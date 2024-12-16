# src/data/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
from dataclasses import dataclass
import numpy as np
from src.data.thread_config import get_optimal_cpu_threads
import logging

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

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.config = config
        self.transform = self._build_transform()
        
        # Pre-allocate reusable tensors on GPU
        self.buffer_size = (32, 3, 1024, 1024)  # Adjustable based on max expected size
        self.tensor_buffer = torch.empty(self.buffer_size, 
                                       dtype=self.config.dtype, 
                                       device=self.config.device)
        self.chunk_size = get_optimal_cpu_threads().chunk_size
        
        logger.info(f"Initialized image processor on {self.config.device} with {self.config.dtype}")

    def _build_transform(self):
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

    @torch.no_grad()
    def process_batch(self, images: List[Image.Image], width: int, height: int) -> torch.Tensor:
        """Process a batch of images with optimized memory usage."""
        batch_size = len(images)
        
        # Resize tensor buffer if needed
        if batch_size > self.buffer_size[0] or width > self.buffer_size[2] or height > self.buffer_size[3]:
            self.buffer_size = (max(batch_size, self.buffer_size[0]), 3, 
                              max(width, self.buffer_size[2]), 
                              max(height, self.buffer_size[3]))
            self.tensor_buffer = torch.empty(self.buffer_size, 
                                           dtype=self.config.dtype, 
                                           device=self.config.device)

        # Process images in place with GPU acceleration
        output = self.tensor_buffer[:batch_size, :, :height, :width]
        
        # Process in smaller batches to avoid OOM
        for i in range(0, batch_size, self.config.vae_batch_size):
            batch_slice = slice(i, min(i + self.config.vae_batch_size, batch_size))
            current_batch = images[batch_slice]
            
            # Process each image in the mini-batch
            for j, img in enumerate(current_batch):
                img = self._resize_image(img, width, height)
                # Transform directly to GPU
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    img_tensor = self.transform(img).to(self.config.device)
                    output[i + j].copy_(img_tensor)

        return output

    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE with memory efficient settings."""
        if self.config.enable_memory_efficient_attention:
            vae.enable_xformers_memory_efficient_attention()
        
        if self.config.enable_vae_slicing:
            vae.enable_slicing()
        
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        # Process in smaller batches
        for i in range(0, batch_size, self.config.vae_batch_size):
            batch_slice = slice(i, min(i + self.config.vae_batch_size, batch_size))
            current_batch = pixel_values[batch_slice]
            
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                latents = vae.encode(current_batch).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents_list.append(latents)
        
        # Combine all batches
        return torch.cat(latents_list, dim=0)