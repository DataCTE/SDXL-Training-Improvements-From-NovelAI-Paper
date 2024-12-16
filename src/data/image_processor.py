# src/data/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
from dataclasses import dataclass
import numpy as np
from src.data import thread_config

@dataclass
class ImageProcessorConfig:
    dtype: torch.dtype = torch.float16
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5)
    device: torch.device = torch.device('cuda')

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.config = config
        self.transform = self._build_transform()
        
        # Pre-allocate reusable tensors
        self.buffer_size = (32, 3, 1024, 1024)  # Adjustable based on max expected size
        self.tensor_buffer = torch.empty(self.buffer_size, 
                                       dtype=self.config.dtype, 
                                       device=self.config.device)
        self.chunk_size = thread_config.chunk_size

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

        # Process images in place
        output = self.tensor_buffer[:batch_size, :, :height, :width]
        
        for idx, img in enumerate(images):
            img = self._resize_image(img, width, height)
            with torch.cuda.amp.autocast():
                output[idx].copy_(self.transform(img))

        return output