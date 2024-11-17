"""
Image conversion utilities for SDXL training pipeline.

This module provides thread-safe conversion between PyTorch tensors and PIL Images
with caching for improved performance. It also includes utilities for image
statistics and normalization.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from src.data.image_processing.validation import validate_tensor, validate_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageStats:
    """Statistics for image normalization.
    
    Attributes:
        mean: Mean values per channel
        std: Standard deviation values per channel
        min_val: Minimum pixel values per channel
        max_val: Maximum pixel values per channel
    """
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    min_val: Tuple[float, ...]
    max_val: Tuple[float, ...]


class ImageConverter:
    """Thread-safe image conversion utilities with caching.
    
    This class provides efficient and thread-safe conversion between PyTorch
    tensors and PIL Images with caching for improved performance.
    """
    
    def __init__(self, cache_size: int = 1024) -> None:
        """Initialize the converter.
        
        Args:
            cache_size: Maximum number of items in each cache
        """
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
            
        self._cache_size = cache_size
        self._tensor_to_pil_cache: Dict[int, Image.Image] = {}
        self._pil_to_tensor_cache: Dict[int, torch.Tensor] = {}
        self._lock = Lock()
        
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._tensor_to_pil_cache.clear()
            self._pil_to_tensor_cache.clear()

    @staticmethod
    def _hash_tensor(tensor: torch.Tensor) -> int:
        """Generate hash for tensor caching."""
        return hash(tensor.data_ptr())

    @staticmethod
    def _hash_image(image: Image.Image) -> int:
        """Generate hash for PIL image caching."""
        return hash(image.tobytes())
    
    def tensor_to_pil(
        self,
        tensor: torch.Tensor,
        normalize: bool = True,
        denormalize: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> Image.Image:
        """Convert a torch tensor to PIL Image.
        
        Args:
            tensor: Input tensor (C,H,W) or (B,C,H,W)
            normalize: Whether to normalize to [0,255]
            denormalize: Whether to denormalize using mean/std
            mean: Channel means for denormalization
            std: Channel stds for denormalization
            
        Returns:
            Converted PIL Image
        """
        validate_tensor(tensor)
        
        # Use cache if available
        tensor_hash = self._hash_tensor(tensor)
        with self._lock:
            if tensor_hash in self._tensor_to_pil_cache:
                return self._tensor_to_pil_cache[tensor_hash]
        
        # Process tensor
        if tensor.ndim == 4:
            tensor = tensor[0]  # Take first batch item
            
        if denormalize and mean is not None and std is not None:
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
                
        if normalize:
            tensor = tensor.mul(255).clamp(0, 255).byte()
            
        # Convert to PIL
        image = Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())
        
        # Cache result
        with self._lock:
            if len(self._tensor_to_pil_cache) >= self._cache_size:
                self._tensor_to_pil_cache.pop(next(iter(self._tensor_to_pil_cache)))
            self._tensor_to_pil_cache[tensor_hash] = image
            
        return image
    
    def pil_to_tensor(
        self,
        image: Image.Image,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """Convert PIL Image to torch tensor.
        
        Args:
            image: Input PIL image
            device: Target device for tensor
            dtype: Target dtype for tensor
            normalize: Whether to normalize to [0,1]
            
        Returns:
            Converted tensor
        """
        validate_image(image)
        
        # Use cache if available
        image_hash = self._hash_image(image)
        with self._lock:
            if image_hash in self._pil_to_tensor_cache:
                tensor = self._pil_to_tensor_cache[image_hash]
                if device is not None:
                    tensor = tensor.to(device)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                return tensor
        
        # Convert to tensor
        tensor = torch.from_numpy(
            image.convert('RGB').__array__()
        ).permute(2, 0, 1)
        
        if normalize:
            tensor = tensor.float().div(255)
            
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
            
        # Cache result
        with self._lock:
            if len(self._pil_to_tensor_cache) >= self._cache_size:
                self._pil_to_tensor_cache.pop(next(iter(self._pil_to_tensor_cache)))
            self._pil_to_tensor_cache[image_hash] = tensor
            
        return tensor


@lru_cache(maxsize=128)
def get_image_stats(size: Tuple[int, int]) -> ImageStats:
    """Calculate optimal normalization stats for given image size.
    
    Args:
        size: Image size as (width, height)
        
    Returns:
        ImageStats object containing normalization statistics
    """
    # Calculate stats based on reference area of 1024x1024
    ref_area = 1024 * 1024
    actual_area = size[0] * size[1]
    scale = (actual_area / ref_area) ** 0.5
    
    return ImageStats(
        mean=(0.5, 0.5, 0.5),
        std=(0.5 / scale, 0.5 / scale, 0.5 / scale),
        min_val=(0.0, 0.0, 0.0),
        max_val=(1.0, 1.0, 1.0)
    )


# Create global instance
converter = ImageConverter()
