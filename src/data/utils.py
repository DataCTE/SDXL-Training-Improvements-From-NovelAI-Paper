import torch
import numpy as np
from PIL import Image
import logging
from typing import Union, Optional, Tuple, List
from functools import lru_cache
from threading import Lock
import traceback

logger = logging.getLogger(__name__)

class ImageConverter:
    """Thread-safe image conversion utilities with caching."""
    
    def __init__(self, cache_size: int = 1024):
        self._cache_size = cache_size
        self._tensor_to_pil_cache = {}
        self._pil_to_tensor_cache = {}
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
        """
        Convert a torch tensor to PIL Image with enhanced functionality.
        
        Args:
            tensor: Input tensor (C,H,W) or (B,C,H,W)
            normalize: Whether to normalize to [0,255]
            denormalize: Whether to denormalize using mean/std
            mean: Channel means for denormalization
            std: Channel stds for denormalization
        """
        try:
            # Check cache first
            tensor_hash = self._hash_tensor(tensor)
            with self._lock:
                if tensor_hash in self._tensor_to_pil_cache:
                    return self._tensor_to_pil_cache[tensor_hash]

            # Process tensor
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # Take first image if batched
            
            # Move to CPU and correct format
            tensor = tensor.cpu().detach()
            
            # Denormalize if requested
            if denormalize and mean is not None and std is not None:
                mean = torch.tensor(mean, device=tensor.device)[:, None, None]
                std = torch.tensor(std, device=tensor.device)[:, None, None]
                tensor = tensor * std + mean
            
            # Convert to numpy and correct format
            tensor = tensor.permute(1, 2, 0).numpy()
            
            # Normalize to [0,255] if requested
            if normalize:
                tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
            
            # Convert to PIL
            image = Image.fromarray(tensor)
            
            # Update cache
            with self._lock:
                if len(self._tensor_to_pil_cache) >= self._cache_size:
                    self._tensor_to_pil_cache.pop(next(iter(self._tensor_to_pil_cache)))
                self._tensor_to_pil_cache[tensor_hash] = image
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert tensor to PIL: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def pil_to_tensor(
        self,
        image: Image.Image,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Convert PIL Image to torch tensor with enhanced functionality.
        
        Args:
            image: Input PIL image
            normalize: Whether to normalize to [0,1]
            device: Target device for tensor
            dtype: Target dtype for tensor
        """
        try:
            # Check cache first
            image_hash = self._hash_image(image)
            with self._lock:
                if image_hash in self._pil_to_tensor_cache:
                    tensor = self._pil_to_tensor_cache[image_hash]
                    if device is not None:
                        tensor = tensor.to(device)
                    if dtype is not None:
                        tensor = tensor.to(dtype)
                    return tensor

            # Convert to numpy
            np_image = np.array(image)
            
            # Handle grayscale images
            if len(np_image.shape) == 2:
                np_image = np_image[:, :, None]
            
            # Normalize if requested
            if normalize:
                np_image = np_image.astype(np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(np_image)
            tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
            
            # Move to device/dtype if specified
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            
            # Update cache
            with self._lock:
                if len(self._pil_to_tensor_cache) >= self._cache_size:
                    self._pil_to_tensor_cache.pop(next(iter(self._pil_to_tensor_cache)))
                self._pil_to_tensor_cache[image_hash] = tensor
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to convert PIL to tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Create global instance
converter = ImageConverter()

# Convenience functions that use the global instance
def tensor_to_pil(tensor: torch.Tensor, **kwargs) -> Image.Image:
    """Convenience function for tensor to PIL conversion."""
    return converter.tensor_to_pil(tensor, **kwargs)

def pil_to_tensor(image: Image.Image, **kwargs) -> torch.Tensor:
    """Convenience function for PIL to tensor conversion."""
    return converter.pil_to_tensor(image, **kwargs)

@lru_cache(maxsize=128)
def get_image_stats(size: Tuple[int, int]) -> Tuple[float, float]:
    """Calculate optimal normalization stats for given image size."""
    area = size[0] * size[1]
    base_area = 1024 * 1024
    scale = (area / base_area) ** 0.25
    return (0.5 * scale, 0.5 / scale)