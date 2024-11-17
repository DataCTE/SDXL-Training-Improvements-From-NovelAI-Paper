"""Utility functions for image processing and data manipulation.

This module provides utilities for efficient image processing, focusing on
conversions between PyTorch tensors and PIL Images with caching and
thread-safety. It also includes utilities for image statistics and
normalization.

The module provides:
- Thread-safe image conversion between tensors and PIL Images
- Caching for improved performance
- Normalization and denormalization utilities
- Image statistics calculation
- Memory-efficient operations

Classes:
    ImageProcessingError: Base exception for image processing errors
    ImageConverter: Thread-safe image conversion with caching
    
Functions:
    tensor_to_pil: Convert tensor to PIL Image
    pil_to_tensor: Convert PIL Image to tensor
    get_image_stats: Calculate image normalization statistics
"""

import logging
import traceback
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Base exception for image processing related errors."""


class ConversionError(ImageProcessingError):
    """Exception raised when image conversion fails."""


class ValidationError(ImageProcessingError):
    """Exception raised when image validation fails."""


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
    
    Attributes:
        _cache_size: Maximum number of items in each cache
        _tensor_to_pil_cache: Cache for tensor to PIL conversions
        _pil_to_tensor_cache: Cache for PIL to tensor conversions
        _lock: Thread lock for cache access
        
    Note:
        The caches use a FIFO policy when full.
    """
    
    def __init__(self, cache_size: int = 1024) -> None:
        """Initialize the converter.
        
        Args:
            cache_size: Maximum number of items in each cache
            
        Raises:
            ValueError: If cache_size is not positive
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
        """Generate hash for tensor caching.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Hash value for the tensor
            
        Note:
            Uses the tensor's data pointer for hashing
        """
        return hash(tensor.data_ptr())

    @staticmethod
    def _hash_image(image: Image.Image) -> int:
        """Generate hash for PIL image caching.
        
        Args:
            image: Input PIL image
            
        Returns:
            Hash value for the image
            
        Note:
            Uses the image's byte representation for hashing
        """
        return hash(image.tobytes())
    
    @staticmethod
    def _validate_tensor(tensor: torch.Tensor) -> None:
        """Validate tensor for conversion.
        
        Args:
            tensor: Input tensor to validate
            
        Raises:
            ValidationError: If tensor format is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError("Input must be a torch.Tensor")
            
        if len(tensor.shape) not in (3, 4):
            raise ValidationError(
                "Tensor must have 3 or 4 dimensions (C,H,W) or (B,C,H,W)"
            )
            
        if len(tensor.shape) == 3 and tensor.shape[0] not in (1, 3, 4):
            raise ValidationError(
                "Tensor must have 1, 3, or 4 channels"
            )
    
    @staticmethod
    def _validate_image(image: Image.Image) -> None:
        """Validate PIL image for conversion.
        
        Args:
            image: Input PIL image to validate
            
        Raises:
            ValidationError: If image format is invalid
        """
        if not isinstance(image, Image.Image):
            raise ValidationError("Input must be a PIL.Image")
            
        if image.mode not in ('L', 'RGB', 'RGBA'):
            raise ValidationError(
                "Image must be in L, RGB, or RGBA mode"
            )

    def tensor_to_pil(
        self,
        tensor: torch.Tensor,
        normalize: bool = True,
        denormalize: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> Image.Image:
        """Convert a torch tensor to PIL Image with enhanced functionality.
        
        This method provides efficient conversion from PyTorch tensors to
        PIL Images with optional normalization and denormalization.
        
        Args:
            tensor: Input tensor (C,H,W) or (B,C,H,W)
            normalize: Whether to normalize to [0,255]
            denormalize: Whether to denormalize using mean/std
            mean: Channel means for denormalization
            std: Channel stds for denormalization
            
        Returns:
            Converted PIL Image
            
        Raises:
            ValidationError: If input format is invalid
            ConversionError: If conversion fails
            
        Note:
            If both normalize and denormalize are True, denormalization
            is applied first, followed by normalization to [0,255].
        """
        try:
            # Validate input
            self._validate_tensor(tensor)
            
            # Check cache first
            tensor_hash = self._hash_tensor(tensor)
            with self._lock:
                if tensor_hash in self._tensor_to_pil_cache:
                    return self._tensor_to_pil_cache[tensor_hash]

            # Process tensor
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # Take first image if batched
            
            # Move to CPU and detach
            tensor = tensor.cpu().detach()
            
            # Denormalize if requested
            if denormalize and mean is not None and std is not None:
                if len(mean) != tensor.shape[0] or len(std) != tensor.shape[0]:
                    raise ValidationError(
                        "mean and std must match number of channels"
                    )
                mean_t = torch.tensor(mean, device=tensor.device)[:, None, None]
                std_t = torch.tensor(std, device=tensor.device)[:, None, None]
                tensor = tensor * std_t + mean_t
            
            # Convert to numpy and correct format
            tensor = tensor.permute(1, 2, 0).numpy()
            
            # Normalize to [0,255] if requested
            if normalize:
                tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
            
            # Convert to PIL
            if tensor.shape[2] == 1:
                image = Image.fromarray(tensor[:, :, 0], mode='L')
            elif tensor.shape[2] == 3:
                image = Image.fromarray(tensor, mode='RGB')
            else:  # tensor.shape[2] == 4
                image = Image.fromarray(tensor, mode='RGBA')
            
            # Update cache
            with self._lock:
                if len(self._tensor_to_pil_cache) >= self._cache_size:
                    self._tensor_to_pil_cache.pop(next(iter(self._tensor_to_pil_cache)))
                self._tensor_to_pil_cache[tensor_hash] = image
            
            return image
            
        except ValidationError:
            raise
        except Exception as error:
            logger.error("Failed to convert tensor to PIL: %s", str(error))
            logger.error(traceback.format_exc())
            raise ConversionError("Failed to convert tensor to PIL") from error

    def pil_to_tensor(
        self,
        image: Image.Image,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Convert PIL Image to torch tensor with enhanced functionality.
        
        This method provides efficient conversion from PIL Images to PyTorch
        tensors with optional normalization and device/dtype specification.
        
        Args:
            image: Input PIL image
            normalize: Whether to normalize to [0,1]
            device: Target device for tensor
            dtype: Target dtype for tensor
            
        Returns:
            Converted tensor on specified device/dtype
            
        Raises:
            ValidationError: If input format is invalid
            ConversionError: If conversion fails
            
        Note:
            The output tensor will have shape (B,C,H,W) where B=1.
        """
        try:
            # Validate input
            self._validate_image(image)
            
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
            
        except ValidationError:
            raise
        except Exception as error:
            logger.error("Failed to convert PIL to tensor: %s", str(error))
            logger.error(traceback.format_exc())
            raise ConversionError("Failed to convert PIL to tensor") from error


# Create global instance
converter = ImageConverter()


def tensor_to_pil(
    tensor: torch.Tensor,
    **kwargs: Union[bool, Optional[Tuple[float, ...]]]
) -> Image.Image:
    """Convenience function for tensor to PIL conversion.
    
    This is a wrapper around ImageConverter.tensor_to_pil using a global
    converter instance.
    
    Args:
        tensor: Input tensor to convert
        **kwargs: Additional arguments for conversion
        
    Returns:
        Converted PIL Image
        
    See Also:
        ImageConverter.tensor_to_pil for full documentation
    """
    return converter.tensor_to_pil(tensor, **cast(Dict, kwargs))


def pil_to_tensor(
    image: Image.Image,
    **kwargs: Union[bool, Optional[torch.device], Optional[torch.dtype]]
) -> torch.Tensor:
    """Convenience function for PIL to tensor conversion.
    
    This is a wrapper around ImageConverter.pil_to_tensor using a global
    converter instance.
    
    Args:
        image: Input PIL image to convert
        **kwargs: Additional arguments for conversion
        
    Returns:
        Converted tensor
        
    See Also:
        ImageConverter.pil_to_tensor for full documentation
    """
    return converter.pil_to_tensor(image, **cast(Dict, kwargs))


@lru_cache(maxsize=128)
def get_image_stats(size: Tuple[int, int]) -> ImageStats:
    """Calculate optimal normalization stats for given image size.
    
    This function calculates normalization statistics based on image size,
    using a reference area of 1024x1024 pixels.
    
    Args:
        size: Image size as (width, height)
        
    Returns:
        ImageStats object containing normalization statistics
        
    Note:
        Results are cached for improved performance.
    """
    area = size[0] * size[1]
    base_area = 1024 * 1024
    scale = (area / base_area) ** 0.25
    
    mean = (0.5 * scale,) * 3
    std = (0.5 / scale,) * 3
    min_val = (0.0,) * 3
    max_val = (1.0,) * 3
    
    return ImageStats(mean=mean, std=std, min_val=min_val, max_val=max_val)
