"""
Image validation and error handling for SDXL training pipeline.

This module provides validation utilities and custom exceptions for
image processing operations.
"""

import logging
import torch
from PIL import Image
from typing import Tuple

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Base exception for image processing errors."""


class CUDAProcessingError(ProcessingError):
    """Exception raised for CUDA and memory-related errors."""


class InputValidationError(ProcessingError):
    """Exception raised for invalid input data."""


class ModelProcessingError(ProcessingError):
    """Exception raised for model-related errors."""


class ConversionError(ProcessingError):
    """Exception raised when image conversion fails."""


class ValidationError(ProcessingError):
    """Exception raised when image validation fails."""


def validate_tensor(tensor: torch.Tensor) -> None:
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


def validate_image(image: Image.Image) -> None:
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


def validate_dimensions(
    width: int,
    height: int,
    min_size: int = 1024,
    max_size: int = 2048
) -> None:
    """Validate image dimensions.
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        
    Raises:
        ValidationError: If dimensions are invalid
    """
    if width < min_size or height < min_size:
        raise ValidationError(
            f"Image dimensions must be at least {min_size}x{min_size}"
        )
    if width > max_size or height > max_size:
        raise ValidationError(
            f"Image dimensions must not exceed {max_size}x{max_size}"
        )


def validate_target_size(target_size: Tuple[int, int]) -> None:
    """Validate target size for resizing.
    
    Args:
        target_size: Target size as (width, height)
        
    Raises:
        ValidationError: If target size is invalid
    """
    if not all(s > 0 for s in target_size):
        raise ValidationError("Target size dimensions must be positive")
