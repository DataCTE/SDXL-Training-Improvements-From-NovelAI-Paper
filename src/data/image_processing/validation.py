"""
Image validation and error handling for SDXL training pipeline.

This module provides validation utilities and custom exceptions for
image processing operations.
"""

import logging
import torch
from PIL import Image
from typing import Tuple, Optional

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


class ImageValidationError(ValidationError):
    """Exception raised for image validation failures."""


class TensorValidationError(ValidationError):
    """Exception raised for tensor validation failures."""


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
    max_size: int = 4096
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


def validate_image_comprehensive(
    image: Image.Image,
    min_size: int = 512,
    max_size: int = 4096,
    required_mode: str = 'RGB'
) -> None:
    """Validate a PIL Image.
    
    Args:
        image: PIL Image to validate
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        required_mode: Required image mode (e.g., 'RGB')
        
    Raises:
        ImageValidationError: If validation fails
    """
    if not isinstance(image, Image.Image):
        raise ImageValidationError(f"Expected PIL Image, got {type(image)}")
        
    if image.mode != required_mode:
        raise ImageValidationError(
            f"Invalid image mode: {image.mode}, expected {required_mode}"
        )
        
    width, height = image.size
    if width < min_size or height < min_size:
        raise ImageValidationError(
            f"Image too small: {width}x{height}, minimum size is {min_size}"
        )
        
    if width > max_size or height > max_size:
        raise ImageValidationError(
            f"Image too large: {width}x{height}, maximum size is {max_size}"
        )
        
    try:
        # Verify image data is valid
        image.verify()
    except Exception as e:
        raise ImageValidationError(f"Invalid image data: {str(e)}")


def validate_tensor_comprehensive(
    tensor: torch.Tensor,
    expected_dims: int = 4,
    expected_channels: int = 3,
    min_value: float = -1.0,
    max_value: float = 1.0
) -> None:
    """Validate a PyTorch tensor.
    
    Args:
        tensor: Input tensor to validate
        expected_dims: Expected number of dimensions
        expected_channels: Expected number of channels
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Raises:
        TensorValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"Expected torch.Tensor, got {type(tensor)}")
        
    if tensor.dim() != expected_dims:
        raise TensorValidationError(
            f"Invalid tensor dimensions: {tensor.dim()}, expected {expected_dims}"
        )
        
    if tensor.shape[1] != expected_channels:
        raise TensorValidationError(
            f"Invalid number of channels: {tensor.shape[1]}, expected {expected_channels}"
        )
        
    if tensor.isnan().any():
        raise TensorValidationError("Tensor contains NaN values")
        
    if tensor.isinf().any():
        raise TensorValidationError("Tensor contains infinite values")
        
    if tensor.min() < min_value:
        raise TensorValidationError(
            f"Tensor contains values below minimum: {tensor.min()}, minimum is {min_value}"
        )
        
    if tensor.max() > max_value:
        raise TensorValidationError(
            f"Tensor contains values above maximum: {tensor.max()}, maximum is {max_value}"
        )


def get_tensor_stats(tensor: torch.Tensor) -> Tuple[float, float, float, float]:
    """Get basic statistics for a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tuple of (min, max, mean, std)
    """
    return (
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
        float(tensor.std())
    )


def check_image_corruption(image_path: str) -> Optional[str]:
    """Check if an image file is corrupted.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Error message if corrupted, None if valid
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return None
    except Exception as e:
        return str(e)
