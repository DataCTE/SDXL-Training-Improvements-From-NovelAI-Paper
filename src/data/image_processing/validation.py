"""
Image validation and error handling for SDXL training pipeline.

This module provides validation utilities and custom exceptions for
image processing operations.
"""

import logging
import torch
from PIL import Image
from typing import Tuple, Optional
import os
import numpy as np
import mmap
from pathlib import Path
import imghdr
import struct

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


def get_image_size_fast(filepath: str) -> Tuple[int, int]:
    """Get image dimensions without fully loading the file.
    
    Uses file headers to quickly extract dimensions for supported formats.
    Supports JPEG, PNG, GIF, BMP, WEBP.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ImageValidationError: If format not supported or dimensions can't be read
    """
    with open(filepath, 'rb') as fhandle:
        head = fhandle.read(32)
        if len(head) < 24:
            raise ImageValidationError("Invalid image file")
            
        format = imghdr.what(None, head)
        if format == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                raise ImageValidationError('Invalid PNG file')
            width, height = struct.unpack('>ii', head[16:24])
        elif format == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif format == 'jpeg':
            try:
                fhandle.seek(0)
                size = 2
                ftype = 0
                while not 0xC0 <= ftype <= 0xCF or ftype in (0xC4, 0xC8, 0xCC):
                    fhandle.seek(size, 1)
                    while ord(fhandle.read(1)) == 0xFF:
                        ftype = ord(fhandle.read(1))
                        size = struct.unpack('>H', fhandle.read(2))[0] - 2
                fhandle.seek(1, 1)
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:
                raise ImageValidationError("Invalid JPEG file")
        elif format == 'webp':
            if head[12:16] == b'VP8 ':
                width = struct.unpack('<H', head[26:28])[0] & 0x3FFF
                height = struct.unpack('<H', head[28:30])[0] & 0x3FFF
            elif head[12:16] == b'VP8L':
                bits = head[21:25]
                width = ((bits[1] & 0x3F) << 8) | bits[0]
                height = ((bits[3] & 0xF) << 10) | (bits[2] << 2) | ((bits[1] & 0xC0) >> 6)
                width += 1
                height += 1
            else:
                raise ImageValidationError("Unsupported WebP format")
        else:
            # Fallback to PIL for other formats
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except Exception as e:
                raise ImageValidationError(f"Unable to determine image size: {str(e)}")
                
        return width, height

def validate_image_size(width: int, height: int, min_size: int = 512, max_size: int = 4096) -> None:
    """Fast validation of image dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
    
    Raises:
        ImageValidationError: If dimensions invalid
    """
    if not (min_size <= width <= max_size and min_size <= height <= max_size):
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) outside allowed range "
            f"[{min_size}-{max_size}]"
        )

def quick_corruption_check(filepath: str, max_header_size: int = 32768) -> None:
    """Quickly check if an image file appears corrupted.
    
    Performs fast checks on file headers without loading entire file.
    
    Args:
        filepath: Path to image file
        max_header_size: Maximum bytes to check
        
    Raises:
        ImageValidationError: If file appears corrupted
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(max_header_size)
            
        # Check magic numbers for common formats
        if header.startswith(b'\xFF\xD8\xFF'):  # JPEG
            if not any(b'\xFF\xDA' in header, b'\xFF\xD9' in header):  # Look for SOS/EOI
                raise ImageValidationError("Incomplete JPEG file")
        elif header.startswith(b'\x89PNG\r\n\x1A\n'):  # PNG
            if b'IEND' not in header and len(header) >= max_header_size:
                raise ImageValidationError("PNG header validation failed")
        elif header.startswith(b'GIF8'):  # GIF
            if header[4:6] not in (b'7a', b'9a'):
                raise ImageValidationError("Invalid GIF version")
        elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':  # WebP
            if len(header) < 30:
                raise ImageValidationError("Incomplete WebP header")
        else:
            # For other formats, check if it's a known image type
            if not imghdr.what(None, header):
                raise ImageValidationError("Unknown image format")
                
    except OSError as e:
        raise ImageValidationError(f"File read error: {str(e)}")

def validate_image(image_path: str, min_size: int = 512, max_size: int = 4096) -> Tuple[int, int]:
    """Optimized image validation.
    
    Performs fast validation using file headers and minimal file reading.
    
    Args:
        image_path: Path to image file
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
    
    Returns:
        Tuple of (width, height)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ImageValidationError: If validation fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Quick corruption check on headers
    quick_corruption_check(image_path)
    
    # Fast dimension check
    try:
        width, height = get_image_size_fast(image_path)
        validate_image_size(width, height, min_size, max_size)
        return width, height
    except Exception as e:
        raise ImageValidationError(f"Validation failed: {str(e)}")

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

def validate_tensor_comprehensive(tensor: torch.Tensor) -> None:
    """Validate tensor format and values.
    
    Args:
        tensor: Input tensor to validate
        
    Raises:
        ValidationError: If tensor invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError("Input must be a torch.Tensor")
        
    if tensor.dim() != 3:
        raise ValidationError("Tensor must have 3 dimensions (C,H,W)")
        
    if tensor.size(0) not in (1, 3, 4):
        raise ValidationError("Tensor must have 1, 3, or 4 channels")
        
    if not torch.isfinite(tensor).all():
        raise ValidationError("Tensor contains non-finite values")

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
