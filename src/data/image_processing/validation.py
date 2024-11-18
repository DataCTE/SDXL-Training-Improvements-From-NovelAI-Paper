"""Ultra-optimized image validation pipeline."""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Union, Optional, Dict
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import numba
from torch.cuda import amp
import logging
import os
import mmap
from pathlib import Path
import imghdr
import struct
from functools import lru_cache
import array

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationConfig:
    """Immutable validation configuration."""
    min_size: int = 512
    max_size: int = 4096
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    check_content: bool = True
    device: str = 'cuda'

@numba.jit(nopython=True)
def _fast_aspect_ratio(height: int, width: int) -> float:
    """Ultra-fast aspect ratio calculation."""
    return width / max(height, 1)

@numba.jit(nopython=True)
def _fast_size_check(height: int, width: int, min_size: int, max_size: int) -> bool:
    """Ultra-fast size validation."""
    return (min_size <= height <= max_size and 
            min_size <= width <= max_size)

class ImageValidator:
    """Ultra-optimized image validator."""
    
    __slots__ = ('config', '_executor', '_lock', '_stats', '_cache')
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        num_workers: int = 4
    ):
        """Initialize with optimized defaults."""
        self.config = config or ValidationConfig()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._lock = threading.RLock()
        self._stats = {'valid': 0, 'invalid': 0}
        self._cache = {}
    
    def _validate_size(self, height: int, width: int) -> bool:
        """Validate image dimensions."""
        return _fast_size_check(
            height, width,
            self.config.min_size,
            self.config.max_size
        )
    
    def _validate_aspect(self, height: int, width: int) -> bool:
        """Validate aspect ratio."""
        ar = _fast_aspect_ratio(height, width)
        return self.config.min_aspect <= ar <= self.config.max_aspect
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _check_content(img: np.ndarray) -> bool:
        """Ultra-fast content validation."""
        if img.ndim != 3:
            return False
            
        # Check for solid color or extreme values
        std = np.std(img)
        if std < 1.0:
            return False
            
        # Check for corrupted data
        if np.isnan(img).any() or np.isinf(img).any():
            return False
            
        return True
    
    def validate_image(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor, str]
    ) -> Tuple[bool, Dict[str, bool]]:
        """Validate a single image."""
        try:
            # Convert to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, torch.Tensor):
                if image.is_cuda:
                    image = image.cpu()
                image = image.numpy()
            elif isinstance(image, str):
                # Fast header read using memory mapping
                with open(image, 'rb') as f:
                    header = memoryview(f.read(32))
                    
                # Get format and validate
                fmt = imghdr.what(None, header)
                if fmt is None:
                    raise ImageValidationError("Unknown format")
                    
                # Get dimensions using format-specific parsing
                width = height = None
                if fmt == 'png':
                    width, height = struct.unpack('>II', header[16:24])
                elif fmt == 'gif':
                    width, height = struct.unpack('<HH', header[6:10])
                elif fmt == 'jpeg':
                    # JPEG requires full header scan
                    idx = 2
                    while idx < len(header):
                        while header[idx] == 0xFF:
                            idx += 1
                        ftype = header[idx]
                        if 0xC0 <= ftype <= 0xCF and ftype not in (0xC4, 0xC8, 0xCC):
                            height, width = struct.unpack('>HH', header[idx+3:idx+7])
                            break
                        idx += 1
                elif fmt == 'webp':
                    if header[12:16] == b'VP8 ':
                        width = struct.unpack('<H', header[26:28])[0] & 0x3FFF
                        height = struct.unpack('<H', header[28:30])[0] & 0x3FFF
                    elif header[12:16] == b'VP8L':
                        bits = header[21:25]
                        width = ((bits[1] & 0x3F) << 8) | bits[0]
                        height = ((bits[3] & 0xF) << 10) | (bits[2] << 2) | ((bits[1] & 0xC0) >> 6)
                        width += 1
                        height += 1
                
                if width is None or height is None:
                    raise ImageValidationError("Failed to extract dimensions")
                
                # Run validations
                size_valid = self._validate_size(height, width)
                aspect_valid = self._validate_aspect(height, width)
                
                # Only load full image if content check is needed
                content_valid = True
                if self.config.check_content:
                    img = cv2.imread(image, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ImageValidationError("Failed to load image")
                    content_valid = self._check_content(img)
                
                # Update stats
                is_valid = size_valid and aspect_valid and content_valid
                with self._lock:
                    self._stats['valid' if is_valid else 'invalid'] += 1
                
                return is_valid, {
                    'size': size_valid,
                    'aspect': aspect_valid,
                    'content': content_valid,
                    'width': width,
                    'height': height,
                    'format': fmt
                }
            
            # For numpy arrays, validate content directly
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                size_valid = self._validate_size(height, width)
                aspect_valid = self._validate_aspect(height, width)
                content_valid = not self.config.check_content or self._check_content(image)
                
                is_valid = size_valid and aspect_valid and content_valid
                with self._lock:
                    self._stats['valid' if is_valid else 'invalid'] += 1
                    
                return is_valid, {
                    'size': size_valid,
                    'aspect': aspect_valid,
                    'content': content_valid,
                    'width': width,
                    'height': height
                }
            
        except Exception as e:
            logger.debug(f"Validation failed: {str(e)}")
            return False, {
                'size': False,
                'aspect': False,
                'content': False,
                'error': str(e)
            }
    
    def validate_batch(
        self,
        images: list,
        batch_size: Optional[int] = None
    ) -> Tuple[list, list]:
        """Validate a batch of images in parallel."""
        if not images:
            return [], []
            
        batch_size = batch_size or len(images)
        
        # Process in parallel
        futures = [
            self._executor.submit(self.validate_image, img)
            for img in images[:batch_size]
        ]
        
        # Gather results
        results = [future.result() for future in futures]
        valid = [img for img, (is_valid, _) in zip(images, results) if is_valid]
        invalid = [img for img, (is_valid, _) in zip(images, results) if not is_valid]
        
        return valid, invalid
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._stats.clear()

class ImageValidationError(Exception):
    """Fast exception for image validation failures."""
    __slots__ = ()

# Global validator instance for common use
_default_validator = ImageValidator()

def validate_image(
    image: Union[np.ndarray, Image.Image, torch.Tensor, str],
    config: Optional[ValidationConfig] = None
) -> Tuple[bool, Dict[str, Union[bool, float, str, int]]]:
    """
    Validate an image using the default validator instance.
    
    Args:
        image: Image to validate (numpy array, PIL Image, torch Tensor, or path)
        config: Optional validation configuration
        
    Returns:
        Tuple of (is_valid, validation_details)
    """
    global _default_validator
    if config is not None:
        _default_validator = ImageValidator(config)
    return _default_validator.validate_image(image)
