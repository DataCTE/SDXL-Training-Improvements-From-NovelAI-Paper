"""
Ultra-optimized image transformation pipeline.

This module provides a collection of image transformations commonly used
in the training pipeline, with support for both PIL Images and tensors.
"""

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
from typing import Tuple, Optional, Dict, List
import kornia
from PIL import Image
import cv2
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import numba
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Define interpolation modes as enum for type safety
class InterpolationMode(str, Enum):
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    LANCZOS = 'lanczos'

@dataclass(frozen=True)
class TransformConfig:
    """Immutable transform configuration."""
    target_size: Tuple[int, int]
    flip_prob: float = 0.0
    interpolation: InterpolationMode = InterpolationMode.BICUBIC
    normalize: bool = True
    to_float: bool = True
    device: str = 'cuda'

@numba.jit(nopython=True, parallel=True)
def _fast_normalize(img: np.ndarray) -> np.ndarray:
    """Ultra-fast normalization using Numba."""
    return (img.astype(np.float32) - 127.5) / 127.5

class ImageTransformPipeline:
    """Ultra-optimized image transformation pipeline."""
    
    __slots__ = ('config', '_executor', '_lock', '_cache', '_stats')
    
    # Define interpolation modes mapping
    _INTERPOLATION_MODES = {
        InterpolationMode.NEAREST: cv2.INTER_NEAREST,
        InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
        InterpolationMode.BICUBIC: cv2.INTER_CUBIC,
        InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4
    }
    
    def __init__(
        self,
        config: TransformConfig,
        num_workers: int = 4,
        cache_size: int = 1024
    ):
        """Initialize with optimized defaults."""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._lock = threading.RLock()
        self._cache = {}
        self._stats = {'cache_hits': 0, 'cache_misses': 0}
    
    @staticmethod
    @lru_cache(maxsize=8)
    def _get_interpolation(mode: InterpolationMode) -> int:
        """Get cached interpolation mode."""
        return ImageTransformPipeline._INTERPOLATION_MODES.get(
            mode, 
            cv2.INTER_CUBIC
        )
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Optimized image resizing."""
        if img.shape[:2] == self.config.target_size:
            return img
            
        interpolation = self._get_interpolation(self.config.interpolation)
        return cv2.resize(
            img,
            self.config.target_size[::-1],
            interpolation=interpolation
        )
    
    @staticmethod
    @numba.jit(nopython=True)
    def _fast_flip(img: np.ndarray) -> np.ndarray:
        """Ultra-fast horizontal flip using Numba."""
        return np.ascontiguousarray(np.fliplr(img))
    
    def _process_single(self, img: np.ndarray) -> torch.Tensor:
        """Process a single image with optimized operations."""
        # Resize
        img = self._resize_image(img)
        
        # Random flip
        if self.config.flip_prob > 0 and np.random.random() < self.config.flip_prob:
            img = self._fast_flip(img)
        
        # Normalize and convert to float
        if self.config.normalize:
            img = _fast_normalize(img)
        elif self.config.to_float:
            img = img.astype(np.float32) / 255.0
            
        # Convert to tensor with memory pinning
        with torch.cuda.amp.autocast():
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            if self.config.device == 'cuda':
                tensor = tensor.pin_memory().to(
                    device='cuda',
                    non_blocking=True
                )
        
        return tensor
    
    def process_batch(
        self,
        images: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Process a batch of images in parallel."""
        if not images:
            raise ValueError("Empty image batch")
            
        batch_size = batch_size or len(images)
        
        # Process in parallel
        futures = [
            self._executor.submit(self._process_single, img)
            for img in images[:batch_size]
        ]
        
        # Gather results maintaining order
        tensors = [future.result() for future in futures]
        
        # Stack efficiently
        with torch.cuda.amp.autocast():
            batch = torch.stack(tensors, dim=0)
        
        return batch
    
    @torch.no_grad()
    def augment_batch(
        self,
        batch: torch.Tensor,
        strength: float = 0.75
    ) -> torch.Tensor:
        """Apply efficient augmentations to a batch."""
        with torch.cuda.amp.autocast():
            # Color jitter
            batch = kornia.enhance.adjust_brightness(
                batch,
                torch.rand(batch.size(0), device=batch.device) * 0.2 + 0.9
            )
            batch = kornia.enhance.adjust_contrast(
                batch,
                torch.rand(batch.size(0), device=batch.device) * 0.2 + 0.9
            )
            
            # Random noise
            if strength > 0:
                noise = torch.randn_like(batch) * (strength * 0.1)
                batch = batch + noise
                batch = batch.clamp(-1, 1)
        
        return batch
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._stats.clear()


def resize_image(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int],
    method: str = 'bilinear'
) -> Union[Image.Image, torch.Tensor]:
    """Resize image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        method: Interpolation method
        
    Returns:
        Resized image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return image.resize(size[::-1], getattr(Image, method.upper()))
    else:
        # validate_tensor(image)
        return F.interpolate(image, size, mode=method, align_corners=False)


def random_crop(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int]
) -> Union[Image.Image, torch.Tensor]:
    """Randomly crop image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        
    Returns:
        Cropped image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        i = np.random.randint(0, image.height - size[0])
        j = np.random.randint(0, image.width - size[1])
        return image.crop((j, i, j + size[1], i + size[0]))
    else:
        # validate_tensor(image)
        return F.interpolate(image, size, mode='nearest', align_corners=False)


def center_crop(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int]
) -> Union[Image.Image, torch.Tensor]:
    """Center crop image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        
    Returns:
        Cropped image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return F.center_crop(image, size)
    else:
        # validate_tensor(image)
        return F.center_crop(image, size)


def random_flip(
    image: Union[Image.Image, torch.Tensor],
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Randomly flip image horizontally.
    
    Args:
        image: Input image (PIL or tensor)
        p: Probability of flipping
        
    Returns:
        Flipped image
    """
    if np.random.random() < p:
        if isinstance(image, Image.Image):
            # validate_image(image)
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            # validate_tensor(image)
            return F.hflip(image)
    return image


def random_rotate(
    image: Union[Image.Image, torch.Tensor],
    angle_range: Tuple[float, float],
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Randomly rotate image within angle range.
    
    Args:
        image: Input image (PIL or tensor)
        angle_range: (min_angle, max_angle) in degrees
        p: Probability of rotating
        
    Returns:
        Rotated image
    """
    if np.random.random() < p:
        angle = np.random.uniform(*angle_range)
        if isinstance(image, Image.Image):
            # validate_image(image)
            return image.rotate(angle, expand=True)
        else:
            # validate_tensor(image)
            return F.rotate(image, angle)
    return image


def adjust_brightness(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image brightness.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Brightness adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return F.adjust_brightness(image, factor)
    else:
        # validate_tensor(image)
        return F.adjust_brightness(image, factor)


def adjust_contrast(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image contrast.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Contrast adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return F.adjust_contrast(image, factor)
    else:
        # validate_tensor(image)
        return F.adjust_contrast(image, factor)


def adjust_saturation(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image saturation.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Saturation adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return F.adjust_saturation(image, factor)
    else:
        # validate_tensor(image)
        return F.adjust_saturation(image, factor)


def random_jitter(
    image: Union[Image.Image, torch.Tensor],
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Apply random color jittering.
    
    Args:
        image: Input image (PIL or tensor)
        brightness: Max brightness adjustment
        contrast: Max contrast adjustment
        saturation: Max saturation adjustment
        hue: Max hue adjustment
        p: Probability of applying jitter
        
    Returns:
        Jittered image
    """
    if np.random.random() < p:
        if isinstance(image, Image.Image):
            # validate_image(image)
            return F.adjust_brightness(
                F.adjust_contrast(
                    F.adjust_saturation(
                        F.adjust_hue(
                            image,
                            np.random.uniform(-hue, hue)
                        ),
                        1 + np.random.uniform(-saturation, saturation)
                    ),
                    1 + np.random.uniform(-contrast, contrast)
                ),
                1 + np.random.uniform(-brightness, brightness)
            )
        else:
            # validate_tensor(image)
            return F.adjust_brightness(
                F.adjust_contrast(
                    F.adjust_saturation(
                        F.adjust_hue(
                            image,
                            np.random.uniform(-hue, hue)
                        ),
                        1 + np.random.uniform(-saturation, saturation)
                    ),
                    1 + np.random.uniform(-contrast, contrast)
                ),
                1 + np.random.uniform(-brightness, brightness)
            )
    return image


def gaussian_blur(
    image: Union[Image.Image, torch.Tensor],
    kernel_size: int,
    sigma: Optional[float] = None
) -> Union[Image.Image, torch.Tensor]:
    """Apply Gaussian blur.
    
    Args:
        image: Input image (PIL or tensor)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Blurred image
    """
    if isinstance(image, Image.Image):
        # validate_image(image)
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
    else:
        # validate_tensor(image)
        return F.gaussian_blur(image, kernel_size, sigma)
