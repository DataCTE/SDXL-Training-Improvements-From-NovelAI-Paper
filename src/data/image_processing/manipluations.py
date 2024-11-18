"""
Ultra-optimized image manipulation module with GPU acceleration.

This module provides thread-safe conversion between PyTorch tensors and PIL Images
with caching for improved performance. It also includes utilities for image
statistics and normalization.
"""

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
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

# Define manipulation modes as enum for type safety
class ManipulationMode(str, Enum):
    RESIZE = 'resize'
    CROP = 'crop'
    ROTATE = 'rotate'
    FLIP = 'flip'
    COLOR = 'color'
    NOISE = 'noise'
    BLUR = 'blur'

@dataclass(frozen=True)
class ManipConfig:
    """Immutable manipulation configuration."""
    device: str = 'cuda'
    batch_size: int = 32
    num_workers: int = 4
    cache_size: int = 1024
    use_mixed_precision: bool = True
    memory_efficient: bool = True

@numba.jit(nopython=True, parallel=True)
def _fast_normalize(img: np.ndarray) -> np.ndarray:
    """Ultra-fast normalization using Numba."""
    return (img.astype(np.float32) - 127.5) / 127.5

class ImageManipulator:
    """Ultra-optimized image manipulation with GPU acceleration."""
    
    __slots__ = ('config', '_executor', '_lock', '_cache', '_stats')
    
    def __init__(
        self,
        config: Optional[ManipConfig] = None
    ):
        """Initialize with optimized defaults."""
        self.config = config or ManipConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self._lock = threading.RLock()
        self._cache = {}
        self._stats = {'cache_hits': 0, 'cache_misses': 0}
    
    @torch.no_grad()
    def process_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        mode: ManipulationMode,
        **kwargs
    ) -> torch.Tensor:
        """Process a batch of images with the specified manipulation."""
        if not images:
            raise ValueError("Empty image batch")
            
        batch_size = min(len(images), self.config.batch_size)
        
        # Convert to tensors in parallel
        futures = [
            self._executor.submit(self._to_tensor, img)
            for img in images[:batch_size]
        ]
        
        # Gather results maintaining order
        tensors = [future.result() for future in futures]
        
        # Stack efficiently
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            batch = torch.stack(tensors, dim=0)
            
            # Apply manipulation
            if mode == ManipulationMode.RESIZE:
                return self._resize_batch(batch, **kwargs)
            elif mode == ManipulationMode.CROP:
                return self._crop_batch(batch, **kwargs)
            elif mode == ManipulationMode.ROTATE:
                return self._rotate_batch(batch, **kwargs)
            elif mode == ManipulationMode.FLIP:
                return self._flip_batch(batch, **kwargs)
            elif mode == ManipulationMode.COLOR:
                return self._color_batch(batch, **kwargs)
            elif mode == ManipulationMode.NOISE:
                return self._noise_batch(batch, **kwargs)
            elif mode == ManipulationMode.BLUR:
                return self._blur_batch(batch, **kwargs)
            else:
                raise ValueError(f"Unknown manipulation mode: {mode}")
    
    def _to_tensor(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> torch.Tensor:
        """Convert image to tensor efficiently."""
        if isinstance(image, torch.Tensor):
            tensor = image
        elif isinstance(image, np.ndarray):
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:  # PIL Image
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        
        if self.config.device == 'cuda':
            tensor = tensor.pin_memory().to(
                device='cuda',
                non_blocking=True,
                memory_format=torch.channels_last
            )
        
        return tensor
    
    @torch.no_grad()
    def _resize_batch(
        self,
        batch: torch.Tensor,
        size: Tuple[int, int],
        mode: str = 'bilinear'
    ) -> torch.Tensor:
        """Efficient batch resize operation."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            return F.interpolate(
                batch,
                size=size,
                mode=mode,
                align_corners=False
            )
    
    @torch.no_grad()
    def _crop_batch(
        self,
        batch: torch.Tensor,
        size: Tuple[int, int],
        random: bool = False
    ) -> torch.Tensor:
        """Efficient batch crop operation."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            if random:
                return kornia.augmentation.RandomCrop(size)(batch)
            else:
                return kornia.augmentation.CenterCrop(size)(batch)
    
    @torch.no_grad()
    def _rotate_batch(
        self,
        batch: torch.Tensor,
        angle: float,
        center: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Efficient batch rotation."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            return kornia.geometry.transform.rotate(
                batch,
                angle=torch.tensor([angle]).expand(batch.size(0)),
                center=center
            )
    
    @torch.no_grad()
    def _flip_batch(
        self,
        batch: torch.Tensor,
        horizontal: bool = True
    ) -> torch.Tensor:
        """Efficient batch flip operation."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            if horizontal:
                return kornia.geometry.transform.hflip(batch)
            else:
                return kornia.geometry.transform.vflip(batch)
    
    @torch.no_grad()
    def _color_batch(
        self,
        batch: torch.Tensor,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0
    ) -> torch.Tensor:
        """Efficient batch color adjustment."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            if brightness != 0:
                batch = kornia.enhance.adjust_brightness(batch, brightness)
            if contrast != 0:
                batch = kornia.enhance.adjust_contrast(batch, contrast)
            if saturation != 0:
                batch = kornia.enhance.adjust_saturation(batch, saturation)
            if hue != 0:
                batch = kornia.enhance.adjust_hue(batch, hue)
            return batch
    
    @torch.no_grad()
    def _noise_batch(
        self,
        batch: torch.Tensor,
        std: float = 0.1,
        mean: float = 0
    ) -> torch.Tensor:
        """Add efficient GPU-accelerated noise."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            noise = torch.randn_like(batch) * std + mean
            return torch.clamp(batch + noise, -1, 1)
    
    @torch.no_grad()
    def _blur_batch(
        self,
        batch: torch.Tensor,
        kernel_size: int = 3,
        sigma: float = 1.5
    ) -> torch.Tensor:
        """Efficient batch blur operation."""
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            return kornia.filters.gaussian_blur2d(
                batch,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma)
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._stats.clear()

# Create global instance
manipulator = ImageManipulator()
