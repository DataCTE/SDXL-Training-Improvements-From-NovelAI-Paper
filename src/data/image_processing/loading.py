"""
Ultra-optimized image loading pipeline.

This module provides utilities for loading and verifying images from files,
with support for various formats and error handling.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import numba
from torch.cuda import amp
import logging
import mmap
from functools import lru_cache
import io

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LoaderConfig:
    """Immutable loader configuration."""
    to_rgb: bool = True
    to_tensor: bool = True
    device: str = 'cuda'
    cache_size: int = 1024

class ImageLoader:
    """Ultra-optimized image loader."""
    
    __slots__ = ('config', '_executor', '_lock', '_cache', '_stats')
    
    def __init__(
        self,
        config: Optional[LoaderConfig] = None,
        num_workers: int = 4
    ):
        """Initialize with optimized defaults."""
        self.config = config or LoaderConfig()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._lock = threading.RLock()
        self._cache = {}
        self._stats = {'cache_hits': 0, 'cache_misses': 0}
    
    @staticmethod
    @lru_cache(maxsize=8)
    def _get_color_mode(mode: str) -> int:
        """Get cached color mode."""
        modes = {
            'L': cv2.COLOR_GRAY2RGB,
            'LA': cv2.COLOR_GRAY2RGB,
            'RGB': -1,
            'RGBA': cv2.COLOR_RGBA2RGB
        }
        return modes.get(mode, cv2.COLOR_BGR2RGB)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load single image with optimized operations."""
        try:
            # Try cache first
            if image_path in self._cache:
                with self._lock:
                    self._stats['cache_hits'] += 1
                return self._cache[image_path]
            
            with self._lock:
                self._stats['cache_misses'] += 1
            
            # Fast loading with cv2
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                # Fallback to PIL for other formats
                with Image.open(image_path) as pil_img:
                    img = np.array(pil_img)
                    
                    # Convert color space if needed
                    if self.config.to_rgb:
                        mode = self._get_color_mode(pil_img.mode)
                        if mode != -1:
                            img = cv2.cvtColor(img, mode)
            
            # Cache result
            if len(self._cache) < self.config.cache_size:
                with self._lock:
                    self._cache[image_path] = img
            
            return img
            
        except Exception as e:
            logger.debug(f"Failed to load image {image_path}: {str(e)}")
            raise
    
    def _process_loaded(
        self,
        img: np.ndarray,
        to_tensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Process loaded image."""
        if not to_tensor:
            return img
            
        # Convert to tensor with memory pinning
        with torch.cuda.amp.autocast():
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
                
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            if self.config.device == 'cuda':
                tensor = tensor.pin_memory().to(
                    device='cuda',
                    non_blocking=True
                )
        
        return tensor
    
    def load_image(
        self,
        image_path: str,
        to_tensor: Optional[bool] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Load a single image."""
        img = self._load_image(image_path)
        return self._process_loaded(
            img,
            to_tensor if to_tensor is not None else self.config.to_tensor
        )
    
    def load_batch(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        to_tensor: Optional[bool] = None
    ) -> Union[List[np.ndarray], torch.Tensor]:
        """Load a batch of images in parallel."""
        if not image_paths:
            return []
            
        batch_size = batch_size or len(image_paths)
        to_tensor = to_tensor if to_tensor is not None else self.config.to_tensor
        
        # Load in parallel
        futures = [
            self._executor.submit(self._load_image, path)
            for path in image_paths[:batch_size]
        ]
        
        # Gather results
        images = [future.result() for future in futures]
        
        if not to_tensor:
            return images
            
        # Convert to tensors
        tensors = [
            self._process_loaded(img, to_tensor=True)
            for img in images
        ]
        
        # Stack efficiently
        with torch.cuda.amp.autocast():
            batch = torch.stack(tensors, dim=0)
        
        return batch
    
    def get_stats(self) -> Dict[str, int]:
        """Get loader statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._cache.clear()
            self._stats.clear()