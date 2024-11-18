"""
Ultra-optimized bucket management for multi-aspect ratio training.

This module implements an ultra-optimized bucket management system for
multi-aspect ratio training. It uses pre-computed lookup tables, parallel
processing, and efficient memory usage to achieve high performance.

Classes:
    Bucket: Immutable bucket configuration for fast hashing and comparison
    BucketManager: Ultra-optimized bucket manager for multi-aspect ratio training
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import torch
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from torch.cuda import amp

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Bucket:
    """Immutable bucket configuration for fast hashing and comparison."""
    width: int
    height: int
    batch_size: int
    
    def __post_init__(self):
        """Validate bucket dimensions."""
        if self.width <= 0 or self.height <= 0 or self.batch_size <= 0:
            raise ValueError("Bucket dimensions must be positive")
    
    @property
    def aspect_ratio(self) -> float:
        """Fast aspect ratio calculation."""
        return self.width / self.height
    
    @property
    def resolution(self) -> int:
        """Fast resolution calculation."""
        return self.width * self.height

class BucketManager:
    """Ultra-optimized bucket manager for multi-aspect ratio training."""
    
    __slots__ = ('_buckets', '_image_buckets', '_lock', '_executor',
                 '_max_resolution', '_batch_sizes', '_stats', '_bucket_cache')
    
    def __init__(self, max_resolution: int = 1024 * 1024,
                 min_batch_size: int = 1, max_batch_size: int = 32,
                 num_workers: int = 4):
        """Initialize with pre-computed lookup tables."""
        self._buckets: Set[Bucket] = set()
        self._image_buckets: Dict[str, Bucket] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._max_resolution = max_resolution
        self._batch_sizes = range(min_batch_size, max_batch_size + 1)
        self._stats = defaultdict(int)
        
        # Pre-compute bucket cache for common resolutions
        self._bucket_cache = {}
        self._precompute_buckets()
    
    def _precompute_buckets(self) -> None:
        """Pre-compute common bucket configurations."""
        common_sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        for w in common_sizes:
            for h in common_sizes:
                if w * h <= self._max_resolution:
                    key = (w, h)
                    self._bucket_cache[key] = self._compute_optimal_bucket(w, h)
    
    def _compute_optimal_bucket(self, width: int, height: int) -> Bucket:
        """Compute optimal bucket configuration using vectorized operations."""
        resolution = width * height
        if resolution > self._max_resolution:
            scale = np.sqrt(self._max_resolution / resolution)
            width = int(width * scale)
            height = int(height * scale)
        
        # Vectorized batch size computation
        mem_per_image = width * height * 4  # Assume 4 bytes per pixel
        batch_sizes = np.array(self._batch_sizes)
        total_mem = mem_per_image * batch_sizes
        valid_sizes = batch_sizes[total_mem <= self._max_resolution * 4]
        
        if len(valid_sizes) == 0:
            return Bucket(width, height, 1)
        
        return Bucket(width, height, valid_sizes[-1])
    
    def add_image(self, image_path: str, width: int, height: int) -> None:
        """Add image to bucket system with minimal locking."""
        with self._lock:
            if image_path in self._image_buckets:
                return
            
            # Try cache first
            key = (width, height)
            bucket = self._bucket_cache.get(key)
            if bucket is None:
                bucket = self._compute_optimal_bucket(width, height)
                self._bucket_cache[key] = bucket
            
            self._buckets.add(bucket)
            self._image_buckets[image_path] = bucket
            self._stats['images_added'] += 1
    
    def get_bucket(self, image_path: str) -> Optional[Bucket]:
        """Ultra-fast bucket lookup."""
        return self._image_buckets.get(image_path)
    
    def get_all_buckets(self) -> Set[Bucket]:
        """Get all unique buckets."""
        return self._buckets.copy()
    
    def get_bucket_for_resolution(self, width: int, height: int) -> Bucket:
        """Get optimal bucket for resolution with caching."""
        key = (width, height)
        bucket = self._bucket_cache.get(key)
        if bucket is None:
            bucket = self._compute_optimal_bucket(width, height)
            self._bucket_cache[key] = bucket
        return bucket
    
    def group_by_bucket(self, image_paths: List[str]) -> Dict[Bucket, List[str]]:
        """Group images by bucket using parallel processing."""
        # Split work into chunks
        chunk_size = max(1, len(image_paths) // (self._executor._max_workers * 4))
        chunks = [image_paths[i:i + chunk_size] 
                 for i in range(0, len(image_paths), chunk_size)]
        
        def process_chunk(paths: List[str]) -> Dict[Bucket, List[str]]:
            groups: Dict[Bucket, List[str]] = defaultdict(list)
            for path in paths:
                bucket = self._image_buckets.get(path)
                if bucket is not None:
                    groups[bucket].append(path)
            return groups
        
        # Process chunks in parallel
        futures = [self._executor.submit(process_chunk, chunk) 
                  for chunk in chunks]
        
        # Merge results efficiently
        result: Dict[Bucket, List[str]] = defaultdict(list)
        for future in futures:
            chunk_result = future.result()
            for bucket, paths in chunk_result.items():
                result[bucket].extend(paths)
        
        return dict(result)
    
    def get_stats(self) -> Dict[str, int]:
        """Get bucket statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats['total_buckets'] = len(self._buckets)
            stats['total_images'] = len(self._image_buckets)
            return stats
    
    def clear(self) -> None:
        """Clear all buckets."""
        with self._lock:
            self._buckets.clear()
            self._image_buckets.clear()
            self._stats.clear()
    
    def __len__(self) -> int:
        """Get total number of images."""
        return len(self._image_buckets)
