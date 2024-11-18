"""Ultra-optimized image grouping for multi-aspect ratio training."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import torch
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from torch.cuda import amp
import heapq

from .bucket_manager import Bucket, BucketManager

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ImageGroup:
    """Immutable image group configuration."""
    bucket: Bucket
    images: Tuple[str, ...]
    total_memory: int
    
    def __post_init__(self):
        """Validate group configuration."""
        if not self.images:
            raise ValueError("Image group cannot be empty")
        if self.total_memory <= 0:
            raise ValueError("Total memory must be positive")

class ImageGrouper:
    """Ultra-optimized image grouper for multi-aspect ratio training."""
    
    __slots__ = ('bucket_manager', '_groups', '_lock', '_executor',
                 '_max_memory', '_stats', '_cache')
    
    def __init__(
        self,
        bucket_manager: BucketManager,
        max_memory_gb: float = 32.0,
        num_workers: int = 4
    ):
        """Initialize with optimized defaults."""
        self.bucket_manager = bucket_manager
        self._groups: List[ImageGroup] = []
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._max_memory = int(max_memory_gb * 1024 * 1024 * 1024)  # Convert to bytes
        self._stats = defaultdict(int)
        self._cache = {}
        
    def _estimate_memory(self, bucket: Bucket, num_images: int) -> int:
        """Estimate memory usage for a group of images."""
        # Basic memory estimation (4 bytes per pixel for float32)
        return bucket.width * bucket.height * 4 * num_images
    
    def _can_fit_in_memory(self, memory_required: int) -> bool:
        """Check if memory requirement can be met."""
        return memory_required <= self._max_memory
    
    def create_groups(self, image_paths: List[str]) -> List[ImageGroup]:
        """Create optimal image groups using parallel processing."""
        # Group images by bucket
        bucket_images = self.bucket_manager.group_by_bucket(image_paths)
        
        # Process each bucket in parallel
        futures = [
            self._executor.submit(self._process_bucket, bucket, images)
            for bucket, images in bucket_images.items()
        ]
        
        # Gather results
        all_groups = []
        for future in futures:
            groups = future.result()
            all_groups.extend(groups)
        
        # Sort groups by memory efficiency (descending)
        all_groups.sort(
            key=lambda g: len(g.images) / g.total_memory,
            reverse=True
        )
        
        return all_groups
    
    def _process_bucket(
        self,
        bucket: Bucket,
        images: List[str]
    ) -> List[ImageGroup]:
        """Process a single bucket of images."""
        groups = []
        remaining = list(images)
        
        while remaining:
            # Try to create maximum size group
            group_size = min(
                bucket.batch_size,
                len(remaining)
            )
            
            # Check memory constraint
            mem_required = self._estimate_memory(bucket, group_size)
            while not self._can_fit_in_memory(mem_required) and group_size > 1:
                group_size -= 1
                mem_required = self._estimate_memory(bucket, group_size)
            
            if group_size < 1:
                logger.warning(
                    f"Cannot fit even one image from bucket {bucket} in memory"
                )
                break
            
            # Create group
            group_images = tuple(remaining[:group_size])
            group = ImageGroup(
                bucket=bucket,
                images=group_images,
                total_memory=mem_required
            )
            
            groups.append(group)
            remaining = remaining[group_size:]
            self._stats['groups_created'] += 1
        
        return groups
    
    def optimize_groups(
        self,
        groups: List[ImageGroup],
        target_memory_usage: float = 0.9
    ) -> List[ImageGroup]:
        """Optimize groups for maximum memory efficiency."""
        target_memory = int(self._max_memory * target_memory_usage)
        optimized = []
        current_memory = 0
        
        # Use heap for efficient group selection
        heap = [(-(len(g.images) / g.total_memory), i, g)
                for i, g in enumerate(groups)]
        heapq.heapify(heap)
        
        while heap:
            _, _, group = heapq.heappop(heap)
            if current_memory + group.total_memory <= target_memory:
                optimized.append(group)
                current_memory += group.total_memory
                self._stats['groups_optimized'] += 1
            else:
                # Try to split group if needed
                if group.total_memory > target_memory - current_memory:
                    new_size = int(
                        (target_memory - current_memory)
                        / (group.total_memory / len(group.images))
                    )
                    if new_size >= 1:
                        new_memory = self._estimate_memory(
                            group.bucket, new_size
                        )
                        new_group = ImageGroup(
                            bucket=group.bucket,
                            images=group.images[:new_size],
                            total_memory=new_memory
                        )
                        optimized.append(new_group)
                        self._stats['groups_split'] += 1
                break
        
        return optimized
    
    def get_stats(self) -> Dict[str, int]:
        """Get grouping statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear(self) -> None:
        """Clear all groups and caches."""
        with self._lock:
            self._groups.clear()
            self._cache.clear()
            self._stats.clear()
