"""Ultra-optimized image grouping for multi-aspect ratio training."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import torch
from collections import defaultdict
import logging
from multiprocessing import Pool, Manager
from dataclasses import dataclass
from torch.cuda import amp

from .bucket_manager import Bucket, BucketManager

logger = logging.getLogger(__name__)

def _process_bucket(args: Tuple[Bucket, List[str], int]) -> List[ImageGroup]:
    """Standalone function for parallel bucket processing."""
    bucket, images, max_memory = args
    groups = []
    remaining = list(images)
    
    while remaining:
        # Calculate maximum possible group size
        group_size = min(bucket.batch_size, len(remaining))
        mem_required = bucket.width * bucket.height * 4 * group_size
        
        # Adjust group size based on memory constraints
        while mem_required > max_memory and group_size > 1:
            group_size -= 1
            mem_required = bucket.width * bucket.height * 4 * group_size
        
        if group_size < 1:
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
        
    return groups

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
    
    __slots__ = ('bucket_manager', '_groups', '_max_memory', 
                 '_stats', '_cache', '_num_workers')
    
    def __init__(
        self,
        bucket_manager: BucketManager,
        max_memory_gb: float = 32.0,
        num_workers: int = 4
    ):
        """Initialize with optimized defaults."""
        self.bucket_manager = bucket_manager
        manager = Manager()
        self._groups = manager.list()
        self._max_memory = int(max_memory_gb * 1024 * 1024 * 1024)
        self._stats = manager.dict()
        self._cache = manager.dict()
        self._num_workers = num_workers
    
    def create_groups(self, image_paths: List[str]) -> List[ImageGroup]:
        """Create optimal image groups using parallel processing."""
        # Group images by bucket
        bucket_images = self.bucket_manager.group_by_bucket(image_paths)
        
        # Prepare arguments for parallel processing
        process_args = [
            (bucket, images, self._max_memory)
            for bucket, images in bucket_images.items()
        ]
        
        # Process buckets in parallel
        with Pool(processes=self._num_workers) as pool:
            all_groups = []
            for groups in pool.map(_process_bucket, process_args):
                all_groups.extend(groups)
                self._stats['groups_created'] = \
                    self._stats.get('groups_created', 0) + len(groups)
        
        # Sort groups by memory efficiency
        all_groups.sort(
            key=lambda g: len(g.images) / g.total_memory,
            reverse=True
        )
        
        return all_groups
    
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
        return dict(self._stats)
    
    def clear(self) -> None:
        """Clear all groups and caches."""
        self._groups.clear()
        self._cache.clear()
        self._stats.clear()
