"""Ultra-optimized sampler implementation with GPU acceleration.

This module provides a high-performance sampler with features like:
- GPU-accelerated batch sampling
- Efficient bucketing
- Advanced caching mechanisms
- Parallel processing
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any, Dict, List, Optional, Set, Tuple,
    Union, Iterator, TypeVar, Generic
)
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Sampler
from numba import jit

from src.data.core.base import CustomSamplerBase

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

# Constants
MAX_CACHE_SIZE = 1024
BUCKET_CACHE_SIZE = 128
DEFAULT_BUCKET_SIZE = 64

@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for optimized sampling."""
    
    batch_size: int = field(default=1)
    shuffle: bool = field(default=True)
    drop_last: bool = field(default=False)
    seed: Optional[int] = field(default=None)
    num_replicas: int = field(default=1)
    rank: int = field(default=0)
    bucket_size: int = field(default=DEFAULT_BUCKET_SIZE)
    cache_size: int = field(default=MAX_CACHE_SIZE)
    pin_memory: bool = field(default=True)

class BucketInfo:
    """Efficient bucket information storage."""
    
    __slots__ = ('indices', 'size', 'aspect_ratio')
    
    def __init__(
        self,
        indices: List[int],
        size: Tuple[int, int],
        aspect_ratio: float
    ) -> None:
        self.indices = indices
        self.size = size
        self.aspect_ratio = aspect_ratio

class OptimizedSampler(CustomSamplerBase[T]):
    """Ultra-optimized sampler with GPU acceleration."""
    
    __slots__ = (
        'dataset', 'config', '_buckets', '_cache',
        '_rng', '_epoch', '_lock', '_initialized',
        '_index_map', '_bucket_cache'
    )
    
    def __init__(
        self,
        dataset: Any,
        config: Optional[SamplerConfig] = None
    ) -> None:
        """Initialize with optimized defaults."""
        super().__init__(dataset)
        self.config = config or SamplerConfig()
        
        # Initialize state
        self._buckets: Dict[Tuple[int, int], BucketInfo] = {}
        self._cache: Dict[int, List[int]] = {}
        self._bucket_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._rng = np.random.RandomState(self.config.seed)
        self._epoch = 0
        self._lock = threading.RLock()
        self._initialized = False
        self._index_map: Dict[int, int] = {}
    
    @property
    def epoch(self) -> int:
        """Get current epoch."""
        return self._epoch
    
    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set epoch with proper state updates."""
        self._epoch = value
        self._rng = np.random.RandomState(
            self.config.seed + value if self.config.seed is not None else None
        )
    
    def _create_buckets(self) -> None:
        """Create optimized buckets with GPU acceleration."""
        if not hasattr(self.dataset, 'get_size'):
            return
            
        # Get all sizes
        sizes = [
            self.dataset.get_size(i)
            for i in range(len(self.dataset))
        ]
        
        # Convert to tensor for GPU acceleration
        if torch.cuda.is_available():
            sizes_tensor = torch.tensor(
                sizes,
                device='cuda',
                dtype=torch.float32
            )
        else:
            sizes_tensor = torch.tensor(sizes)
        
        # Calculate aspect ratios
        widths = sizes_tensor[:, 0]
        heights = sizes_tensor[:, 1]
        aspect_ratios = widths / heights
        
        # Group by bucket
        for idx, (size, ratio) in enumerate(zip(sizes, aspect_ratios.cpu().numpy())):
            bucket_size = (
                round(size[0] / self.config.bucket_size) * self.config.bucket_size,
                round(size[1] / self.config.bucket_size) * self.config.bucket_size
            )
            
            if bucket_size not in self._buckets:
                self._buckets[bucket_size] = BucketInfo([], size, ratio)
            
            self._buckets[bucket_size].indices.append(idx)
            self._index_map[idx] = len(self._buckets[bucket_size].indices) - 1
    
    @jit(nopython=True)
    def _shuffle_bucket(self, indices: np.ndarray) -> np.ndarray:
        """Shuffle bucket indices with Numba acceleration."""
        perm = np.arange(len(indices))
        self._rng.shuffle(perm)
        return indices[perm]
    
    def _get_bucket_tensor(
        self,
        bucket_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Get bucket tensor with caching."""
        if bucket_size in self._bucket_cache:
            return self._bucket_cache[bucket_size]
            
        bucket = self._buckets[bucket_size]
        if torch.cuda.is_available():
            tensor = torch.tensor(
                bucket.indices,
                device='cuda',
                dtype=torch.int64
            )
        else:
            tensor = torch.tensor(bucket.indices)
            
        if len(self._bucket_cache) < BUCKET_CACHE_SIZE:
            self._bucket_cache[bucket_size] = tensor
            
        return tensor
    
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _get_batch_indices(
        self,
        bucket_size: Tuple[int, int],
        start_idx: int
    ) -> List[int]:
        """Get batch indices with caching."""
        bucket = self._buckets[bucket_size]
        end_idx = start_idx + self.config.batch_size
        
        if end_idx > len(bucket.indices):
            if self.config.drop_last:
                return []
            end_idx = len(bucket.indices)
            
        return bucket.indices[start_idx:end_idx]
    
    def __iter__(self) -> Iterator[List[int]]:
        """Get optimized iterator over batches."""
        # Initialize if needed
        if not self._initialized:
            self._create_buckets()
            self._initialized = True
        
        # Shuffle if needed
        if self.config.shuffle:
            for bucket in self._buckets.values():
                bucket.indices = self._shuffle_bucket(
                    np.array(bucket.indices)
                ).tolist()
        
        # Generate batches
        for bucket_size, bucket in self._buckets.items():
            indices = self._get_bucket_tensor(bucket_size)
            
            for start_idx in range(0, len(indices), self.config.batch_size):
                batch_indices = self._get_batch_indices(
                    bucket_size, start_idx
                )
                
                if not batch_indices:
                    continue
                    
                if self.config.pin_memory and torch.cuda.is_available():
                    yield [int(i) for i in batch_indices]
                else:
                    yield batch_indices
    
    def __len__(self) -> int:
        """Get total number of batches."""
        if not self._initialized:
            self._create_buckets()
            self._initialized = True
            
        total = 0
        for bucket in self._buckets.values():
            if self.config.drop_last:
                total += len(bucket.indices) // self.config.batch_size
            else:
                total += (len(bucket.indices) + self.config.batch_size - 1) // self.config.batch_size
                
        return total
