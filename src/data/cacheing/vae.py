"""Ultra-optimized VAE encoding cache system."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import logging
from torch.cuda import amp
from pathlib import Path
import os
from .memory import MemoryCache, MemoryManager

logger = logging.getLogger(__name__)

class VAECache:
    """Ultra-optimized VAE encoding cache with parallel processing."""
    
    __slots__ = ('vae', '_memory_cache', '_lock', '_executor', '_batch_size',
                 '_stats', '_scaler', '_cache_dir', '_max_cache_size')
    
    def __init__(self, vae: nn.Module, cache_dir: Optional[str] = None,
                 max_cache_size: int = 10000, num_workers: int = 4,
                 batch_size: int = 8):
        """Initialize VAE cache with optimized defaults.
        
        Args:
            vae: VAE model to use for encoding
            cache_dir: Optional directory to save cached tensors
            max_cache_size: Maximum number of items to keep in cache
            num_workers: Number of worker threads for parallel processing
            batch_size: Batch size for parallel processing
        """
        # Initialize thread safety first
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Initialize model and parameters
        self.vae = vae.eval()  # Ensure eval mode
        self._batch_size = batch_size
        self._max_cache_size = max_cache_size
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._scaler = amp.GradScaler()  # Fixed: removed device_type parameter
        
        # Initialize cache last
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._memory_cache = MemoryCache(str(self._cache_dir))
        else:
            self._memory_cache = MemoryManager()
        
        # Pre-warm CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    def _get_cache_key(self, image: torch.Tensor) -> str:
        """Ultra-fast cache key generation."""
        return f"{image.shape}_{hash(image.data.cpu().numpy().tobytes())}"
    
    def _should_evict(self) -> bool:
        """Check if cache needs eviction."""
        return len(self._memory_cache) >= self._max_cache_size
    
    def _evict_items(self) -> None:
        """Efficient batch eviction."""
        with self._lock:
            while self._should_evict():
                _, tensor = self._memory_cache.popitem(last=False)
                if isinstance(tensor, torch.Tensor):
                    tensor.detach_()
                    del tensor
                self._stats['evictions'] += 1
    
    @torch.no_grad()
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized batch encoding with mixed precision."""
        if not torch.cuda.is_available():
            return self.vae.encode(images)[0].sample()
            
        # Use mixed precision for faster encoding
        with amp.autocast():
            encoded = self.vae.encode(images)[0]
            return self._scaler.scale(encoded).sample()
    
    def _parallel_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Parallel batch processing with optimal batch size."""
        batch_size = self._batch_size
        num_images = images.shape[0]
        
        if num_images <= batch_size:
            return self._encode_batch(images)
            
        # Split into optimal batches
        batches = torch.split(images, batch_size)
        futures = [
            self._executor.submit(self._encode_batch, batch)
            for batch in batches
        ]
        
        # Gather results efficiently
        encoded = [future.result() for future in futures]
        return torch.cat(encoded, dim=0)
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Ultra-fast VAE encoding with caching."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Process single vs batch
        if images.ndim == 3:
            images = images.unsqueeze(0)
            
        # Check cache for each image
        encoded_list = []
        uncached_indices = []
        uncached_images = []
        
        for i, image in enumerate(images):
            key = self._get_cache_key(image)
            cached = self._memory_cache.get(key)
            
            if cached is not None:
                encoded_list.append(cached)
                self._stats['hits'] += 1
            else:
                uncached_indices.append(i)
                uncached_images.append(image)
                self._stats['misses'] += 1
        
        # Process uncached images in parallel
        if uncached_images:
            uncached_batch = torch.stack(uncached_images)
            encoded_batch = self._parallel_encode(uncached_batch)
            
            # Cache new encodings
            for i, encoded in zip(uncached_indices, encoded_batch):
                key = self._get_cache_key(images[i])
                if self._should_evict():
                    self._evict_items()
                self._memory_cache.put(key, encoded.detach())
                encoded_list.append(encoded)
        
        # Combine results maintaining order
        return torch.stack(encoded_list)
    
    def clear(self) -> None:
        """Efficient cache clearing."""
        with self._lock:
            self._memory_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {**self._stats, **self._memory_cache.get_stats()}
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._memory_cache)
    
    def __del__(self) -> None:
        """Clean shutdown."""
        self.clear()
        self._executor.shutdown(wait=False)