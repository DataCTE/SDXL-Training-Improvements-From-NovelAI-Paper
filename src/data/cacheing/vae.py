"""Ultra-optimized VAE encoding cache system."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import logging
from torch.cuda import amp
from pathlib import Path
import os
from .memory import MemoryCache, MemoryManager
from multiprocessing import Manager
from multiprocessing import Pool

logger = logging.getLogger(__name__)

class VAECache:
    """Ultra-optimized VAE encoding cache with parallel processing."""
    
    __slots__ = ('vae', '_memory_cache', '_batch_size', '_stats',
                 '_scaler', '_cache_dir', '_max_cache_size', '_manager',
                 '_num_workers', '_pool')
    
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
        # Initialize multiprocessing components
        self._manager = Manager()
        self._stats = self._manager.dict({'hits': 0, 'misses': 0, 'evictions': 0})
        self._num_workers = num_workers
        
        # Initialize model and parameters
        self.vae = vae.eval()  # Ensure eval mode
        if torch.cuda.is_available():
            self.vae = self.vae.cuda()
        self._batch_size = batch_size
        self._max_cache_size = max_cache_size
        self._scaler = amp.GradScaler()
        
        # Initialize cache
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._memory_cache = MemoryCache(str(self._cache_dir))
        else:
            self._memory_cache = MemoryManager()
        
        # Initialize process pool
        self._pool = Pool(processes=num_workers) if num_workers > 0 else None
        
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
        while self._should_evict() and self._memory_cache:
            try:
                key = next(iter(self._memory_cache))
                del self._memory_cache[key]
                self._stats['evictions'] = self._stats.get('evictions', 0) + 1
            except (StopIteration, RuntimeError):
                break
            
        if self._should_evict() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized batch encoding with mixed precision."""
        # Ensure images start on CPU
        if images.device.type == 'cuda':
            images = images.cpu()
        
        # Pin memory if using CUDA
        if torch.cuda.is_available():
            images = images.pin_memory()
            images = images.cuda()
        
        # Use mixed precision for faster encoding
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                encoded = self.vae.encode(images)[0]
                encoded = self._scaler.scale(encoded)
                encoded = encoded.sample()
                
                # Expand from 3 to 4 channels if needed
                if encoded.shape[1] == 3:
                    # Add zero-filled 4th channel
                    zero_channel = torch.zeros_like(encoded[:, :1, :, :])
                    encoded = torch.cat([encoded, zero_channel], dim=1)
                    
                    # Log channel expansion
                    logger.debug("Expanded VAE output from 3 to 4 channels")
        else:
            # CPU fallback without mixed precision
            encoded = self.vae.encode(images)[0]
            encoded = encoded.sample()
            
            # Expand channels if needed
            if encoded.shape[1] == 3:
                zero_channel = torch.zeros_like(encoded[:, :1, :, :])
                encoded = torch.cat([encoded, zero_channel], dim=1)
        
        # Return tensor on CPU for caching
        return encoded.cpu()
    
    def _parallel_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Parallel batch processing with optimal batch size."""
        batch_size = self._batch_size
        num_images = images.shape[0]
        
        if num_images <= batch_size or not self._pool:
            return self._encode_batch(images)
            
        # Split into optimal batches
        batches = torch.split(images, batch_size)
        
        # Process batches in parallel
        encoded = self._pool.map(self._encode_batch, batches)
        return torch.cat(encoded, dim=0)
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Ultra-fast VAE encoding with caching."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Process single vs batch
        if images.ndim == 3:
            images = images.unsqueeze(0)
            
        # Ensure input is on CPU
        if images.device.type == 'cuda':
            images = images.cpu()
            
        # Check cache for each image
        encoded_list = []
        uncached_indices = []
        uncached_images = []
        
        for i, image in enumerate(images):
            key = self._get_cache_key(image)
            cached = self._memory_cache.get(key)
            
            if cached is not None:
                # Get cached tensor and ensure it's on CPU
                encoded = cached.cpu() if cached.device.type == 'cuda' else cached
                encoded_list.append(encoded)
                self._stats['hits'] += 1
            else:
                uncached_indices.append(i)
                uncached_images.append(image)
                self._stats['misses'] += 1
        
        # Process uncached images in parallel
        if uncached_images:
            uncached_batch = torch.stack(uncached_images)
            encoded_batch = self._parallel_encode(uncached_batch)
            
            # Cache new encodings (ensure on CPU)
            for i, encoded in zip(uncached_indices, encoded_batch):
                key = self._get_cache_key(images[i])
                if self._should_evict():
                    self._evict_items()
                # Store in cache on CPU
                encoded_cpu = encoded.cpu() if encoded.device.type == 'cuda' else encoded
                self._memory_cache.put(key, encoded_cpu)
                encoded_list.append(encoded_cpu)
        
        # Stack results and pin memory if using CUDA
        result = torch.stack(encoded_list)
        if torch.cuda.is_available():
            result = result.pin_memory()
        
        return result
    
    def clear(self) -> None:
        """Efficient cache clearing."""
        self._memory_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._memory_cache)
    
    def __del__(self) -> None:
        """Clean shutdown."""
        self.clear()
        if self._pool:
            self._pool.close()
            self._pool.join()