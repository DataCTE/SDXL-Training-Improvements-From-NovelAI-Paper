"""Ultra-optimized VAE encoding cache system."""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging
from torch.cuda import amp
from pathlib import Path
import os
from .memory import MemoryCache
from multiprocessing import Manager
from multiprocessing import Pool
import multiprocessing
from src.utils.vae_utils import normalize_vae_latents

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Set start method to 'spawn' for CUDA multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

class VAECache:
    """Ultra-optimized VAE encoding cache with parallel processing."""
    
    __slots__ = ('vae', '_memory_cache', '_batch_size', '_stats',
                 '_scaler', '_cache_dir', '_max_cache_size', '_manager',
                 '_num_workers', '_pool')
    
    def __init__(self, vae: nn.Module, cache_dir: Optional[str] = None,
                 max_cache_size: int = 10000, num_workers: int = 4,
                 batch_size: int = 8, max_memory_gb: float = 32.0):
        """Initialize VAE cache with optimized defaults.
        
        Args:
            vae: VAE model to use for encoding
            cache_dir: Optional directory to save cached tensors
            max_cache_size: Maximum number of items to keep in cache
            num_workers: Number of worker threads for parallel processing
            batch_size: Batch size for parallel processing
            max_memory_gb: Maximum memory in GB for cache
        """
        # Initialize multiprocessing components
        self._manager = Manager()
        self._stats = self._manager.dict({'hits': 0, 'misses': 0, 'evictions': 0})
        self._num_workers = num_workers
        
        # Initialize process pool
        self._pool = Pool(processes=num_workers) if num_workers > 0 else None
        
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
            self._memory_cache = MemoryCache(
                max_memory_gb=max_memory_gb,
                max_cache_size=max_cache_size,
                cache_dir=str(self._cache_dir)
            )
        else:
            self._memory_cache = MemoryCache(
                max_memory_gb=max_memory_gb,
                max_cache_size=max_cache_size
            )
        
        # Pre-warm CUDA if available
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
        """Evict items when cache is full."""
        if len(self._memory_cache) > self._max_cache_size:
            # Calculate number of items to evict (20% of cache)
            evict_count = max(1, int(self._max_cache_size * 0.2))
            self._memory_cache.evict(evict_count)
    
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
        with torch.cuda.amp.autocast():
            encoded = self.vae.encode(images)[0]
            encoded = self._scaler.scale(encoded)
            encoded = encoded.sample()
            
            # Normalize using NAI statistics
            encoded = normalize_vae_latents(encoded)
            
            return encoded
    
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
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Main interface for encoding images."""
        batch_size = images.shape[0]
        
        # Process in batches
        results = []
        for idx in range(0, batch_size, self._batch_size):
            batch = images[idx:idx + self._batch_size]
            
            # Check cache first
            cache_keys = [self._get_cache_key(img) for img in batch]
            cached = [self._memory_cache.get(k) for k in cache_keys]
            
            # Find which need encoding
            to_encode_idx = [i for i, c in enumerate(cached) if c is None]
            if to_encode_idx:
                batch_to_encode = batch[to_encode_idx]
                encoded = self._encode_batch(batch_to_encode)
                
                # Cache results
                for i, idx in enumerate(to_encode_idx):
                    self._memory_cache.put(cache_keys[idx], encoded[i])
                    cached[idx] = encoded[i]
            
            results.extend(cached)
        
        return torch.stack(results)
    
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
    
    def __delitem__(self, key: str) -> None:
        """Support item deletion."""
        if key in self._memory_cache:
            del self._memory_cache[key]

    