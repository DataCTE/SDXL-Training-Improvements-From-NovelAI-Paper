"""Memory management and caching utilities."""

import os
import mmap
import numpy as np
from typing import Dict, Any, Optional, Set, Tuple
import torch
from multiprocessing import Pool, Manager, RLock
import logging
import psutil
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class MemoryCache:
    """Ultra-optimized memory manager for dual encoder cache system."""
    
    __slots__ = ('_cache', '_max_memory_gb', '_max_cache_size', '_stats', '_manager', '_lock', 'cache_dir', 'cached_files')
    
    def __init__(self, max_memory_gb: float = 32.0, max_cache_size: int = 100000, cache_dir: Optional[Path] = None):
        """Initialize memory manager with pre-allocated resources."""
        manager = Manager()
        self._manager = manager
        self._cache = manager.dict()
        self._max_memory_gb = max_memory_gb
        self._max_cache_size = max_cache_size
        self._stats = manager.dict({'hits': 0, 'misses': 0, 'evictions': 0})
        self._lock = RLock()
        
        # Disk cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cached_files = manager.set()
        
        # Pre-warm the cache
        self._prewarm_cache()
    
    def _prewarm_cache(self) -> None:
        """Pre-warm cache for better initial performance."""
        try:
            # Reserve memory for cache
            reserved_mem = int(self._max_memory_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
            torch.empty(reserved_mem // 4, dtype=torch.float32)  # Reserve 1/4 for cache
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Cache pre-warming failed: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text."""
        # Use SHA-256 for consistent hashing
        return hashlib.sha256(text.encode()).hexdigest()

    def _should_evict(self) -> bool:
        """Check if cache needs eviction based on memory usage."""
        if not torch.cuda.is_available():
            return len(self._cache) >= self._max_cache_size
            
        # Check GPU memory usage
        memory_used = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        return memory_used >= self._max_memory_gb * 0.9  # 90% threshold

    def _evict_items(self) -> None:
        """Efficient batch eviction with memory management."""
        with self._lock:
            while self._should_evict() and self._cache:
                try:
                    # Get oldest item
                    key = next(iter(self._cache))
                    
                    # Try to save to disk before evicting
                    if self.cache_dir:
                        self._save_to_disk(key, self._cache[key])
                        self.cached_files.add(key)
                    
                    del self._cache[key]
                    self._stats['evictions'] = self._stats.get('evictions', 0) + 1
                except (StopIteration, RuntimeError) as e:
                    logger.error(f"Eviction error: {e}")
                    break
                    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_cache_path(self, key: str) -> Path:
        """Get path for disk cache file."""
        return self.cache_dir / f"{key}.pt"

    def put(self, key: str, value: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store embeddings tuple in cache with disk fallback."""
        with self._lock:
            if self._should_evict():
                self._evict_items()
            
            # Store on CPU for memory efficiency
            if isinstance(value, tuple):
                value = tuple(t.cpu() for t in value)
            else:
                value = value.cpu()
                
            self._cache[key] = value
            
            # Backup to disk if enabled
            if self.cache_dir:
                self._save_to_disk(key, value)
                self.cached_files.add(key)

    def get(self, key: str, default: Any = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get embeddings from cache with disk fallback."""
        with self._lock:
            value = self._cache.get(key)
            
            if value is None and self.cache_dir and key in self.cached_files:
                try:
                    cache_path = self._get_cache_path(key)
                    if cache_path.exists():
                        value = torch.load(cache_path, map_location='cpu')
                        self._cache[key] = value
                        self._stats['hits'] += 1
                        return value
                except Exception as e:
                    logger.error(f"Failed to load from disk cache: {e}")
                    self._stats['misses'] += 1
                    return default
            
            if value is not None:
                self._stats['hits'] += 1
            else:
                self._stats['misses'] += 1
                
            return value if value is not None else default

    def clear(self) -> None:
        """Clear both memory and disk cache."""
        with self._lock:
            self._cache.clear()
            if self.cache_dir and self.cache_dir.exists():
                try:
                    for f in self.cache_dir.iterdir():
                        if f.is_file():
                            f.unlink()
                    self.cached_files.clear()
                except Exception as e:
                    logger.error(f"Failed to clear disk cache: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        with self._lock:
            stats = dict(self._stats)  # Create a copy of stats
            stats.update({
                'size': len(self._cache),
                'disk_cached': len(self.cached_files) if hasattr(self, 'cached_files') else 0,
            })
            
            if torch.cuda.is_available():
                stats['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            
            return stats

    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)