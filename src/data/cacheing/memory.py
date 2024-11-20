"""Memory management and caching utilities."""

import os
import mmap
import numpy as np
from typing import Dict, Any, Optional, Set, Tuple, Union
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
    
    def __init__(self, max_memory_gb: float = 32.0, max_cache_size: int = 100000, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize memory manager with pre-allocated resources."""
        manager = Manager()
        self._manager = manager
        self._cache = manager.dict()
        self._max_memory_gb = float(max_memory_gb)
        self._max_cache_size = int(max_cache_size)
        self._stats = manager.dict({'hits': 0, 'misses': 0, 'evictions': 0})
        self._lock = RLock()
        
        # Disk cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cached_files = manager.list()

    def __delitem__(self, key: str) -> None:
        """Delete item from cache with proper cleanup."""
        with self._lock:
            if key in self._cache:
                # Remove from memory cache
                del self._cache[key]
                
                # Clean up disk cache if enabled
                if self.cache_dir and key in self.cached_files:
                    try:
                        cache_path = self._get_cache_path(key)
                        if cache_path.exists():
                            cache_path.unlink()
                        self.cached_files.remove(key)
                    except Exception as e:
                        logger.error(f"Failed to delete cached file: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cached file."""
        return self.cache_dir / f"{key}.pt"

    def put(self, key: str, value: Any) -> None:
        """Store value in cache with disk fallback."""
        with self._lock:
            self._cache[key] = value
            if self.cache_dir:
                try:
                    cache_path = self._get_cache_path(key)
                    torch.save(value, cache_path)
                    if key not in self.cached_files:
                        self.cached_files.append(key)
                except Exception as e:
                    logger.error(f"Failed to save to disk cache: {e}")

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get value from cache with disk fallback."""
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
            return value

    def clear(self) -> None:
        """Clear cache and remove cached files."""
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

    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def __iter__(self):
        """Iterate over cache keys."""
        return iter(self._cache)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return dict(self._stats)  # Return a copy of stats dict

    def _evict_items(self, count: int = 1) -> None:
        """Evict oldest items from cache."""
        with self._lock:
            keys_to_evict = list(self._cache.keys())[:count]
            for key in keys_to_evict:
                self.__delitem__(key)
                self._stats['evictions'] += 1