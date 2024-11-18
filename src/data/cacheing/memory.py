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

logger = logging.getLogger(__name__)

class MemoryManager:
    """Ultra-optimized memory manager for cache system."""
    
    __slots__ = ('_cache', '_max_memory_gb', '_stats', '_manager')
    
    def __init__(self, max_memory_gb: float = 32.0):
        """Initialize memory manager with pre-allocated resources."""
        manager = Manager()
        self._manager = manager
        self._cache = manager.dict()  # Use Manager dict instead of OrderedDict
        self._max_memory_gb = max_memory_gb
        self._stats = manager.dict({'hits': 0, 'misses': 0, 'evictions': 0})
        
        # Pre-warm the cache
        self._prewarm_cache()
    
    def _prewarm_cache(self) -> None:
        """Pre-warm cache for better initial performance."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception as e:
            logger.warning(f"Cache pre-warming failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage ratio."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return psutil.Process().memory_percent() / 100.0
        except Exception:
            return 0.0
    
    def _should_evict(self) -> bool:
        """Check if memory pressure requires eviction."""
        return self._get_memory_usage() > 0.9
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get item from cache."""
        try:
            value = self._cache.get(key, default)
            if value is not None:
                self._stats['hits'] = self._stats.get('hits', 0) + 1
                return value
            self._stats['misses'] = self._stats.get('misses', 0) + 1
            return default
        except Exception:
            return default
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache."""
        if self._should_evict():
            self._evict_items()
            
        if isinstance(value, torch.Tensor):
            value = value.detach()
                
        self._cache[key] = value
    
    def _evict_items(self) -> None:
        """Evict items when memory pressure is high."""
        while self._should_evict() and self._cache:
            try:
                key = next(iter(self._cache))
                del self._cache[key]
                self._stats['evictions'] = self._stats.get('evictions', 0) + 1
            except (StopIteration, RuntimeError):
                break
            
        if self._should_evict() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._cache)

class MemoryCache(MemoryManager):
    """Memory cache with disk persistence."""
    
    def __init__(self, cache_dir: str = "cache", max_memory_gb: float = 32.0):
        """Initialize the memory cache."""
        super().__init__(max_memory_gb)
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cached_files = self._manager.list()
        self._load_cached_files()
    
    def _load_cached_files(self) -> None:
        """Load list of cached files."""
        if self.cache_dir.exists():
            existing_files = set(f.stem for f in self.cache_dir.iterdir() if f.is_file())
            self.cached_files.extend(list(existing_files))
    
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        return str(self.cache_dir / f"{hash(key)}.pt")
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache with disk persistence."""
        super().put(key, value)
        if key not in self.cached_files:
            self.cached_files.append(key)
            
        if self.cache_dir:
            try:
                cache_path = self._get_cache_path(key)
                torch.save(value, cache_path)
            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get item from cache with disk fallback."""
        value = super().get(key, None)
        if value is None and key in self.cached_files:
            try:
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    value = torch.load(cache_path)
                    super().put(key, value)  # Load into memory cache
                    return value
            except Exception as e:
                logger.error(f"Failed to load from disk cache: {e}")
        return value if value is not None else default
    
    def clear(self) -> None:
        """Clear both memory and disk cache."""
        super().clear()
        self.cached_files.clear()
        if self.cache_dir and self.cache_dir.exists():
            try:
                for f in self.cache_dir.iterdir():
                    if f.is_file():
                        f.unlink()
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")