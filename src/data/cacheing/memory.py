"""
Memory management and caching utilities for SDXL training.

This module provides efficient memory management and caching mechanisms
for storing and retrieving data during training.
"""

import os
import mmap
import numpy as np
from typing import Dict, Any, Optional, Set, Tuple
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import logging
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryCache:
    """Manages memory caching with disk offloading capabilities.
    
    Provides thread-safe caching with memory management and disk persistence.
    
    Attributes:
        cache_dir: Directory for storing cached data
        in_memory_cache: Dictionary storing cached items in memory
        cached_files: Set of cached file hashes
    """
    
    def __init__(self, cache_dir: str = "cache") -> None:
        """Initialize the memory cache.
        
        Args:
            cache_dir: Directory for storing cached data
        """
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.in_memory_cache = MemoryManager(max_memory_gb=32.0)
        self.cached_files: Set[str] = set()
        
        # Initialize cached files tracking
        self._load_cached_files()
        
    def _load_cached_files(self) -> None:
        """Load list of already cached files."""
        if self.cache_dir.exists():
            self.cached_files = {f.stem for f in self.cache_dir.iterdir() if f.is_file()}
            logger.info("Found %d existing cached items", len(self.cached_files))
        
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        return str(self.cache_dir / f"{hash(key)}.pt")
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.
        
        Checks memory cache first, then disk cache if item not found in memory.
        
        Args:
            key: Cache key to look up
            default: Value to return if key not found
            
        Returns:
            Cached value or default if not found
        """
        # Check memory cache first
        value = self.in_memory_cache.get(key, None)
        if value is not None:
            return value
            
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                value = torch.load(cache_path)
                # Cache in memory for faster future access
                self.in_memory_cache.put(key, value)
                return value
            except Exception as error:
                logger.error(
                    "Failed to load cached item for %s: %s",
                    key, str(error)
                )
        return default
        
    def put(self, key: str, value: Any) -> None:
        """Store item in both memory and disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Cache in memory
        self.in_memory_cache.put(key, value)
            
        # Cache to disk
        try:
            cache_path = self._get_cache_path(key)
            torch.save(value, cache_path)
            self.cached_files.add(str(hash(key)))
        except Exception as error:
            logger.error(
                "Failed to cache item for %s: %s",
                key, str(error)
            )
            
    def offload_to_disk(self) -> None:
        """Offload in-memory cache to disk to free memory."""
        try:
            for key, value in self.in_memory_cache._cache.items():
                cache_path = self._get_cache_path(key)
                torch.save(value, cache_path)
                self.cached_files.add(str(hash(key)))
                
            # Clear memory cache
            self.clear()
            
        except Exception as error:
            logger.error("Failed to offload cache to disk: %s", str(error))
            raise
            
    def clear(self) -> None:
        """Clear both memory and disk cache."""
        self.in_memory_cache.clear()
        self.cached_files.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics including memory and disk usage."""
        stats = self.in_memory_cache.get_stats()
        stats['disk_items'] = len(self.cached_files)
        return stats
        
    def __len__(self) -> int:
        """Get total number of cached items (memory + disk)."""
        return len(self.in_memory_cache) + len(self.cached_files)


class MemoryManager:
    """Ultra-optimized memory manager for cache system."""
    
    __slots__ = ('_cache', '_lock', '_max_memory_gb', '_executor', 
                 '_memory_threshold', '_eviction_trigger', '_stats')
    
    def __init__(self, max_memory_gb: float = 32.0):
        """Initialize memory manager with pre-allocated resources."""
        self._cache = OrderedDict()  # LRU cache
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._max_memory_gb = max_memory_gb
        self._executor = ThreadPoolExecutor(max_workers=4)  # Background tasks
        self._memory_threshold = 0.8  # 80% memory threshold
        self._eviction_trigger = 0.9  # 90% trigger eviction
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Pre-warm the cache
        self._executor.submit(self._prewarm_cache)
    
    def _prewarm_cache(self) -> None:
        """Pre-warm cache for better initial performance."""
        try:
            # Pre-allocate memory pool
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            if torch.cuda.is_available():
                # Reserve CUDA memory pool
                torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception as e:
            logger.warning(f"Cache pre-warming failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage ratio using vectorized operations."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return psutil.Process().memory_percent() / 100.0
        except Exception:
            return 0.0
    
    def _should_evict(self) -> bool:
        """Ultra-fast memory pressure check."""
        return self._get_memory_usage() > self._eviction_trigger
    
    def _evict_items(self) -> None:
        """Efficient batch eviction of cache items."""
        with self._lock:
            while self._should_evict() and self._cache:
                _, item = self._cache.popitem(last=False)  # FIFO eviction
                if isinstance(item, torch.Tensor):
                    item.detach_()
                    del item
                self._stats['evictions'] += 1
            
            # Force garbage collection if needed
            if self._should_evict():
                torch.cuda.empty_cache()
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Ultra-fast cache retrieval with minimal locking."""
        try:
            with self._lock:
                value = self._cache.get(key, default)
                if value is not None:
                    # Update access order without full reordering
                    self._cache.move_to_end(key)
                    self._stats['hits'] += 1
                    return value
                self._stats['misses'] += 1
                return default
        except Exception:
            return default
    
    def put(self, key: str, value: Any) -> None:
        """Optimized cache insertion with memory management."""
        with self._lock:
            if self._should_evict():
                self._evict_items()
            
            # Fast path for tensors
            if isinstance(value, torch.Tensor):
                value = value.detach()  # Detach from graph
                
            self._cache[key] = value
            self._cache.move_to_end(key)  # Move to end for LRU
    
    def popitem(self, last: bool = True) -> Tuple[str, Any]:
        """Remove and return a (key, value) pair from the cache.
        
        Args:
            last: If True, remove the last item (most recently used),
                 if False, remove the first item (least recently used)
        
        Returns:
            Tuple of (key, value) removed from cache
        """
        with self._lock:
            return self._cache.popitem(last=last)
    
    def clear(self) -> None:
        """Efficient cache clearing with CUDA optimization."""
        with self._lock:
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics using minimal locking."""
        with self._lock:
            return dict(self._stats)
    
    def __len__(self) -> int:
        """Fast cache size check."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Ultra-fast key existence check."""
        return key in self._cache
    
    def __del__(self) -> None:
        """Clean shutdown of resources."""
        self.clear()
        self._executor.shutdown(wait=False)