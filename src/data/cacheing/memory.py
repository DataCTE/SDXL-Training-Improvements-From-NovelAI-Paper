"""
Memory management and caching utilities for SDXL training.

This module provides efficient memory management and caching mechanisms
for storing and retrieving data during training.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Set
from threading import Lock

logger = logging.getLogger(__name__)

class MemoryCache:
    """Manages memory caching with disk offloading capabilities.
    
    Provides thread-safe caching with memory management and disk persistence.
    
    Attributes:
        cache_dir: Directory for storing cached data
        in_memory_cache: Dictionary storing cached items in memory
        cached_files: Set of cached file hashes
        cache_lock: Thread lock for safe concurrent access
    """
    
    def __init__(self, cache_dir: str = "cache") -> None:
        """Initialize the memory cache.
        
        Args:
            cache_dir: Directory for storing cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.in_memory_cache: Dict[str, torch.Tensor] = {}
        self.cache_lock = Lock()
        
        # Initialize cached files tracking
        self.cached_files: Set[str] = set()
        self._load_cached_files()
        
    def _load_cached_files(self) -> None:
        """Load list of already cached files."""
        self.cached_files = {f.stem for f in self.cache_dir.glob("*.pt")}
        logger.info("Found %d existing cached items", len(self.cached_files))
        
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self.cache_dir / f"{hash(key)}.pt"
        
    def is_cached(self, key: str) -> bool:
        """Check if an item is already cached."""
        return str(hash(key)) in self.cached_files
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get item from cache."""
        # Check memory cache first
        if key in self.in_memory_cache:
            return self.in_memory_cache[key]
            
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as error:
                logger.error(
                    "Failed to load cached item for %s: %s",
                    key, str(error)
                )
        return None
        
    def put(self, key: str, value: torch.Tensor) -> None:
        """Store item in cache."""
        # Cache in memory
        with self.cache_lock:
            self.in_memory_cache[key] = value
            
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
        """Offload in-memory cache to disk."""
        try:
            with self.cache_lock:
                for key, value in self.in_memory_cache.items():
                    cache_path = self._get_cache_path(key)
                    torch.save(value, cache_path)
                    self.cached_files.add(str(hash(key)))
                
                # Clear memory cache
                self.in_memory_cache.clear()
                torch.cuda.empty_cache()
            
        except Exception as error:
            logger.error("Failed to offload cache to disk: %s", str(error))
            raise