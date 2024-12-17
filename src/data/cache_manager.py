# src/data/cache_manager.py
from pathlib import Path
import torch
from typing import Dict, Any, Optional
import logging
import asyncio
import aiofiles
import io
from src.data.utils import create_thread_pool, MemoryCache, get_memory_usage_gb

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str, max_workers: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.latent_cache = self.cache_dir / "latents"
        self.text_cache = self.cache_dir / "text"
        
        # Create cache directories
        for cache_dir in [self.latent_cache, self.text_cache]:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize thread pool and memory cache
        self.executor = create_thread_pool(max_workers)
        self.memory_cache = MemoryCache(max_items=1000)
        
        # Stats
        self.total_saved = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"Initialized cache manager:\n"
            f"- Cache dir: {self.cache_dir}\n"
            f"- Workers: {self.executor._max_workers}\n"
            f"- Memory cache size: {self.memory_cache.max_items}"
        )

    def get_cache_paths(self, image_path: str) -> Dict[str, Path]:
        """Get cache paths and validate they exist."""
        stem = Path(image_path).stem
        paths = {
            'latent': self.latent_cache / f"{stem}.pt",
            'text': self.text_cache / f"{stem}.pt"
        }
        
        # Log cache status
        for name, path in paths.items():
            if path.exists():
                logger.debug(f"Cache hit for {name}: {path}")
                self.cache_hits += 1
            else:
                logger.debug(f"Cache miss for {name}: {path}")
                self.cache_misses += 1
                
        return paths

    async def save_latent_async(self, path: Path, tensor: torch.Tensor):
        """Asynchronously save latent tensor with validation."""
        try:
            # Validate tensor
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(tensor)}")
                
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temporary file first
            temp_path = path.with_suffix('.tmp')
            buffer = io.BytesIO()
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, torch.save, tensor, buffer
            )
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(buffer.getvalue())
                
            # Atomic rename
            temp_path.rename(path)
            
            # Update memory cache
            self.memory_cache.set(str(path), tensor)
            self.total_saved += 1
            
            logger.debug(f"Successfully saved latent to {path}")
            
        except Exception as e:
            logger.error(f"Error saving latent to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def save_text_data_async(self, path: Path, data: Dict[str, Any]):
        """Asynchronously save text embedding data with validation."""
        try:
            # Validate data
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")
                
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temporary file first
            temp_path = path.with_suffix('.tmp')
            buffer = io.BytesIO()
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, torch.save, data, buffer
            )
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(buffer.getvalue())
                
            # Atomic rename
            temp_path.rename(path)
            
            # Update memory cache
            self.memory_cache.set(str(path), data)
            self.total_saved += 1
            
            logger.debug(f"Successfully saved text data to {path}")
            
        except Exception as e:
            logger.error(f"Error saving text data to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_latent(self, path: Path) -> torch.Tensor:
        """Load latent tensor with memory caching and validation."""
        # Try memory cache first
        data = self.memory_cache.get(str(path))
        if data is not None:
            self.cache_hits += 1
            return data
            
        try:
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {path}")
                
            data = torch.load(path)
            if not isinstance(data, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(data)}")
                
            self.memory_cache.set(str(path), data)
            self.cache_misses += 1
            return data
            
        except Exception as e:
            logger.error(f"Error loading latent from {path}: {e}")
            raise

    def load_text_data(self, path: Path) -> Dict[str, Any]:
        """Load text embedding data with memory caching."""
        data = self.memory_cache.get(str(path))
        if data is not None:
            return data
            
        try:
            data = torch.load(path)
            self.memory_cache.set(str(path), data)
            return data
        except Exception as e:
            logger.error(f"Error loading text data from {path}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        cache_stats = self.memory_cache.get_stats()
        return {
            **cache_stats,
            "total_saved": self.total_saved,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "memory_usage_gb": get_memory_usage_gb()
        }