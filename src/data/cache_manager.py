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
        
        logger.info(f"Initialized cache manager with {self.executor._max_workers} workers")

    def get_cache_paths(self, image_path: str) -> Dict[str, Path]:
        stem = Path(image_path).stem
        return {
            'latent': self.latent_cache / f"{stem}.pt",
            'text': self.text_cache / f"{stem}.pt"
        }

    async def save_latent_async(self, path: Path, tensor: torch.Tensor):
        """Asynchronously save latent tensor."""
        try:
            buffer = io.BytesIO()
            await asyncio.get_event_loop().run_in_executor(
                self.executor, torch.save, tensor, buffer
            )
            
            async with aiofiles.open(path, 'wb') as f:
                await f.write(buffer.getvalue())
                
            # Update memory cache
            self.memory_cache.set(str(path), tensor)
            self.total_saved += 1
            
        except Exception as e:
            logger.error(f"Error saving latent to {path}: {e}")
            raise

    async def save_text_data_async(self, path: Path, data: Dict[str, Any]):
        """Asynchronously save text embedding data."""
        try:
            buffer = io.BytesIO()
            await asyncio.get_event_loop().run_in_executor(
                self.executor, torch.save, data, buffer
            )
            
            async with aiofiles.open(path, 'wb') as f:
                await f.write(buffer.getvalue())
                
            # Update memory cache
            self.memory_cache.set(str(path), data)
            self.total_saved += 1
            
        except Exception as e:
            logger.error(f"Error saving text data to {path}: {e}")
            raise

    def load_latent(self, path: Path) -> torch.Tensor:
        """Load latent tensor with memory caching."""
        data = self.memory_cache.get(str(path))
        if data is not None:
            return data
            
        try:
            data = torch.load(path)
            self.memory_cache.set(str(path), data)
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
        """Get cache statistics."""
        cache_stats = self.memory_cache.get_stats()
        return {
            **cache_stats,
            "total_saved": self.total_saved,
            "memory_usage_gb": get_memory_usage_gb()
        }