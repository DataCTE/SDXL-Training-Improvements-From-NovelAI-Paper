# src/data/cache_manager.py
from pathlib import Path
import torch
from typing import Dict, Any, Optional
import logging
import asyncio
import aiofiles
import io
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from src.data.thread_config import get_optimal_cpu_threads

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str, max_workers: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.latent_cache = self.cache_dir / "latents"
        self.text_cache = self.cache_dir / "text"
        
        # Create cache directories
        for cache_dir in [self.latent_cache, self.text_cache]:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize thread pool with optimal number of workers
        if max_workers is None:
            max_workers = min(32, multiprocessing.cpu_count() * 4)
        self.executor = ThreadPoolExecutor(max_workers=get_optimal_cpu_threads())
        
        # Memory cache with size limit
        self.memory_cache = {}
        self.max_cache_items = 1000

    def get_cache_paths(self, image_path: str) -> Dict[str, Path]:
        stem = Path(image_path).stem
        return {
            'latent': self.latent_cache / f"{stem}.pt",
            'text': self.text_cache / f"{stem}.pt"
        }

    async def save_latent_async(self, path: Path, latent: torch.Tensor):
        """Asynchronously save latent tensor."""
        buffer = io.BytesIO()
        await asyncio.get_event_loop().run_in_executor(
            self.executor, torch.save, latent.cpu(), buffer
        )
        
        async with aiofiles.open(path, 'wb') as f:
            await f.write(buffer.getvalue())
            
        # Update memory cache
        cache_key = str(path)
        self.memory_cache[cache_key] = latent
        self._cleanup_cache()

    async def save_text_data_async(self, path: Path, data: Dict[str, Any]):
        """Asynchronously save text embedding data."""
        buffer = io.BytesIO()
        await asyncio.get_event_loop().run_in_executor(
            self.executor, torch.save, data, buffer
        )
        
        async with aiofiles.open(path, 'wb') as f:
            await f.write(buffer.getvalue())
            
        # Update memory cache
        cache_key = str(path)
        self.memory_cache[cache_key] = data
        self._cleanup_cache()

    def load_latent(self, path: Path) -> torch.Tensor:
        """Load latent tensor with memory caching."""
        cache_key = str(path)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        data = torch.load(path)
        self.memory_cache[cache_key] = data
        self._cleanup_cache()
        return data

    def load_text_data(self, path: Path) -> Dict[str, Any]:
        """Load text embedding data with memory caching."""
        cache_key = str(path)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        data = torch.load(path)
        self.memory_cache[cache_key] = data
        self._cleanup_cache()
        return data

    def _cleanup_cache(self):
        """Clean up memory cache if it gets too large."""
        if len(self.memory_cache) > self.max_cache_items:
            # Remove oldest items
            remove_keys = list(self.memory_cache.keys())[:-self.max_cache_items]
            for key in remove_keys:
                del self.memory_cache[key]