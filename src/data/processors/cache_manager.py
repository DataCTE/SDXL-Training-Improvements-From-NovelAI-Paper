# src/data/processors/cache_manager.py
from pathlib import Path
import torch
from typing import Dict, Any, Optional
import logging
import asyncio
import aiofiles
import io

# Internal imports from utils
from src.data.processors.utils.system_utils import create_thread_pool, get_memory_usage_gb, MemoryCache

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str, max_workers: Optional[int] = None, use_caching: bool = True):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            max_workers: Maximum number of worker threads
            use_caching: Whether to enable caching
        """
        self.cache_dir = Path(cache_dir)
        self.use_caching = use_caching
        self.latent_cache = self.cache_dir / "latents"
        self.text_cache = self.cache_dir / "text"
        
        # Create cache directories if caching is enabled
        if self.use_caching:
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
            f"- Memory cache size: {self.memory_cache.max_items}\n"
            f"- Caching enabled: {self.use_caching}"
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

    async def get_cached_item(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Get cached item data for both latent and text asynchronously.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing cached data if both latent and text exist, None otherwise
        """
        try:
            # Get cache paths
            cache_paths = self.get_cache_paths(image_path)
            
            # Check if both cache files exist
            if not (cache_paths['latent'].exists() and cache_paths['text'].exists()):
                return None
                
            # Load cached data asynchronously
            latent, text_data = await asyncio.gather(
                asyncio.to_thread(self.load_latent, cache_paths['latent']),
                asyncio.to_thread(self.load_text_data, cache_paths['text'])
            )
            
            # Return combined data
            cached_item = {
                'image_path': image_path,
                'latent': latent,
                'text_data': text_data,
                'latent_cache': cache_paths['latent'],
                'text_cache': cache_paths['text']
            }
            
            # Add tag weights if they exist in text data
            if 'tag_weights' in text_data:
                cached_item['tag_weights'] = text_data['tag_weights']
            
            return cached_item
            
        except Exception as e:
            logger.debug(f"Cache miss for {image_path}: {str(e)}")
            return None

    async def cache_item(self, image_path: str, item: Dict[str, Any]) -> None:
        """Cache both latent and text data for an item asynchronously.
        
        Args:
            image_path: Path to the image file
            item: Dictionary containing processed data
        """
        if not self.use_caching:
            return
            
        try:
            # Get cache paths
            cache_paths = self.get_cache_paths(image_path)
            
            # Prepare text data with tag weights if available
            text_data = item['text_data']
            if 'tag_weights' in item:
                text_data['tag_weights'] = item['tag_weights']
            
            # Save latent and text data concurrently
            await asyncio.gather(
                self.save_latent_async(cache_paths['latent'], item['processed_image']),
                self.save_text_data_async(cache_paths['text'], text_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching item {image_path}: {e}")
            # Continue processing even if caching fails