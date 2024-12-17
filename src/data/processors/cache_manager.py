# src/data/processors/cache_manager.py
from pathlib import Path
import torch
from typing import Dict, Any, Optional, List
import logging
import asyncio
import aiofiles
import io
import gc

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
        self.memory_cache = MemoryCache(max_items=100)  # Reduced from 1000 to prevent memory buildup
        
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

    async def save_latent_async(self, path: Path, tensor: torch.Tensor) -> None:
        """Save latent tensor to disk immediately."""
        try:
            # Ensure tensor is on CPU and detached
            if tensor.device.type != 'cpu':
                tensor = tensor.detach().cpu()
            
            # Save tensor directly without buffering
            torch.save(tensor, str(path), pickle_protocol=4)
            
            # Ensure file is written to disk
            with open(path, 'rb') as f:
                f.read(1)  # Force disk write
            
            # Clear reference and force garbage collection
            del tensor
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error saving latent to {path}: {e}")

    async def save_text_data_async(self, path: Path, data: Dict) -> None:
        """Save text data to disk immediately."""
        try:
            # Convert any tensors to CPU and detach
            cpu_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    cpu_data[key] = value.detach().cpu()
                else:
                    cpu_data[key] = value
            
            # Save data directly without buffering
            torch.save(cpu_data, str(path), pickle_protocol=4)
            
            # Ensure file is written to disk
            with open(path, 'rb') as f:
                f.read(1)  # Force disk write
            
            # Clear references
            del cpu_data
            del data  # Also clear original data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error saving text data to {path}: {e}")

    def load_latent(self, path: Path) -> torch.Tensor:
        """Load latent tensor with memory caching and validation."""
        # Try memory cache first
        data = self.memory_cache.get(str(path))
        if data is not None:
            self.cache_hits += 1
            return data.cpu()  # Always return CPU tensor
            
        try:
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {path}")
                
            data = torch.load(path)
            if not isinstance(data, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(data)}")
                
            # Store CPU tensor in cache
            data = data.cpu()
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
            return self._ensure_cpu_tensors(data)
            
        try:
            data = torch.load(path)
            # Ensure all tensors are on CPU
            data = self._ensure_cpu_tensors(data)
            self.memory_cache.set(str(path), data)
            return data
        except Exception as e:
            logger.error(f"Error loading text data from {path}: {e}")
            raise

    def _ensure_cpu_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all tensors in dictionary are on CPU."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu()
            elif isinstance(value, dict):
                result[key] = self._ensure_cpu_tensors(value)
            else:
                result[key] = value
        return result

    async def cleanup(self):
        """Clean up resources and clear caches."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Successfully cleaned up cache manager resources")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def clear_memory_cache(self):
        """Clear the in-memory cache and force garbage collection."""
        self.memory_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

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
        """Cache both latent and text data for an item asynchronously."""
        if not self.use_caching:
            return
            
        try:
            # Get cache paths
            cache_paths = self.get_cache_paths(image_path)
            
            # Save latent immediately if present
            if 'processed_image' in item:
                await self.save_latent_async(
                    cache_paths['latent'],
                    item['processed_image']
                )
                # Clear the tensor from the item dict after saving
                del item['processed_image']
            
            # Save text data immediately if present
            if 'text_data' in item:
                text_data = item['text_data'].copy()  # Make a copy to avoid modifying original
                if 'tag_weights' in item:
                    text_data['tag_weights'] = item['tag_weights']
                await self.save_text_data_async(
                    cache_paths['text'],
                    text_data
                )
                # Clear the data from the item dict after saving
                del item['text_data']
                if 'tag_weights' in item:
                    del item['tag_weights']
            
            # Update cache stats
            self.total_saved += 1
            
            # Clear memory cache periodically
            if self.total_saved % 50 == 0:  # More frequent cleanup
                self.memory_cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error caching item {image_path}: {e}")

    async def cache_batch_items(self, items: List[Dict[str, Any]]) -> None:
        """Cache a batch of items immediately after processing."""
        if not self.use_caching:
            return
            
        try:
            # Process each item sequentially to avoid memory buildup
            for item in items:
                if 'image_path' not in item:
                    continue
                    
                # Get cache paths
                cache_paths = self.get_cache_paths(item['image_path'])
                
                # Save latent immediately if present
                if 'processed_image' in item:
                    await self.save_latent_async(
                        cache_paths['latent'],
                        item['processed_image']
                    )
                    del item['processed_image']  # Clear after saving
                
                # Save text data immediately if present
                if 'text_data' in item:
                    text_data = item['text_data'].copy()
                    if 'tag_weights' in item:
                        text_data['tag_weights'] = item['tag_weights']
                    await self.save_text_data_async(
                        cache_paths['text'],
                        text_data
                    )
                    del item['text_data']  # Clear after saving
                    if 'tag_weights' in item:
                        del item['tag_weights']
                
                # Clear memory after each item
                gc.collect()
                torch.cuda.empty_cache()
            
            # Final cleanup
            self.memory_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error caching batch items: {e}")