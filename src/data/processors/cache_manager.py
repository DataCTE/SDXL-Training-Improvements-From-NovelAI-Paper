# src/data/processors/cache_manager.py
import os
import logging
import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import gc
from collections import OrderedDict
import json
import functools
import concurrent.futures

from src.data.processors.utils.system_utils import get_gpu_memory_usage, cleanup_processor
from src.data.processors.utils.file_utils import ensure_dir
from src.config.config import CacheConfig

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of processed images and text embeddings with optimized performance."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache manager with configuration."""
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        
        # Create cache directories in parallel or in a single pass
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(ensure_dir, [self.cache_dir / d for d in ["latents", "text", "metadata"]])

        self.latents_dir = self.cache_dir / "latents"
        self.text_dir = self.cache_dir / "text"
        self.metadata_dir = self.cache_dir / "metadata"

        # Use OrderedDict for LRU behavior
        self._memory_cache = OrderedDict()
        self._metadata_memory_cache = OrderedDict()  # Additional small metadata cache
        self._max_memory_items = getattr(config, 'max_memory_items', 1000)
        
        # Create a more aggressive thread pool for parallel I/O
        worker_count = min(64, (os.cpu_count() or 8) * 2)
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cache_io_"
        )
        
        # Prepare torch.save options
        self._save_options = {
            '_use_new_zipfile_serialization': True,
            'pickle_protocol': 4
        }
        
        logger.info(
            f"Initialized CacheManager:\n"
            f"- Cache directory: {self.cache_dir}\n"
            f"- Memory cache enabled: {config.use_memory_cache}\n"
            f"- Max memory items: {self._max_memory_items}\n"
            f"- Cache format: {config.cache_format}\n"
            f"- I/O pool workers: {worker_count}"
        )

    @functools.lru_cache(maxsize=1024)
    def get_cache_paths(self, image_path: str) -> Dict[str, Path]:
        """Get cache paths for an image file with caching."""
        image_path = Path(image_path)
        relative_path = image_path.stem
        
        return {
            'latent': self.latents_dir / f"{relative_path}.pt",
            'text': self.text_dir / f"{relative_path}.pt",
            'metadata': self.metadata_dir / f"{relative_path}.json"
        }

    async def cache_item(self, image_path: str, processed_item: Dict[str, Any]) -> None:
        """Cache processed item data with parallel I/O operations."""
        try:
            cache_paths = self.get_cache_paths(image_path)
            cache_tasks = []
            
            # Extract data for caching
            latents = processed_item.get('latents')
            text_data = self._prepare_text_data(processed_item)
            metadata = self._prepare_metadata(processed_item)
            
            # Schedule parallel save operations
            if latents is not None and torch.is_tensor(latents):
                cache_tasks.append(self._io_pool.submit(
                    torch.save,
                    latents.cpu(),
                    cache_paths['latent'],
                    **self._save_options
                ))
            
            if text_data:
                cache_tasks.append(self._io_pool.submit(
                    torch.save,
                    text_data,
                    cache_paths['text'],
                    **self._save_options
                ))
            
            if any(metadata.values()):
                cache_tasks.append(self._io_pool.submit(
                    self._save_json,
                    cache_paths['metadata'],
                    metadata
                ))

            # Wait for all I/O operations to complete
            if cache_tasks:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    concurrent.futures.wait,
                    cache_tasks
                )
            
            # Update memory cache
            if self.config.use_memory_cache:
                self._update_memory_cache(image_path, latents, text_data, metadata)
                # Also store metadata in a small memory cache if you access it often
                if any(metadata.values()):
                    self._metadata_memory_cache[str(image_path)] = metadata

        except Exception as e:
            logger.error(f"Error caching item {image_path}: {e}")
            self._log_debug_info(processed_item)

    def _prepare_text_data(self, processed_item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare text data for caching."""
        text_data = {}
        for key in ['prompt_embeds', 'pooled_prompt_embeds', 'tag_weights']:
            if key in processed_item and torch.is_tensor(processed_item[key]):
                text_data[key] = processed_item[key].cpu()
        return text_data

    def _prepare_metadata(self, processed_item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for caching."""
        return {
            'original_size': processed_item.get('original_size'),
            'crop_top_left': processed_item.get('crop_top_left'),
            'target_size': (processed_item.get('width'), processed_item.get('height'))
        }

    def _update_memory_cache(self, image_path: str, latents, text_data, metadata):
        """Update memory cache with LRU behavior."""
        cache_key = str(image_path)
        
        # Implement efficient LRU using OrderedDict
        if len(self._memory_cache) >= self._max_memory_items:
            self._memory_cache.popitem(last=False)
        
        self._memory_cache[cache_key] = {
            'latents': latents.cpu() if (latents is not None and torch.is_tensor(latents)) else None,
            'text_data': text_data if text_data else None,
            'metadata': metadata
        }

    @staticmethod
    def _save_json(path: Path, data: Dict):
        """
        Save JSON data with optimized settings. Switch to orjson if installed.
        """
        try:
            import orjson
            with open(path, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_COMPACT))
        except ImportError:
            with open(path, 'w') as f:
                json.dump(data, f, separators=(',', ':'))

    def _log_debug_info(self, processed_item: Dict[str, Any]):
        """Log debug information for troubleshooting."""
        logger.debug(f"Processed item keys: {processed_item.keys()}")
        logger.debug(f"Latents type: {type(processed_item.get('latents'))}")
        if 'latents' in processed_item:
            logger.debug(f"Is latents tensor? {torch.is_tensor(processed_item['latents'])}")

    async def load_cached_item(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load item from cache with optimized parallel loading."""
        try:
            # Check memory cache for metadata
            if self.config.use_memory_cache:
                cache_key = str(image_path)
                if cache_key in self._memory_cache:
                    self._memory_cache.move_to_end(cache_key)
                    # If metadata is in the small metadata cache, attach it
                    if cache_key in self._metadata_memory_cache:
                        self._memory_cache[cache_key]['metadata'] = self._metadata_memory_cache[cache_key]
                    return self._memory_cache[cache_key]
            
            cache_paths = self.get_cache_paths(image_path)
            
            # Quick existence check
            if not all(p.exists() for p in cache_paths.values()):
                return None
            
            # Load data in parallel
            load_tasks = [
                self._io_pool.submit(torch.load, cache_paths['latent']) if cache_paths['latent'].exists() else None,
                self._io_pool.submit(torch.load, cache_paths['text']) if cache_paths['text'].exists() else None,
                self._io_pool.submit(self._load_json, cache_paths['metadata']) if cache_paths['metadata'].exists() else None
            ]
            
            # Wait for all loads to complete
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: [task.result() if task else None for task in load_tasks]
            )
            
            cached_item = {
                'latents': results[0],
                'text_data': results[1],
                'metadata': results[2]
            }
            
            # Update memory cache
            if self.config.use_memory_cache:
                self._memory_cache[str(image_path)] = cached_item
                # Save metadata separately in our small memory cache
                if cached_item.get('metadata'):
                    self._metadata_memory_cache[str(image_path)] = cached_item['metadata']
                if len(self._memory_cache) > self._max_memory_items:
                    self._memory_cache.popitem(last=False)
            
            return cached_item
            
        except Exception as e:
            logger.error(f"Error loading cached item {image_path}: {e}")
            return None

    @staticmethod
    def _load_json(path: Path) -> Dict:
        """
        Load JSON data. If orjson is installed, it's generally faster 
        but be mindful of data type handling.
        """
        try:
            import orjson
            with open(path, 'rb') as f:
                return orjson.loads(f.read())
        except ImportError:
            with open(path) as f:
                return json.load(f)

    async def cleanup(self):
        """Clean up resources."""
        try:
            self._memory_cache.clear()
            self._io_pool.shutdown(wait=True)
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Successfully cleaned up cache manager resources")
        except Exception as e:
            logger.error(f"Error during cache manager cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when cache manager is deleted."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during cache manager deletion: {e}")