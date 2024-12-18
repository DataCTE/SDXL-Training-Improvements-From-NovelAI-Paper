# src/data/processors/cache_manager.py
import os
import logging
import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import gc
from weakref import WeakValueDictionary

from src.data.processors.utils.system_utils import get_gpu_memory_usage, cleanup_processor
from src.data.processors.utils.file_utils import ensure_dir
from src.config.config import CacheConfig

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of processed images and text embeddings."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache manager with configuration."""
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        
        # Create cache directories
        self.latents_dir = self.cache_dir / "latents"
        self.text_dir = self.cache_dir / "text"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.latents_dir, self.text_dir, self.metadata_dir]:
            ensure_dir(dir_path)
            
        # Initialize memory cache
        self._memory_cache = WeakValueDictionary()
        
        logger.info(
            f"Initialized CacheManager:\n"
            f"- Cache directory: {self.cache_dir}\n"
            f"- Memory cache enabled: {config.use_memory_cache}\n"
            f"- Cache format: {config.cache_format}"
        )

    def get_cache_paths(self, image_path: str) -> Dict[str, Path]:
        """Get cache paths for an image file."""
        image_path = Path(image_path)
        relative_path = image_path.stem
        
        return {
            'latent': self.latents_dir / f"{relative_path}.pt",
            'text': self.text_dir / f"{relative_path}.pt",
            'metadata': self.metadata_dir / f"{relative_path}.json"
        }

    async def cache_item(self, image_path: str, processed_item: Dict[str, Any]) -> None:
        """Cache processed item data including tag weights."""
        try:
            cache_paths = self.get_cache_paths(image_path)
            
            # Extract data to cache
            latents = processed_item.get('latents')
            text_data = {
                'prompt_embeds': processed_item['prompt_embeds'].cpu(),
                'pooled_prompt_embeds': processed_item['pooled_prompt_embeds'].cpu()
            }
            if 'tag_weights' in processed_item:
                text_data['tag_weights'] = processed_item['tag_weights'].cpu()
                
            metadata = {
                'original_size': processed_item.get('original_size'),
                'crop_top_left': processed_item.get('crop_top_left'),
                'target_size': (processed_item.get('width'), processed_item.get('height'))
            }
            
            # Save latents
            if latents is not None:
                torch.save(
                    latents.cpu(), 
                    cache_paths['latent'],
                    _use_new_zipfile_serialization=True
                )
            
            # Save text embeddings
            if text_data is not None:
                torch.save(
                    text_data,
                    cache_paths['text'],
                    _use_new_zipfile_serialization=True
                )
            
            # Save metadata
            if any(metadata.values()):
                import json
                with open(cache_paths['metadata'], 'w') as f:
                    json.dump(metadata, f)
            
            # Add to memory cache if enabled
            if self.config.use_memory_cache:
                cache_key = str(image_path)
                self._memory_cache[cache_key] = {
                    'latents': latents.cpu() if latents is not None else None,
                    'text_data': text_data if text_data is not None else None,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Error caching item {image_path}: {e}")

    async def load_cached_item(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load item from cache."""
        try:
            # Check memory cache first
            if self.config.use_memory_cache:
                cached = self._memory_cache.get(str(image_path))
                if cached is not None:
                    return cached
            
            cache_paths = self.get_cache_paths(image_path)
            
            if not all(p.exists() for p in cache_paths.values()):
                return None
                
            # Load cached data
            latents = torch.load(cache_paths['latent']) if cache_paths['latent'].exists() else None
            text_data = torch.load(cache_paths['text']) if cache_paths['text'].exists() else None
            
            metadata = None
            if cache_paths['metadata'].exists():
                import json
                with open(cache_paths['metadata']) as f:
                    metadata = json.load(f)
            
            cached_item = {
                'latents': latents,
                'text_data': text_data,
                'metadata': metadata
            }
            
            # Add to memory cache
            if self.config.use_memory_cache:
                self._memory_cache[str(image_path)] = cached_item
                
            return cached_item
            
        except Exception as e:
            logger.error(f"Error loading cached item {image_path}: {e}")
            return None

    async def cleanup(self):
        """Clean up cache manager resources."""
        try:
            # Clear memory cache
            self._memory_cache.clear()
            
            # Force garbage collection
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