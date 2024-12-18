# src/data/processors/cache_manager.py
import os
import logging
import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
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
    """
    Faster CacheManager with orjson, single thread pool, and minimal disk checks.
    """

    def __init__(self, config: CacheConfig, io_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.cache_dir = Path(config.cache_dir)

        subdirs = ["latents", "text", "metadata"]
        for sd in subdirs:
            ensure_dir(self.cache_dir / sd)

        self.latents_dir = self.cache_dir / "latents"
        self.text_dir = self.cache_dir / "text"
        self.metadata_dir = self.cache_dir / "metadata"

        self._memory_cache = OrderedDict()
        self._metadata_memory_cache = OrderedDict()
        self._max_memory_items = getattr(config, "max_memory_items", 1000)

        # Possibly reuse an external thread pool to reduce overhead
        if io_pool is not None:
            self._io_pool = io_pool
        else:
            worker_count = min(16, (os.cpu_count() or 8))
            self._io_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="cache_io_"
            )

        self._save_options = {
            '_use_new_zipfile_serialization': True,
            'pickle_protocol': 4
        }

        self.logger.info(
            f"CacheManager init: {self.cache_dir}\n"
            f"Memory cache: {config.use_memory_cache}, max items: {self._max_memory_items}\n"
            f"Thread workers: {self._io_pool._max_workers}"
        )

    def get_cache_paths(self, image_path: Union[str, Path]) -> dict:
        image_path = Path(image_path)
        name = image_path.stem
        return {
            'latent': self.latents_dir / f"{name}.pt",
            'text': self.text_dir / f"{name}.pt",
            'metadata': self.metadata_dir / f"{name}.json"
        }

    async def cache_item(self, image_path: Union[str, Path], processed_item: dict):
        """
        Save latents, text embeddings, metadata in parallel if present.
        """
        paths = self.get_cache_paths(image_path)

        latents = processed_item.get('latents')
        text_data = self._prepare_text_data(processed_item)
        metadata = self._prepare_metadata(processed_item)

        tasks = []
        if latents is not None and isinstance(latents, torch.Tensor):
            tasks.append(self._io_pool.submit(torch.save, latents.cpu(), paths['latent'], **self._save_options))
        if text_data:
            tasks.append(self._io_pool.submit(torch.save, text_data, paths['text'], **self._save_options))
        if any(metadata.values()):
            tasks.append(self._io_pool.submit(self._save_json, paths['metadata'], metadata))

        if tasks:
            await asyncio.get_event_loop().run_in_executor(None, lambda: [t.result() for t in tasks])

        # Memory cache
        if self.config.use_memory_cache:
            self._update_memory_cache(image_path, latents, text_data, metadata)

    def _prepare_text_data(self, item: dict):
        import torch
        text_data = {}
        for key in ['prompt_embeds', 'pooled_prompt_embeds', 'tag_weights']:
            if key in item and isinstance(item[key], torch.Tensor):
                text_data[key] = item[key].cpu()
        return text_data

    def _prepare_metadata(self, item: dict):
        # Minimal example, expand if needed
        return {
            'original_size': item.get('original_size'),
            'crop_top_left': item.get('crop_top_left'),
            'target_size': (item.get('width'), item.get('height'))
        }

    def _save_json(self, path: Path, data: dict):
        """Use orjson if available, else fallback to normal JSON."""
        try:
            import orjson
            with open(path, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_COMPACT))
        except ImportError:
            with open(path, 'w') as f:
                json.dump(data, f, separators=(',', ':'))

    def _update_memory_cache(self, image_path: Union[str, Path], latents, text_data, metadata):
        cache_key = str(image_path)
        if len(self._memory_cache) >= self._max_memory_items:
            self._memory_cache.popitem(last=False)
        self._memory_cache[cache_key] = {
            'latents': latents.cpu() if latents is not None and isinstance(latents, torch.Tensor) else None,
            'text_data': text_data,
            'metadata': metadata
        }
        if any(metadata.values()):
            self._metadata_memory_cache[cache_key] = metadata

    async def load_cached_item(self, image_path: Union[str, Path]):
        cache_key = str(image_path)
        if self.config.use_memory_cache and cache_key in self._memory_cache:
            # LRU logic
            self._memory_cache.move_to_end(cache_key)
            if cache_key in self._metadata_memory_cache:
                self._memory_cache[cache_key]['metadata'] = self._metadata_memory_cache[cache_key]
            return self._memory_cache[cache_key]

        paths = self.get_cache_paths(image_path)
        if not all(p.exists() for p in paths.values()):
            return None

        tasks = []
        if paths['latent'].exists():
            tasks.append(self._io_pool.submit(torch.load, paths['latent']))
        if paths['text'].exists():
            tasks.append(self._io_pool.submit(torch.load, paths['text']))
        if paths['metadata'].exists():
            tasks.append(self._io_pool.submit(self._load_json, paths['metadata']))

        if not tasks:
            return None

        results = await asyncio.get_event_loop().run_in_executor(None, lambda: [t.result() for t in tasks])
        latents_data = results[0] if len(results) > 0 else None
        text_data = results[1] if len(results) > 1 else None
        metadata = results[2] if len(results) > 2 else None

        out = {
            'latents': latents_data,
            'text_data': text_data,
            'metadata': metadata
        }

        # Store in memory if enabled
        if self.config.use_memory_cache:
            self._memory_cache[cache_key] = out
            if metadata:
                self._metadata_memory_cache[cache_key] = metadata
            if len(self._memory_cache) > self._max_memory_items:
                self._memory_cache.popitem(last=False)

        return out

    def _load_json(self, path: Path):
        """Same fallback logic: try orjson first."""
        try:
            import orjson
            with open(path, 'rb') as f:
                return orjson.loads(f.read())
        except ImportError:
            with open(path, 'r') as f:
                return json.load(f)

    async def cleanup(self):
        self._memory_cache.clear()
        self._metadata_memory_cache.clear()
        self._io_pool.shutdown(wait=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("CacheManager cleanup done.")

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"CacheManager destructor error: {e}")