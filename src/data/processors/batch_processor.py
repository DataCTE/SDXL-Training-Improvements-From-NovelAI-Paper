# src/data/processors/batch_processor.py
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
from PIL import Image
import asyncio
import time
import gc
from multiprocessing import cpu_count
import concurrent.futures

# Internal imports from utils
from src.data.processors.utils.batch_utils import (
    BatchProcessor as GenericBatchProcessor,
    process_in_chunks,
)
from src.data.processors.utils.system_utils import (
    create_thread_pool,
    get_optimal_workers,
    get_gpu_memory_usage,
    cleanup_processor
)
from src.data.processors.utils.progress_utils import (
    log_progress,
    create_progress_tracker,
    update_tracker,
    ProgressStats
)
from src.data.processors.utils.image_utils import load_and_validate_image
from src.config.config import BatchProcessorConfig  # Import the new config

# Internal imports from processors
from src.data.processors.cache_manager import CacheManager
from src.data.processors.text_processor import TextProcessor
from src.data.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class BatchProcessor(GenericBatchProcessor):
    """
    Process batches of images and text with GPU optimization and chunking.
    """

    def __init__(
        self,
        config: BatchProcessorConfig,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        cache_manager: CacheManager,
        vae
    ):
        """
        Initialize with consolidated config.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.cache_manager = cache_manager
        self.vae = vae

        # Max concurrency for item-level tasks
        self.num_workers = min(
            get_optimal_workers(),
            config.num_workers or cpu_count()
        )

        # Initialize internal queues
        self.process_queue = asyncio.Queue(maxsize=self.config.batch_size * 2)
        self.result_queue = asyncio.Queue(maxsize=self.config.batch_size * 2)

        # Create thread pool for asynchronous operations
        self.thread_pool = create_thread_pool(
            self.num_workers,
            thread_name_prefix='batch_worker'
        )

        # Track active tasks
        self.active_tasks = set()

        self.logger.info(
            f"Initialized BatchProcessor:\n"
            f"- Device: {config.device}\n"
            f"- Batch size: {config.batch_size}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Prefetch factor: {config.prefetch_factor}\n"
            f"- Max memory usage: {config.max_memory_usage:.1%}"
        )

    async def _load_and_validate_image(self, image_path: str):
        """
        Load and validate an image file asynchronously.
        """
        try:
            img_path = Path(image_path)
            return await asyncio.to_thread(
                load_and_validate_image,
                img_path,
                config=self.config  # If your load_and_validate_image uses self.config
            )
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    async def process_dataset(
        self,
        items: List[str],
        progress_callback: Optional[callable] = None
    ):
        """
        Process an entire dataset of item paths. 
        Items are chunked, processed in parallel, and results aggregated.
        """
        import math
        from src.data.processors.utils.progress_utils import (
            create_progress_tracker,
            update_tracker,
            log_progress
        )

        # Create progress tracker with batch info
        tracker = create_progress_tracker(
            total_items=len(items),
            batch_size=self.config.batch_size,
            device=self.config.device
        )

        # Helper to chunk item paths
        def chunker(seq, size):
            for pos in range(0, len(seq), size):
                yield seq[pos:pos + size]

        all_results = []
        for chunk_id, chunk_items in enumerate(chunker(items, self.config.batch_size)):
            results_chunk, stats_chunk = await self._process_chunk(chunk_items, tracker)
            all_results.extend(results_chunk)

            # Fire optional progress callback
            if progress_callback:
                progress_callback(len(all_results), stats_chunk)

            # Print progress occasionally
            if chunk_id % 10 == 0:
                log_progress(tracker, prefix=f"Batch {chunk_id}")

        # Finalize
        final_stats = {
            'processed': tracker.processed_items,
            'failed': tracker.failed_items,
            'cache_hits': tracker.cache_hits,
            'cache_misses': tracker.cache_misses,
            'error_types': tracker.error_types
        }
        return all_results, final_stats

    async def _process_chunk(self, chunk: List[str], tracker):
        """
        Process a chunk of item paths concurrently.
        """
        import torch
        import asyncio
        from pathlib import Path
        from src.data.processors.utils.progress_utils import update_tracker

        chunk_results = []
        chunk_stats = {
            'total': 0,
            'errors': 0,
            'skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_types': {},
            'processed_items': []
        }

        async def process_item(item_path: str):
            try:
                # Check cache
                cache_result = await self.cache_manager.load_cached_item(item_path)
                if cache_result is not None:
                    chunk_stats['cache_hits'] += 1
                    update_tracker(tracker, processed=1, cache_hits=1)
                    return cache_result

                # Load/validate image
                img = await self._load_and_validate_image(item_path)
                if img is None:
                    chunk_stats['skipped'] += 1
                    update_tracker(tracker, failed=1, error_type='invalid_image')
                    return None

                # Process image
                processed_image = await self.image_processor.process_image(img)
                if not processed_image:
                    chunk_stats['skipped'] += 1
                    update_tracker(tracker, failed=1, error_type='image_processing_failed')
                    return None

                # Process text
                text_path = Path(item_path).with_suffix('.txt')
                text_result = await self.text_processor.process_text_file(text_path)
                if not text_result:
                    chunk_stats['errors'] += 1
                    update_tracker(tracker, failed=1, error_type='text_processing_failed')
                    return None

                text_data, tags = text_result

                # Create final item
                item = {
                    'image_path': item_path,
                    'processed_image': processed_image,
                    'text_data': text_data,
                    'tags': tags
                }

                # Cache item
                await self.cache_manager.cache_item(item_path, item)

                chunk_stats['total'] += 1
                chunk_stats['cache_misses'] += 1
                update_tracker(tracker, processed=1, cache_misses=1)
                return item

            except Exception as e:
                chunk_stats['errors'] += 1
                etype = type(e).__name__
                chunk_stats['error_types'][etype] = chunk_stats['error_types'].get(etype, 0) + 1
                update_tracker(tracker, failed=1, error_type=etype)
                self.logger.error(f"Error processing item {item_path}: {str(e)}")
                return None

        item_tasks = [asyncio.create_task(process_item(p)) for p in chunk]
        processed_batch = await asyncio.gather(*item_tasks)

        # Gather valid items
        for item in processed_batch:
            if item is not None:
                chunk_results.append(item)

        return chunk_results, chunk_stats

    async def cleanup(self):
        """
        Optional cleanup for thread pools, GPU caches, etc.
        """
        import gc
        import torch
        self.logger.info("BatchProcessor cleanup invoked.")
        self.thread_pool.shutdown(wait=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            self.logger.error(f"BatchProcessor destructor error: {e}")