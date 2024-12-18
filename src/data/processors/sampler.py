from typing import List, Optional, Iterator, Dict, AsyncIterator, Any
import torch
from torch.utils.data import Sampler
import logging
import math
import time
import traceback
import gc
import asyncio
from PIL import Image
from collections import defaultdict

# Internal imports from utils
from src.data.processors.utils.progress_utils import (
    create_progress_tracker,
    update_tracker,
    log_progress,
    format_time
)

# Internal imports from processors
from src.data.processors.bucket import BucketManager
from src.data.processors.utils.thread_config import get_optimal_thread_config
from src.data.processors.batch_processor import BatchProcessor
from src.data.processors.utils.system_utils import cleanup_processor
from src.config.config import NovelAIDatasetConfig

logger = logging.getLogger(__name__)

class AspectBatchSampler(Sampler[List[int]]):
    """
    Sampler that handles aspects, bucket-based batching, and concurrency for dataset items.
    Includes retry logic, memory management, and progress tracking.
    """

    def __init__(
        self,
        dataset,  # e.g. NovelAIDataset
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        max_consecutive_batch_samples: int = 2,
        min_bucket_length: int = 1,
        debug_mode: bool = False,
        prefetch_factor: Optional[int] = None,
        bucket_manager: Optional[BucketManager] = None,
        config: Optional['NovelAIDatasetConfig'] = None
    ):
        import logging
        import traceback
        from torch.utils.data import Sampler
        from weakref import proxy
        from src.data.processors.batch_processor import BatchProcessor

        self.logger = logging.getLogger(__name__)
        self.dataset = proxy(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_consecutive_batch_samples = max_consecutive_batch_samples
        self.min_bucket_length = min_bucket_length
        self.debug_mode = debug_mode
        self.prefetch_factor = prefetch_factor
        self.config = config

        if bucket_manager is None:
            if not hasattr(dataset, 'bucket_manager') or not isinstance(dataset.bucket_manager, BucketManager):
                raise ValueError("Dataset must have a valid BucketManager")
            self.bucket_manager = dataset.bucket_manager
        else:
            self.bucket_manager = bucket_manager

        # BatchProcessor instance for batch-level parallel
        try:
            self.batch_processor = BatchProcessor(
                config=dataset.config.batch_processor_config,
                image_processor=dataset.image_processor,
                text_processor=dataset.text_processor,
                cache_manager=dataset.cache_manager,
                vae=dataset.vae
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize BatchProcessor: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

        # Validate
        if batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if max_consecutive_batch_samples < 1:
            raise ValueError(f"max_consecutive_batch_samples must be positive, got {max_consecutive_batch_samples}")
        if min_bucket_length < 1:
            raise ValueError(f"min_bucket_length must be positive, got {min_bucket_length}")

        # Sampler state
        self.epoch = 0
        self.total_batches = 0

        # Possibly shuffle or set random seed
        # ...

    async def __aiter__(self) -> AsyncIterator[List[int]]:
        """
        Asynchronous iterator that yields lists of indices forming each batch.
        Includes concurrency logic and memory management.
        """
        import gc
        import torch
        import time
        import asyncio
        from PIL import Image
        from src.data.processors.utils.progress_utils import (
            create_progress_tracker, update_tracker, log_progress, format_time
        )
        from src.data.processors.utils.system_utils import cleanup_processor
        import traceback

        # Start epoch
        epoch_start = time.time()
        tracker = create_progress_tracker(
            total_items=len(self.dataset.items),
            batch_size=self.batch_size,
            device='cuda'  # or 'cpu' if no GPU
        )

        # Access your bucket manager
        active_buckets = self.bucket_manager.get_active_buckets()

        try:
            while active_buckets:
                selected_key = self.bucket_manager.select_next_bucket_key(active_buckets)
                bucket = self.bucket_manager.buckets[selected_key]
                width, height = bucket.target_size

                # Retrieve next batch
                batch = bucket.get_next_batch(self.batch_size)
                if batch:
                    # Prepare items
                    batch_items = []
                    for idx in batch:
                        item = self.dataset.items[idx]
                        img_path = item['image_path']
                        # Get original image size
                        with Image.open(img_path) as img:
                            original_size = (img.height, img.width)

                        batch_items.append({
                            **item,
                            'original_size': original_size,
                            'width': width,
                            'height': height
                        })

                    # Process current batch
                    try:
                        processed_items, batch_stats = await self._process_batch_with_retry(
                            batch_items=batch_items,
                            width=width,
                            height=height
                        )

                        if processed_items:
                            self.total_batches += 1
                            update_tracker(tracker, **batch_stats)
                            yield batch

                        # Memory cleanup
                        del processed_items
                        del batch_items
                        gc.collect()
                        torch.cuda.empty_cache()

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {str(e)}")
                        update_tracker(tracker, failed=len(batch), error_type='batch_processing_error')
                elif not self.drop_last and bucket.remaining_samples() > 0:
                    # Final partial batch
                    final_batch = bucket.get_next_batch(bucket.remaining_samples())
                    if final_batch:
                        batch_items = []
                        for idx in final_batch:
                            item = self.dataset.items[idx]
                            batch_items.append({
                                **item,
                                'width': width,
                                'height': height
                            })

                        try:
                            processed_items, batch_stats = await self._process_batch_with_retry(
                                batch_items=batch_items,
                                width=width,
                                height=height
                            )
                            if processed_items:
                                self.total_batches += 1
                                update_tracker(
                                    tracker,
                                    processed=len(processed_items),
                                    cache_hits=batch_stats.get('cache_hits', 0),
                                    cache_misses=batch_stats.get('cache_misses', 0)
                                )
                                yield final_batch

                            del processed_items
                            del batch_items
                            gc.collect()
                            torch.cuda.empty_cache()

                        except Exception as e:
                            self.logger.error(f"Error processing final batch: {str(e)}")
                            update_tracker(tracker, failed=len(final_batch), error_type='final_batch_error')
                            del batch_items
                            gc.collect()
                            torch.cuda.empty_cache()

                    active_buckets.remove(selected_key)
                else:
                    # No more items in this bucket
                    active_buckets.remove(selected_key)

                # Housekeeping
                gc.collect()
                torch.cuda.empty_cache()
                await asyncio.sleep(0.01)

            # End of epoch
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"\nEpoch {self.epoch} completed:"
                f"\n- Time: {format_time(epoch_time)}"
                f"\n- Processed: {tracker.processed_items}/{tracker.total_items}"
                f"\n- Success rate: {(tracker.processed_items/tracker.total_items)*100:.1f}%"
                f"\n- Processing rate: {tracker.rate:.1f} items/s"
                f"\n- Cache hits/misses: {tracker.cache_hits}/{tracker.cache_misses}"
                f"\n- Failed items: {tracker.failed_items}"
                f"\n- Error types: {tracker.error_types}"
            )

            self.epoch += 1
            del tracker
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Error in epoch iteration: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def cleanup(self):
        """
        Clean up sampler resources, plus batch_processor.
        """
        import gc
        from src.data.processors.utils.system_utils import cleanup_processor

        await cleanup_processor(self)
        if self.batch_processor:
            await self.batch_processor.cleanup()
        gc.collect()

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            self.logger.error(f"Error during sampler deletion: {e}")

    async def _process_batch_with_retry(self, batch_items, width, height, max_retries=3):
        """
        Process batch with memory management and retry logic.
        """
        import asyncio

        for attempt in range(max_retries):
            try:
                return await self._process_batch_with_memory_management(
                    batch_items=batch_items,
                    width=width,
                    height=height
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to process batch after {max_retries} attempts.")
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                self._manage_memory()

    async def _process_batch_with_memory_management(self, batch_items, width, height):
        """
        Actually process the batch, then do memory cleanup.
        """
        import gc
        import torch

        processed_items, batch_stats = await self.batch_processor.process_batch(
            batch_items=batch_items,
            width=width,
            height=height,
            cache_manager=self.dataset.cache_manager
        )

        self._manage_memory()
        return processed_items, batch_stats

    def _manage_memory(self):
        """
        Basic memory cleanup after each batch.
        """
        import gc
        import torch
        if hasattr(self.dataset, 'cache_manager'):
            # Potentially clear or remove old cache items
            # e.g., self.dataset.cache_manager.clear_unused_cache()
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self) -> int:
        return self.total_batches