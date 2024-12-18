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
    """Batch sampler that groups images with similar aspect ratios for efficient processing."""
    
    def __init__(
        self,
        dataset,  # NovelAIDataset
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        max_consecutive_batch_samples: int = 2,
        min_bucket_length: int = 1,
        debug_mode: bool = False,
        prefetch_factor: Optional[int] = None,
        bucket_manager: Optional[BucketManager] = None,
        config: Optional['NovelAIDatasetConfig'] = None  # Add config parameter
    ):
        """Initialize using dataset's bucket information and optimal thread configuration."""
        super().__init__(dataset)
        
        # Use config if provided, otherwise use parameters
        if config:
            self.batch_size = config.batch_size
            self.shuffle = config.shuffle
            self.drop_last = config.drop_last
            self.max_consecutive_batch_samples = config.max_consecutive_batch_samples
            self.min_bucket_length = config.min_bucket_length
            self.debug_mode = config.debug_mode
            self.prefetch_factor = config.prefetch_factor
        else:
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.max_consecutive_batch_samples = max_consecutive_batch_samples
            self.min_bucket_length = min_bucket_length
            self.debug_mode = debug_mode
            self.prefetch_factor = prefetch_factor
        
        # Get optimal thread configuration
        thread_config = get_optimal_thread_config()
        self.prefetch_factor = prefetch_factor or thread_config.prefetch_factor
        
        # Initialize batch processor with optimal configuration
        self.batch_processor = BatchProcessor(
            image_processor=dataset.image_processor,
            cache_manager=dataset.cache_manager,
            vae=dataset.vae,
            device=dataset.device,
            batch_size=batch_size,
            prefetch_factor=self.prefetch_factor
        )
        
        # Validate inputs
        if batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if max_consecutive_batch_samples < 1:
            raise ValueError(f"Max consecutive samples must be positive, got {max_consecutive_batch_samples}")
        if min_bucket_length < 1:
            raise ValueError(f"Min bucket length must be positive, got {min_bucket_length}")
        
        # Store weak references to avoid memory leaks
        from weakref import proxy
        self.dataset = proxy(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_consecutive_batch_samples = max_consecutive_batch_samples
        self.min_bucket_length = min_bucket_length
        self.debug_mode = debug_mode
        
        # Get bucket manager from dataset
        if not hasattr(dataset, 'bucket_manager') or not isinstance(dataset.bucket_manager, BucketManager):
            raise ValueError("Dataset must have a valid BucketManager instance")
        self.bucket_manager = bucket_manager or dataset.bucket_manager
        
        # Initialize state
        self.epoch = 0
        self.total_batches = 0
        
        # Performance tracking
        self.creation_time = time.time()
        
        try:
            # Assign samples to buckets
            self.bucket_manager.assign_to_buckets(dataset.items, shuffle=shuffle)
            
            # Calculate number of batches
            self.total_batches = self._calculate_total_batches()
            
            # Log initialization stats
            self._log_initialization_stats()
            
        except Exception as e:
            logger.error(f"Failed to initialize sampler: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        self.creation_time = time.time() - self.creation_time

    def _log_initialization_stats(self):
        """Log detailed initialization statistics."""
        stats = self.bucket_manager.get_stats()
        logger.info(
            f"\nSampler Initialization Complete:"
            f"\n- Buckets: {stats['total_buckets']:,}"
            f"\n- Total samples: {stats['total_samples']:,}"
            f"\n- Total batches: {self.total_batches:,}"
            f"\n- Batch size: {self.batch_size}"
            f"\n- Prefetch factor: {self.prefetch_factor}"
            f"\n- Initialization time: {format_time(self.creation_time)}"
            f"\n\nBucket Statistics:"
            f"\n- Max bucket size: {stats['max_bucket_size']:,}"
            f"\n- Min bucket size: {stats['min_bucket_size']:,}"
            f"\n- Avg bucket size: {stats['avg_bucket_size']:.1f}"
        )
        
        if self.debug_mode:
            # Log detailed bucket distribution
            logger.debug("\nBucket Distribution:")
            for key, count in sorted(stats['samples_per_bucket'].items()):
                logger.debug(f"- {key}: {count:,} samples")

    def _calculate_total_batches(self) -> int:
        """Calculate total number of batches efficiently."""
        total = 0
        for bucket in self.bucket_manager.buckets.values():
            if len(bucket) >= self.min_bucket_length:
                if self.drop_last:
                    total += len(bucket) // self.batch_size
                else:
                    total += math.ceil(len(bucket) / self.batch_size)
        return total

    async def __aiter__(self) -> AsyncIterator[List[int]]:
        """Create and yield batches for current epoch efficiently."""
        try:
            # Create progress tracker
            tracker = create_progress_tracker(
                total_items=len(self.dataset),
                desc="Processing batches",
                unit="images"
            )
            
            epoch_start = time.time()
            active_buckets = set(self.bucket_manager.buckets.keys())
            consecutive_samples = defaultdict(int)
            
            while active_buckets and tracker.processed_items < tracker.total_items:
                # Select bucket based on sampling strategy
                selected_key = self._select_bucket(active_buckets, consecutive_samples)
                bucket = self.bucket_manager.get_bucket_by_key(selected_key)
                
                if not bucket:
                    active_buckets.remove(selected_key)
                    continue
                    
                width, height = map(int, selected_key.split('x'))
                
                # Get batch from bucket
                batch = bucket.get_next_batch(self.batch_size)
                
                if batch:
                    # Create batch items with dimensions and original sizes
                    batch_items = []
                    for idx in batch:
                        item = self.dataset.items[idx]
                        img_path = item['image_path']
                        # Get original image size for SDXL conditioning
                        with Image.open(img_path) as img:
                            original_size = (img.height, img.width)
                        batch_items.append({
                            **item,
                            'original_size': original_size,
                            'width': width,
                            'height': height
                        })
                    
                    # Process batch
                    try:
                        processed_items, batch_stats = await self._process_batch_with_retry(
                            batch_items=batch_items,
                            width=width,
                            height=height
                        )
                        
                        if processed_items:
                            update_tracker(tracker, **batch_stats)
                            yield batch
                            
                        # Memory cleanup
                        del processed_items
                        del batch_items
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        update_tracker(tracker, failed=len(batch), error_type='batch_processing_error')
                        
                elif not self.drop_last and bucket.remaining_samples() > 0:
                    # Handle final partial batch similarly to regular batch
                    final_batch = bucket.get_next_batch(bucket.remaining_samples())
                    if final_batch:
                        # Process final batch
                        batch_items = [
                            {**self.dataset.items[idx], 'width': width, 'height': height}
                            for idx in final_batch
                        ]
                        try:
                            processed_items, batch_stats = await self._process_batch_with_retry(
                                batch_items=batch_items,
                                width=width,
                                height=height
                            )
                            
                            if processed_items:
                                update_tracker(
                                    tracker,
                                    processed=len(processed_items),
                                    cache_hits=batch_stats.get('cache_hits', 0),
                                    cache_misses=batch_stats.get('cache_misses', 0)
                                )
                                yield final_batch
                                
                                # Clear final batch data
                                del processed_items
                                del batch_items
                                gc.collect()
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            logger.error(f"Error processing final batch: {str(e)}")
                            logger.error(traceback.format_exc())
                            update_tracker(tracker, failed=len(final_batch), error_type='final_batch_error')
                            
                            # Clear any failed batch data
                            del batch_items
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                    active_buckets.remove(selected_key)
                else:
                    active_buckets.remove(selected_key)
                
                # Clear memory after each batch
                gc.collect()
                torch.cuda.empty_cache()
                await asyncio.sleep(0.01)
            
            # Log final epoch stats
            epoch_time = time.time() - epoch_start
            logger.info(
                f"\nEpoch {self.epoch} completed:"
                f"\n- Time: {format_time(epoch_time)}"
                f"\n- Processed: {tracker.processed_items}/{tracker.total_items}"
                f"\n- Success rate: {(tracker.processed_items/tracker.total_items)*100:.1f}%"
                f"\n- Processing rate: {tracker.rate:.1f} items/s"
                f"\n- Cache hits/misses: {tracker.cache_hits}/{tracker.cache_misses}"
                f"\n- Failed items: {tracker.failed_items}"
                f"\n- Error types: {tracker.error_types}"
            )
            
            # Increment epoch
            self.epoch += 1
            
            # Clear epoch data
            del tracker
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in epoch iteration: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def cleanup(self):
        """Clean up sampler resources."""
        await cleanup_processor(self)

    def __del__(self):
        """Ensure cleanup when sampler is deleted."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during sampler deletion: {e}")

    def __len__(self) -> int:
        return self.total_batches

    def _manage_memory(self):
        """Manage memory usage during batch processing."""
        if hasattr(self.dataset, 'cache_manager'):
            self.dataset.cache_manager.clear_unused_cache()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    async def _process_batch_with_memory_management(self, batch_items, width, height):
        """Process batch with memory management."""
        try:
            processed_items, batch_stats = await self.batch_processor.process_batch(
                batch_items=batch_items,
                width=width,
                height=height,
                cache_manager=self.dataset.cache_manager
            )
            
            # Clear memory after processing
            self._manage_memory()
            
            return processed_items, batch_stats
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self._manage_memory()
            raise

    async def _process_batch_with_retry(self, batch_items, width, height, max_retries=3):
        """Process batch with retry logic."""
        for attempt in range(max_retries):
            try:
                return await self._process_batch_with_memory_management(
                    batch_items=batch_items,
                    width=width,
                    height=height
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to process batch after {max_retries} attempts")
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                self._manage_memory()

# Add detailed progress tracking
class BatchProgress:
    def __init__(self, total_items: int, desc: str = "Processing batches"):
        self.tracker = create_progress_tracker(total_items, desc)
        self.start_time = time.time()
        self.batch_times: List[float] = []
        self.memory_usage: List[float] = []
        
    def update(self, batch_size: int, batch_stats: Dict[str, Any]):
        self.tracker.update(batch_size)
        self.batch_times.append(time.time() - self.start_time)
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            
    def get_stats(self) -> Dict[str, Any]:
        return {
            "processed_items": self.tracker.processed_items,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
            "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "total_time": time.time() - self.start_time
        }