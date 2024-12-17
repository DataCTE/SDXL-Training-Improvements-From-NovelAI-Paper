from typing import List, Optional, Iterator, Dict, AsyncIterator
import torch
from torch.utils.data import Sampler
import logging
import math
import time
import traceback
import gc
import asyncio

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
        prefetch_factor: Optional[int] = None
    ):
        """Initialize using dataset's bucket information and optimal thread configuration."""
        super().__init__(dataset)
        
        # Get optimal thread configuration
        thread_config = get_optimal_thread_config()
        self.prefetch_factor = prefetch_factor or thread_config.prefetch_factor
        
        # Initialize batch processor with optimal configuration
        self.batch_processor = BatchProcessor(
            image_processor=dataset.image_processor,
            cache_manager=dataset.cache_manager,
            text_embedder=dataset.text_embedder,
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
        self.bucket_manager = dataset.bucket_manager
        
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
            epoch_start = time.time()
            logger.info(f"\nStarting epoch {self.epoch}")
            
            # Create progress tracker for this epoch
            tracker = create_progress_tracker(
                total_items=self.total_batches,
                batch_size=self.batch_size,
                device=self.dataset.device
            )
            
            # Shuffle buckets if needed
            if self.shuffle:
                self.bucket_manager.shuffle_buckets(self.epoch)
            
            # Track consecutive samples from each bucket
            consecutive_samples = {key: 0 for key in self.bucket_manager.buckets}
            active_buckets = set(key for key, bucket in self.bucket_manager.buckets.items() 
                               if len(bucket) >= self.min_bucket_length)
            
            # Process buckets in order of remaining samples
            while active_buckets:
                # Find bucket with most remaining samples
                max_remaining = -1
                selected_key = None
                
                for key in active_buckets:
                    bucket = self.bucket_manager.buckets[key]
                    remaining = bucket.remaining_samples()
                    
                    if remaining == 0:
                        active_buckets.remove(key)
                        continue
                        
                    if consecutive_samples[key] >= self.max_consecutive_batch_samples:
                        continue
                        
                    if remaining > max_remaining:
                        max_remaining = remaining
                        selected_key = key
                
                if selected_key is None:
                    # Reset consecutive counts if all buckets hit limit
                    if any(bucket.remaining_samples() > 0 
                          for bucket in self.bucket_manager.buckets.values()):
                        consecutive_samples = {key: 0 for key in self.bucket_manager.buckets}
                        continue
                    break
                
                # Get batch from selected bucket
                bucket = self.bucket_manager.buckets[selected_key]
                batch = bucket.get_next_batch(self.batch_size)
                
                if batch:
                    consecutive_samples[selected_key] += 1
                    for other_key in consecutive_samples:
                        if other_key != selected_key:
                            consecutive_samples[other_key] = 0
                            
                    # Get bucket dimensions for batch processing
                    width, height = bucket.width, bucket.height
                    
                    # Create batch items with dimensions
                    batch_items = [
                        {**self.dataset.items[idx], 'width': width, 'height': height}
                        for idx in batch
                    ]
                    
                    # Process batch using BatchProcessor
                    try:
                        processed_items, batch_stats = await self.batch_processor.process_batch(
                            batch_items=batch_items,
                            width=width,
                            height=height,
                            cache_manager=self.dataset.cache_manager
                        )
                        
                        # Update progress tracker
                        if processed_items:
                            update_tracker(
                                tracker,
                                processed=len(processed_items),
                                cache_hits=batch_stats.get('cache_hits', 0),
                                cache_misses=batch_stats.get('cache_misses', 0)
                            )
                            
                            if batch_stats.get('errors', 0) > 0:
                                for error_type, count in batch_stats.get('error_types', {}).items():
                                    update_tracker(tracker, failed=count, error_type=error_type)
                            
                            yield batch
                            
                            # Clear processed items and batch references
                            del processed_items
                            del batch_items
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                        # Log progress periodically
                        if tracker.should_log():
                            extra_stats = {
                                'active_buckets': len(active_buckets),
                                'current_bucket': f"{width}x{height}",
                                'consecutive_samples': consecutive_samples[selected_key],
                                'error_types': tracker.error_types
                            }
                            log_progress(tracker, prefix="Batch Processing: ", extra_stats=extra_stats)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        logger.error(traceback.format_exc())
                        update_tracker(tracker, failed=len(batch), error_type='batch_processing_error')
                        
                        # Clear any failed batch data
                        del batch_items
                        gc.collect()
                        torch.cuda.empty_cache()
                    
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
                            processed_items, batch_stats = await self.batch_processor.process_batch(
                                batch_items=batch_items,
                                width=width,
                                height=height,
                                cache_manager=self.dataset.cache_manager
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
        try:
            # Clean up batch processor
            if hasattr(self, 'batch_processor'):
                await self.batch_processor.cleanup()
                del self.batch_processor
            
            # Clear bucket references
            if hasattr(self, 'bucket_manager'):
                del self.bucket_manager
            
            # Clear dataset reference
            if hasattr(self, 'dataset'):
                del self.dataset
            
            # Clear any remaining data
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("Successfully cleaned up sampler resources")
            
        except Exception as e:
            logger.error(f"Error during sampler cleanup: {e}")

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