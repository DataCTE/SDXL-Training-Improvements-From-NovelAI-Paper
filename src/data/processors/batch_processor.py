# src/data/processors/batch_processor.py
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
from PIL import Image
import asyncio
import time

# Internal imports from utils
from src.data.processors.utils.batch_utils import (
    BatchConfig,
    BatchProcessor as GenericBatchProcessor,
    process_in_chunks,
    calculate_optimal_batch_size
)
from src.data.processors.utils.system_utils import (
    create_thread_pool,
    get_optimal_workers,
    get_gpu_memory_usage,
    get_memory_usage_gb
)
from src.data.processors.utils.progress_utils import (
    log_progress,
    create_progress_tracker,
    update_tracker,
    ProgressStats
)
from src.data.processors.utils.image_utils import load_and_validate_image

# Internal imports from processors
from src.data.processors.cache_manager import CacheManager
from src.data.processors.text_processor import TextProcessor
from src.data.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class BatchProcessor(GenericBatchProcessor):
    """Process batches of images and text with GPU optimization."""
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        cache_manager: CacheManager,
        vae,
        device: torch.device,
        batch_size: Optional[int] = None,
        prefetch_factor: int = 2,
        max_memory_usage: float = 0.9,
        num_workers: Optional[int] = None
    ):
        # Calculate optimal batch size and workers if not provided
        self.batch_size = batch_size or calculate_optimal_batch_size(
            device=device,
            min_batch_size=1,
            max_batch_size=32,
            target_memory_usage=max_memory_usage
        )
        self.num_workers = num_workers or get_optimal_workers()
        
        # Create thread pool for async operations
        self.thread_pool = create_thread_pool(self.num_workers)
        
        # Initialize components
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.cache_manager = cache_manager
        self.vae = vae
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.max_memory_usage = max_memory_usage
        
        # Create batch config
        self.batch_config = BatchConfig(
            batch_size=self.batch_size,
            device=device,
            max_memory_usage=max_memory_usage,
            prefetch_factor=prefetch_factor
        )
        
        logger.info(
            f"Initialized BatchProcessor:\n"
            f"- Device: {device}\n"
            f"- Batch size: {self.batch_size}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Prefetch factor: {prefetch_factor}"
        )
        
    async def _load_and_validate_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and validate an image file."""
        try:
            img_path = Path(image_path)
            return await asyncio.to_thread(
                load_and_validate_image,
                img_path,
                min_size=(256, 256),  # Minimum size for processing
                max_size=(8192, 8192)  # Maximum size supported
            )
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)[:200]}...")
            return None

    async def process_dataset(
        self,
        items: List[str],
        progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process entire dataset with chunking and progress tracking."""
        # Create progress tracker with batch info
        tracker = create_progress_tracker(
            total_items=len(items),
            batch_size=self.batch_size,
            device=self.device
        )
        
        async def process_chunk(chunk: List[str], chunk_id: int) -> Tuple[List[Dict], Dict[str, Any]]:
            """Process a single chunk of items."""
            chunk_items = []
            chunk_stats = {
                'total': 0,
                'errors': 0,
                'skipped': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'error_types': {},
                'processed_items': []
            }
            
            # Process items in parallel using thread pool
            async def process_item(item_path: str) -> Optional[Dict]:
                try:
                    # Check cache first
                    cache_result = await self.cache_manager.get_cached_item(item_path)
                    if cache_result is not None:
                        chunk_stats['cache_hits'] += 1
                        return cache_result
                        
                    # Load and validate image
                    img = await self._load_and_validate_image(item_path)
                    if img is None:
                        chunk_stats['skipped'] += 1
                        update_tracker(tracker, failed=1, error_type='invalid_image')
                        return None
                        
                    # Process image
                    processed_image = await self.image_processor.process_image(img)
                    if processed_image is None:
                        chunk_stats['skipped'] += 1
                        update_tracker(tracker, failed=1, error_type='image_processing_failed')
                        return None
                        
                    # Process text
                    text_path = Path(item_path).with_suffix('.txt')
                    text_result = await self.text_processor.process_text_file(text_path)
                    if text_result is None:
                        chunk_stats['error_types']['text_processing_failed'] = \
                            chunk_stats['error_types'].get('text_processing_failed', 0) + 1
                        update_tracker(tracker, failed=1, error_type='text_processing_failed')
                        return None
                        
                    text_data, tags = text_result
                    
                    # Create item
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
                    logger.error(f"Error processing {item_path}: {str(e)[:200]}...")
                    chunk_stats['errors'] += 1
                    chunk_stats['error_types'][type(e).__name__] = \
                        chunk_stats['error_types'].get(type(e).__name__, 0) + 1
                    update_tracker(tracker, failed=1, error_type=type(e).__name__)
                    return None
            
            # Process chunk items in parallel
            tasks = [process_item(item_path) for item_path in chunk]
            results = await asyncio.gather(*tasks)
            
            # Filter out None results and extend chunk items
            chunk_items.extend([r for r in results if r is not None])
            chunk_stats['processed_items'] = chunk_items
            
            return chunk_items, chunk_stats
        
        # Process chunks
        processed_items, final_stats = await process_in_chunks(
            items=items,
            chunk_size=self.batch_size,
            process_fn=process_chunk,
            progress_callback=lambda n, chunk_stats: self._handle_progress(
                n, chunk_stats, tracker, progress_callback
            )
        )
        
        return processed_items, final_stats
        
    def _handle_progress(
        self,
        n: int,
        chunk_stats: Dict[str, Any],
        tracker: ProgressStats,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """Handle progress updates and callbacks."""
        # Add extra monitoring stats
        extra_stats = {
            'processed': len(chunk_stats.get('processed_items', [])),
            'errors': chunk_stats.get('errors', 0),
            'skipped': chunk_stats.get('skipped', 0),
            'error_types': chunk_stats.get('error_types', {}),
            'gpu_memory': f"{get_gpu_memory_usage(self.device):.1%}",
            'batch_size': self.batch_size
        }
        
        # Update tracker
        update_tracker(tracker, processed=n)
        
        # Call progress callback if provided
        if progress_callback is not None:
            try:
                progress_callback(n, extra_stats)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    async def cleanup(self):
        """Clean up resources asynchronously."""
        try:
            # Clean up thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Clean up processors
            if hasattr(self.image_processor, 'cleanup'):
                await self.image_processor.cleanup()
            if hasattr(self.text_processor, 'cleanup'):
                await self.text_processor.cleanup()
            
            # Clean up cache manager
            if hasattr(self.cache_manager, 'cleanup'):
                await self.cache_manager.cleanup()
            
            # Clear CUDA cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up batch processor resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Don't re-raise as this is cleanup code

    async def process_batch(
        self,
        batch_items: List[Dict],
        width: int,
        height: int
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process a batch of items with progress tracking."""
        try:
            # Create progress tracker for this batch
            tracker = create_progress_tracker(
                total_items=len(batch_items),
                batch_size=self.batch_size,
                device=self.device
            )
            
            # Convert images to tensors
            images = []
            for item in batch_items:
                try:
                    img = await self._load_and_validate_image(item['image_path'])
                    if img is not None:
                        images.append(img)
                        update_tracker(tracker, processed=1)
                    else:
                        update_tracker(tracker, failed=1, error_type='invalid_image')
                except Exception as e:
                    logger.error(f"Error loading image {item['image_path']}: {e}")
                    update_tracker(tracker, failed=1, error_type=type(e).__name__)

            if not images:
                return [], {'processed': 0, 'errors': len(batch_items)}

            # Process images in batch
            try:
                processed_images = self.image_processor.process_batch(images, width, height)
                
                # Create processed items
                processed_items = []
                for i, (item, img_tensor) in enumerate(zip(batch_items, processed_images)):
                    if torch.any(img_tensor):  # Check if tensor contains any non-zero values
                        processed_items.append({
                            **item,
                            'processed_image': img_tensor
                        })
                    else:
                        update_tracker(tracker, failed=1, error_type='processing_failed')

                # Log final batch stats
                if tracker.should_log():
                    extra_stats = {
                        'width': width,
                        'height': height,
                        'success_rate': f"{len(processed_items)/len(batch_items)*100:.1f}%"
                    }
                    log_progress(tracker, prefix="Batch: ", extra_stats=extra_stats)

                return processed_items, tracker.get_stats()

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                update_tracker(tracker, failed=len(images), error_type='batch_processing_error')
                return [], tracker.get_stats()

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [], {'processed': 0, 'errors': len(batch_items)}