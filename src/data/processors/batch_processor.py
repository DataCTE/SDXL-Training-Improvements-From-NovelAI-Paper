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
    """Process batches of images and text with GPU optimization."""
    
    def __init__(
        self,
        config: BatchProcessorConfig,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        cache_manager: CacheManager,
        vae
    ):
        """Initialize with consolidated config."""
        # Initialize components
        self.config = config
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.cache_manager = cache_manager
        self.vae = vae
        
        # Create thread pool for async operations
        self.num_workers = min(
            get_optimal_workers(),
            config.num_workers or cpu_count()
        )
        
        # Initialize async queues
        self.process_queue = asyncio.Queue(maxsize=self.config.batch_size * 2)
        self.result_queue = asyncio.Queue(maxsize=self.config.batch_size * 2)
        
        # Create thread pool with proper shutdown
        self.thread_pool = create_thread_pool(
            self.num_workers,
            thread_name_prefix='batch_worker'
        )
        
        # Track active tasks
        self.active_tasks = set()
        
        logger.info(
            f"Initialized BatchProcessor:\n"
            f"- Device: {config.device}\n"
            f"- Batch size: {config.batch_size}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Prefetch factor: {config.prefetch_factor}\n"
            f"- Max memory usage: {config.max_memory_usage:.1%}"
        )

    async def _load_and_validate_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and validate an image file."""
        try:
            img_path = Path(image_path)
            return await asyncio.to_thread(
                load_and_validate_image,
                img_path,
                config=self.config  # Pass the config to use its image size settings
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
            batch_size=self.config.batch_size,
            device=self.config.device
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
            chunk_size=self.config.batch_size,
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
            'gpu_memory': f"{get_gpu_memory_usage(self.config.device):.1%}",
            'batch_size': self.config.batch_size
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
        await cleanup_processor(self)

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during batch processor deletion: {e}")

    async def process_batch(
        self,
        batch_items: List[Dict],
        cache_manager: Optional[CacheManager] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process a batch of items with tag weight support."""
        try:
            results = []
            batch_stats = {
                'processed': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'error_types': {}
            }

            async def process_item(item):
                try:
                    text_result = await self.text_processor.process_text(item['text'])
                    if text_result is None:
                        return None

                    # Adjust to handle either two or three returned values safely
                    if len(text_result) == 3:
                        prompt_embeds, pooled_prompt_embeds, tag_weights = text_result
                    elif len(text_result) == 2:
                        prompt_embeds, pooled_prompt_embeds = text_result
                        tag_weights = None
                    else:
                        logger.error("Unexpected number of items in text_result.")
                        return None

                    # Process image
                    img_result = await self.image_processor.process_image(
                        item['image_path'],
                        original_size=item['original_size']
                    )
                    if img_result is None:
                        return None

                    processed_item = {
                        'image_path': item['image_path'],
                        'latents': img_result.get('latents'),
                        'prompt_embeds': prompt_embeds,
                        'pooled_prompt_embeds': pooled_prompt_embeds,
                        'original_size': item['original_size'],
                        'crop_top_left': img_result.get('crop_top_left', None)
                    }

                    # Optionally set tag_weights if present
                    if tag_weights is not None:
                        processed_item['tag_weights'] = tag_weights

                    return processed_item

                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    return None

            # Process items concurrently
            tasks = [process_item(item) for item in batch_items]
            processed_items = await asyncio.gather(*tasks)
            
            # Filter valid results
            results = [item for item in processed_items if item is not None]
            
            # Update stats
            batch_stats['processed'] = len(results)
            batch_stats['errors'] = len(batch_items) - len(results)
            
            return results, batch_stats
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [], batch_stats