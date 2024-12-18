# src/data/processors/text_processor.py
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path
import gc

from src.data.processors.cache_manager import CacheManager
from src.data.processors.utils.caption.text_embedder import TextEmbedder
from src.data.processors.utils.caption.tag_weighter import parse_tags, TagWeighter
from src.data.processors.utils.system_utils import get_optimal_workers, create_thread_pool, get_gpu_memory_usage, cleanup_processor
from src.data.processors.utils.progress_utils import (
    create_progress_tracker,
    update_tracker,
    log_progress
)
from src.config.config import TextProcessorConfig

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process text data with tag weighting and text embeddings."""
    
    def __init__(
        self,
        config: TextProcessorConfig,
        text_embedder: TextEmbedder,
        tag_weighter: Optional[TagWeighter] = None
    ):
        """Initialize text processor with tag weighting support."""
        self.config = config
        self.text_embedder = text_embedder
        self.tag_weighter = tag_weighter
        
        # Initialize thread pool for parallel processing
        self.num_workers = min(
            self.config.num_workers,
            get_optimal_workers()
        )
        self.executor = create_thread_pool(self.num_workers)
        
        logger.info(
            f"Initialized TextProcessor:\n"
            f"- Device: {config.device}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Batch size: {config.batch_size}\n"
            f"- Tag Weighter: {'Yes' if tag_weighter else 'No'}\n"
            f"- Max token length: {config.max_token_length}"
        )

    async def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and extract embeddings with tag weights."""
        try:
            # Parse tags and get embeddings
            tag_dict = parse_tags(text)
            embeddings = self.text_embedder([text], self.config.proportion_empty_prompts)
            
            # Add tag weights if weighter is available
            if self.tag_weighter is not None:
                # Update frequencies
                for tag_class, tags in tag_dict.items():
                    for tag in tags:
                        self.tag_weighter.update_frequencies(tag_class, tag)
                
                # Get weights tensor
                weights_tensor = self.tag_weighter.get_weights_tensor(tag_dict)
                embeddings['tag_weights'] = weights_tensor
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)[:200]}...")
            return None

    async def process_batch(
        self,
        texts: List[str],
        cache_manager: Optional[CacheManager] = None
    ) -> List[Tuple[Dict[str, Any], List[str]]]:
        """Process a batch of texts in parallel with immediate caching."""
        results = []
        
        # Create progress tracker
        tracker = create_progress_tracker(
            total_items=len(texts),
            batch_size=self.config.batch_size,
            device=self.config.device
        )
        
        # Process in smaller batches to manage memory
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = []
            
            # Process batch items in parallel
            tasks = [self.process_text(text) for text in batch]
            text_results = await asyncio.gather(*tasks)
            
            # Process results and prepare for caching
            for j, (text_data, tags) in enumerate(text_results):
                if text_data is not None:
                    # Move tensors to CPU immediately
                    for key, value in text_data.items():
                        if isinstance(value, torch.Tensor):
                            text_data[key] = value.cpu()
                            del value
                    
                    # Create cache item if caching is enabled
                    if self.config.use_caching and cache_manager is not None:
                        cache_item = {
                            'text_path': getattr(batch[j], 'filename', f'text_{i+j}'),
                            'text_data': text_data,
                            'tags': tags
                        }
                        batch_results.append(cache_item)
                    
                    results.append((text_data, tags))
                    update_tracker(tracker, processed=1)
                else:
                    update_tracker(tracker, failed=1, error_type='text_processing_failed')
            
            # Cache batch results if enabled
            if self.config.use_caching and cache_manager is not None and batch_results:
                await cache_manager.cache_batch_items(batch_results)
                del batch_results[:]
            
            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(0.01)
            
            # Log progress
            if tracker.should_log():
                extra_stats = {
                    'successful': len(results),
                    'failed': tracker.failed_items,
                    'error_types': tracker.error_types,
                    'memory_usage': f"{get_gpu_memory_usage(self.config.device):.1%}"
                }
                log_progress(tracker, prefix="Processing texts: ", extra_stats=extra_stats)
        
        return results

    async def process_text_file(
        self,
        text_path: Path
    ) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        """Process text from a file without caching."""
        try:
            if not text_path.exists():
                logger.error(f"Text file not found: {text_path}")
                return None
                
            # Read text file
            text = await asyncio.to_thread(
                lambda: text_path.read_text(encoding='utf-8')
            )
            
            # Process text and tags
            text_data, tags = await self.process_text(text)
            
            if text_data is not None:
                # Move tensors to CPU and clear GPU memory
                for key, value in text_data.items():
                    if isinstance(value, torch.Tensor):
                        text_data[key] = value.cpu()
                        del value
                
                # Clear text data after processing
                del text
                gc.collect()
                torch.cuda.empty_cache()
            
            return text_data, tags
            
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {str(e)[:200]}...")
            return None

    async def cleanup(self):
        await cleanup_processor(self)

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during text processor deletion: {e}")