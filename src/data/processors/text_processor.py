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

    async def process_text(self, text: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Process text and extract embeddings with tag weights."""
        try:
            # Parse tags
            tag_dict = parse_tags(text)
            
            # Embed text (using named args for clarity, including device)
            embeddings = self.text_embedder(
                text=[text],
                device=self.config.device,
                clean_caption=False  # or as needed
            )
            
            # Optionally add tag weights
            if self.tag_weighter is not None:
                for tag_class, tags in tag_dict.items():
                    for t in tags:
                        self.tag_weighter.update_frequencies(tag_class, t)
                weights_tensor = self.tag_weighter.get_weights_tensor(tag_dict)
                embeddings["tag_weights"] = weights_tensor
            
            # Now return both embeddings and the tag_dict
            return embeddings, tag_dict

        except Exception as e:
            logger.error(f"Error processing text: {str(e)[:200]}...")
            return None, {}

    async def process_batch(
        self,
        texts: List[str],
        cache_manager: Optional[CacheManager] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, List[str]]]]:
        """Process a batch of texts in parallel with immediate caching."""
        results = []
        
        # Create progress tracker
        tracker = create_progress_tracker(
            total_items=len(texts),
            batch_size=self.config.batch_size,
            device=self.config.device
        )
        
        # Process in smaller batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            tasks = [self.process_text(text) for text in batch]
            text_results = await asyncio.gather(*tasks)
            
            batch_results_for_cache = []
            for j, (text_data, tags) in enumerate(text_results):
                if text_data is not None:
                    # Move tensors to CPU
                    for key, value in text_data.items():
                        if isinstance(value, torch.Tensor):
                            text_data[key] = value.to("cpu")
                    
                    # Cache item if enabled
                    if self.config.use_caching and cache_manager is not None:
                        cache_item = {
                            "text_path": getattr(batch[j], "filename", f"text_{i+j}"),
                            "text_data": text_data,
                            "tags": tags
                        }
                        batch_results_for_cache.append(cache_item)
                    
                    results.append((text_data, tags))
                    update_tracker(tracker, processed=1)
                else:
                    update_tracker(tracker, failed=1, error_type="text_processing_failed")
            
            # Cache batch results
            if self.config.use_caching and cache_manager and batch_results_for_cache:
                await cache_manager.cache_batch_items(batch_results_for_cache)
            
            # Clear memory less frequently or conditionally
            if (i // self.config.batch_size) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            await asyncio.sleep(0.01)
            if tracker.should_log():
                log_progress(
                    tracker,
                    prefix="Processing texts: ",
                    extra_stats={
                        "successful": len(results),
                        "failed": tracker.failed_items,
                        "error_types": tracker.error_types,
                        "memory_usage": f"{get_gpu_memory_usage(self.config.device):.1%}"
                    }
                )
        
        return results

    async def process_text_file(
        self,
        text_path: Path
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, List[str]]]]:
        """Process text from a file without caching."""
        try:
            if not text_path.exists():
                logger.error(f"Text file not found: {text_path}")
                return None
            
            # Read file text
            text_content = await asyncio.to_thread(
                lambda: text_path.read_text(encoding="utf-8")
            )
            
            # Process text
            text_data, tags = await self.process_text(text_content)
            
            if text_data is not None:
                # Move to CPU
                for key, value in text_data.items():
                    if isinstance(value, torch.Tensor):
                        text_data[key] = value.to("cpu")

                del text_content
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