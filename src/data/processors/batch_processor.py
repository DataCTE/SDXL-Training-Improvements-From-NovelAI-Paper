# src/data/processors/batch_processor.py
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from PIL import Image
import asyncio

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
    get_gpu_memory_usage
)
from src.data.processors.utils.progress_utils import (
    create_progress_stats,
    update_progress_stats,
    log_progress
)
from src.data.processors.utils.image_utils import load_and_validate_image

# Internal imports from processors
from src.data.processors.cache_manager import CacheManager
from src.data.processors.text_embedder import TextEmbedder
from src.data.processors.image_processor import ImageProcessor
from src.data.processors.tag_weighter import parse_tags

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process batches of images and text with GPU optimization."""
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        cache_manager: CacheManager,
        text_embedder: TextEmbedder,
        vae,
        device: torch.device,
        batch_size: int = 32,
        prefetch_factor: int = 2,
        max_memory_usage: float = 0.9,
        num_workers: Optional[int] = None
    ):
        self.image_processor = image_processor
        self.cache_manager = cache_manager
        self.text_embedder = text_embedder
        self.vae = vae
        self.device = device
        
        # Calculate optimal batch size based on memory
        self.batch_size = calculate_optimal_batch_size(
            device=device,
            min_batch_size=1,
            max_batch_size=batch_size,
            target_memory_usage=max_memory_usage
        )
        
        # Initialize batch processor with optimal configuration
        self.batch_processor = GenericBatchProcessor(
            config=BatchConfig(
                batch_size=self.batch_size,
                device=device,
                max_memory_usage=max_memory_usage,
                prefetch_factor=prefetch_factor
            ),
            executor=create_thread_pool(num_workers or get_optimal_workers()),
            name="BatchProcessor"
        )

    async def _load_images(self, batch_items: List[Dict], width: int, height: int) -> Tuple[List[Image.Image], List[Dict]]:
        """Load and validate images asynchronously."""
        images = []
        valid_items = []
        
        # Create progress stats for image loading
        stats = create_progress_stats(len(batch_items))
        
        for item in batch_items:
            try:
                img = load_and_validate_image(
                    item['path'],
                    min_size=(width, height),
                    max_size=(width*2, height*2)  # Allow some flexibility
                )
                if img is not None:
                    images.append(img)
                    valid_items.append(item)
                update_progress_stats(stats, 1)
            except Exception as e:
                logger.error(f"Error loading image {item['path']}: {e}")
                stats.failed_items += 1
                
            # Log progress periodically
            if stats.should_log():
                log_progress(stats, prefix="Loading Images: ")
                
        return images, valid_items

    async def process_batch(
        self,
        batch_items: List[Dict],
        width: int,
        height: int,
        proportion_empty_prompts: float = 0.0
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process a batch of images and return results with stats."""
        stats = create_progress_stats(len(batch_items))
        stats.update({
            'cache_hits': 0,
            'cache_misses': 0,
            'error_types': {}
        })
        
        try:
            # Load images asynchronously
            images, valid_items = await self._load_images(batch_items, width, height)
            if not images:
                stats.skipped_items = len(batch_items)
                return [], stats.get_stats()

            # Process images through VAE
            batch_tensor = self.image_processor.process_batch(images, width, height)
            with torch.cuda.amp.autocast():
                latents = self.vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Process text in parallel
            save_tasks = []
            text_batch = []
            text_indices = []
            
            for i, item in enumerate(valid_items):
                if not item['text_cache'].exists():
                    stats.cache_misses += 1
                    with open(item['path'].with_suffix('.txt'), 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    text_batch.append(caption)
                    text_indices.append(i)
                else:
                    stats.cache_hits += 1

            # Process text and save results
            if text_batch:
                text_embeds = self.text_embedder.encode_prompt_list(
                    text_batch,
                    proportion_empty_prompts=proportion_empty_prompts
                )
                tags = [parse_tags(txt) for txt in text_batch]

                for idx, i in enumerate(text_indices):
                    save_tasks.append(
                        self.cache_manager.save_text_data_async(
                            valid_items[i]['text_cache'],
                            {
                                'embeds': text_embeds['prompt_embeds'][idx],
                                'pooled_embeds': text_embeds['pooled_prompt_embeds'][idx],
                                'tags': tags[idx]
                            }
                        )
                    )

            # Save latents
            for latent, item in zip(latents, valid_items):
                save_tasks.append(
                    self.cache_manager.save_latent_async(item['latent_cache'], latent)
                )

            # Wait for all save operations
            if save_tasks:
                await asyncio.gather(*save_tasks)
                
            # Update final stats
            stats.processed_items = len(valid_items)
            return valid_items, stats.get_stats()

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            stats.failed_items = len(batch_items)
            stats.error_types['batch_failure'] = stats.error_types.get('batch_failure', 0) + 1
            return [], stats.get_stats()