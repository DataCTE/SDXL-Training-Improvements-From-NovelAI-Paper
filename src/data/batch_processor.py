# src/data/batch_processor.py
import torch
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from PIL import Image
import asyncio
from src.data.cache_manager import CacheManager
from src.data.text_embedder import TextEmbedder
from src.data.image_processor import ImageProcessor
from src.data.tag_weighter import parse_tags
from src.data.utils import (
    BatchConfig,
    BatchProcessor as GenericBatchProcessor,
    create_thread_pool,
    get_optimal_workers
)

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
        
        # Initialize batch processor
        self.batch_processor = GenericBatchProcessor(
            config=BatchConfig(
                batch_size=batch_size,
                device=device,
                max_memory_usage=max_memory_usage,
                prefetch_factor=prefetch_factor
            ),
            executor=create_thread_pool(num_workers or get_optimal_workers()),
            name="BatchProcessor"
        )
        
        logger.info(
            f"Initialized BatchProcessor:\n"
            f"- Device: {device}\n"
            f"- Batch size: {batch_size}\n"
            f"- Workers: {self.batch_processor.executor._max_workers}\n"
            f"- Prefetch factor: {prefetch_factor}\n"
            f"- Max memory usage: {max_memory_usage*100:.0f}%"
        )

    async def _load_images(self, batch_items: List[Dict], width: int, height: int) -> List[Image.Image]:
        """Load and validate images asynchronously."""
        images = []
        valid_items = []
        
        for item in batch_items:
            try:
                with Image.open(item['path']) as img:
                    img = img.convert('RGB')
                    if img.size != (width, height):
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    images.append(img.copy())
                    valid_items.append(item)
            except Exception as e:
                logger.error(f"Error loading image {item['path']}: {e}")
                
        return images, valid_items

    async def _process_vae(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Process batch through VAE with memory optimization."""
        try:
            with torch.cuda.amp.autocast():
                latents = self.vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            return latents
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce batch size and retry
                self.batch_processor.config.batch_size = max(1, self.batch_processor.config.batch_size - 1)
                logger.warning(f"Out of memory, reducing batch size to {self.batch_processor.config.batch_size}")
                raise
            raise

    async def process_batch(
        self,
        batch_items: List[Dict],
        width: int,
        height: int,
        proportion_empty_prompts: float = 0.0
    ) -> int:
        """Process a batch of images efficiently with async I/O and GPU optimization."""
        async def process_fn(items: List[Dict]) -> List[Dict]:
            # Load images asynchronously
            images, valid_items = await self._load_images(items, width, height)
            if not images:
                return []

            # Process images through VAE
            batch_tensor = self.image_processor.process_batch(images, width, height)
            latents = await self._process_vae(batch_tensor)

            # Process in parallel
            save_tasks = []
            text_batch = []
            text_indices = []
            
            # Prepare text processing
            for i, (img_path, text_path) in enumerate(zip([item['path'] for item in valid_items], 
                                                      [item['text_cache'] for item in valid_items])):
                if not text_path.exists():
                    with open(Path(img_path).with_suffix('.txt'), 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    text_batch.append(caption)
                    text_indices.append(i)

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

            return valid_items

        def cleanup_fn():
            # Clean up CUDA memory
            torch.cuda.empty_cache()

        # Process batch using generic batch processor
        processed_items = await self.batch_processor.process_batches(
            items=batch_items,
            process_fn=process_fn,
            cleanup_fn=cleanup_fn
        )

        return len(processed_items)