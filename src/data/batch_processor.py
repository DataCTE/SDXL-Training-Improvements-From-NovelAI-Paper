# src/data/batch_processor.py
import torch
from typing import List, Dict, Any
import logging
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import asyncio
from src.data.cache_manager import CacheManager
from src.data.text_embedder import TextEmbedder
from src.data.image_processor import ImageProcessor
from src.data.thread_config import get_optimal_cpu_threads

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(
        self,
        image_processor: ImageProcessor,
        cache_manager: CacheManager,
        text_embedder: TextEmbedder,
        vae,
        device: torch.device,
        batch_size: int = 32
    ):
        self.image_processor = image_processor
        self.cache_manager = cache_manager
        self.text_embedder = text_embedder
        self.vae = vae
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = get_optimal_cpu_threads.chunk_size

    @torch.no_grad()
    async def process_batch(
        self,
        batch_items: List[Dict],
        width: int,
        height: int,
        proportion_empty_prompts: float = 0.0
    ) -> int:
        """Process a batch of images efficiently with async I/O."""
        images = []
        paths = []
        latent_paths = []
        text_paths = []

        # Load images
        for item in batch_items:
            try:
                with Image.open(item['path']) as img:
                    img = img.convert('RGB')
                    images.append(img)
                    paths.append(item['path'])
                    latent_paths.append(item['latent_cache'])
                    text_paths.append(item['text_cache'])
            except Exception as e:
                logger.error(f"Error loading {item['path']}: {e}")
                continue

        if not images:
            return 0

        try:
            # Process images through VAE
            batch_tensor = self.image_processor.process_batch(images, width, height)
            
            with torch.cuda.amp.autocast():
                latents = self.vae.encode(batch_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Save latents asynchronously
            save_tasks = []
            for latent, path in zip(latents, latent_paths):
                save_tasks.append(
                    self.cache_manager.save_latent_async(path, latent)
                )

            # Process text embeddings
            text_batch = []
            text_indices = []
            for i, (img_path, text_path) in enumerate(zip(paths, text_paths)):
                if not text_path.exists():
                    with open(Path(img_path).with_suffix('.txt'), 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    text_batch.append(caption)
                    text_indices.append(i)

            if text_batch:
                text_embeds = self.text_embedder.encode_prompt_list(
                    text_batch,
                    proportion_empty_prompts=proportion_empty_prompts
                )
                tags = [parse_tags(txt) for txt in text_batch]

                for idx, i in enumerate(text_indices):
                    save_tasks.append(
                        self.cache_manager.save_text_data_async(
                            text_paths[i],
                            {
                                'embeds': text_embeds['prompt_embeds'][idx],
                                'tags': tags[idx],
                            }
                        )
                    )

            # Wait for all save operations to complete
            await asyncio.gather(*save_tasks)

            return len(images)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return 0

        finally:
            # Clean up
            del batch_tensor, latents
            torch.cuda.empty_cache()