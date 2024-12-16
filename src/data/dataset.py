# src/data/dataset.py
from typing import List, Optional, Tuple, Callable, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from dataclasses import dataclass
import os
import glob
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter
from src.data.bucket import AspectRatioBucket
from src.data.image_processor import ImageProcessor, ImageProcessorConfig
from src.data.cache_manager import CacheManager
from src.data.batch_processor import BatchProcessor
from src.data.sampler import AspectBatchSampler

logger = logging.getLogger(__name__)

@dataclass
class NovelAIDatasetConfig:
    """Configuration for NovelAI dataset."""
    image_size: Tuple[int, int] = (1024, 1024)
    min_size: Tuple[int, int] = (256, 256)
    max_dim: int = 1024
    bucket_step: int = 64
    min_bucket_size: int = 1
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 3.0
    cache_dir: str = "cache"
    use_caching: bool = True
    proportion_empty_prompts: float = 0.0
    batch_size: int = 32
    max_consecutive_batch_samples: int = 2

class NovelAIDataset(Dataset):
    def __init__(
        self,
        image_dirs: List[str],
        text_embedder: TextEmbedder,
        tag_weighter: TagWeighter,
        vae,  # AutoencoderKL
        config: NovelAIDatasetConfig,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize NovelAI dataset with optimized components."""
        self.config = config
        self.device = device
        self.text_embedder = text_embedder
        self.tag_weighter = tag_weighter
        self.vae = vae
        
        # Get model dtype from VAE
        self.dtype = next(vae.parameters()).dtype

        # Initialize optimized components
        self.image_processor = ImageProcessor(
            ImageProcessorConfig(
                dtype=self.dtype,
                device=device
            )
        )
        
        # Initialize parallel processing with optimal workers
        self.num_workers = min(32, multiprocessing.cpu_count() * 2)
        
        # Initialize cache manager with same worker count for consistency
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=self.num_workers
        )
        
        self.batch_processor = BatchProcessor(
            image_processor=self.image_processor,
            cache_manager=self.cache_manager,
            text_embedder=text_embedder,
            vae=vae,
            device=device,
            batch_size=config.batch_size
        )

        # Initialize bucketing
        self.bucket_manager = AspectRatioBucket(
            max_image_size=config.image_size,
            min_image_size=config.min_size,
            max_dim=config.max_dim,
            bucket_step=config.bucket_step
        )

        # Initialize thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Process data with parallel execution
        self.items = []
        self._parallel_process_data(image_dirs)
        
        logger.info(f"Initialized dataset with {len(self)} samples in {len(self.bucket_manager.buckets)} buckets")

    def get_sampler(self, batch_size: Optional[int] = None, shuffle: bool = True, drop_last: bool = False) -> AspectBatchSampler:
        """Create an aspect-aware batch sampler for the dataset."""
        return AspectBatchSampler(
            dataset=self,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_tolerance=self.config.bucket_tolerance,
            max_aspect_ratio=self.config.max_aspect_ratio,
            min_bucket_length=self.config.min_bucket_size,
            max_consecutive_batch_samples=self.config.max_consecutive_batch_samples
        )

    def _parallel_process_data(self, image_dirs: List[str]):
        """Process data using parallel execution."""
        image_files = []
        for image_dir in image_dirs:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

        # Process files in parallel chunks
        chunk_size = max(1, len(image_files) // (self.num_workers * 4))
        chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]
        
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, chunk)
            futures.append(future)
            
        # Collect results
        for future in futures:
            chunk_items = future.result()
            self.items.extend(chunk_items)

    def _process_chunk(self, image_files: List[str]) -> List[Dict]:
        """Process a chunk of image files."""
        chunk_items = []
        for img_path in image_files:
            try:
                if not Path(img_path).with_suffix('.txt').exists():
                    continue
                    
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    width, height = img.size
                    bucket = self.bucket_manager.find_bucket(width, height)
                    if bucket is None:
                        continue
                        
                    cache_paths = self.cache_manager.get_cache_paths(img_path)
                    
                    chunk_items.append({
                        'image_path': img_path,
                        'bucket': bucket,
                        'latent_cache': cache_paths['latent'],
                        'text_cache': cache_paths['text'],
                        'original_size': (height, width),
                        'crop_top_left': (0, 0)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
                
        return chunk_items

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item with cached data."""
        item = self.items[idx]
        
        # Load cached data efficiently
        latent = self.cache_manager.load_latent(item['latent_cache'])
        if len(latent.shape) == 4:
            latent = latent.squeeze(0)
        
        text_data = self.cache_manager.load_text_data(item['text_cache'])
        tag_weight = torch.tensor(self.tag_weighter.get_weight(text_data['tags']))
        
        return {
            'model_input': latent,
            'text_embeds': text_data['embeds'],
            'tag_weights': tag_weight,
            'original_sizes': item['original_size'],
            'crop_top_lefts': item['crop_top_left'],
            'target_sizes': (item['bucket'].height, item['bucket'].width)
        }

    def __len__(self) -> int:
        return len(self.items)