# src/data/dataset.py
from typing import List, Optional, Tuple, Dict, Union
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from dataclasses import dataclass
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import psutil
from collections import defaultdict

from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter
from src.data.image_processor import ImageProcessor, ImageProcessorConfig
from src.data.cache_manager import CacheManager
from src.data.batch_processor import BatchProcessor
from src.data.sampler import AspectBatchSampler
from src.data.bucket import BucketManager
from src.data.utils import (
    find_matching_files,
    get_optimal_workers,
    process_in_chunks,
    calculate_chunk_size,
    log_system_info,
    get_memory_usage_gb
)

logger = logging.getLogger(__name__)

@dataclass
class NovelAIDatasetConfig:
    """Configuration for NovelAI dataset."""
    image_size: Union[Tuple[int, int], int] = (1024, 1024)
    min_size: Union[Tuple[int, int], int] = (256, 256)
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
    
    def __post_init__(self):
        """Convert single integers to tuples for sizes."""
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        if isinstance(self.min_size, int):
            self.min_size = (self.min_size, self.min_size)

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
        self.num_workers = get_optimal_workers()
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=self.num_workers
        )
        
        # Initialize bucket manager
        self.bucket_manager = BucketManager(
            max_image_size=config.image_size,
            min_image_size=config.min_size,
            bucket_step=config.bucket_step,
            min_bucket_resolution=config.min_size[0] * config.min_size[1],
            max_aspect_ratio=config.max_aspect_ratio,
            bucket_tolerance=config.bucket_tolerance
        )
        
        self.batch_processor = BatchProcessor(
            image_processor=self.image_processor,
            cache_manager=self.cache_manager,
            text_embedder=text_embedder,
            vae=vae,
            device=device,
            batch_size=config.batch_size
        )

        # Process data with parallel execution
        self.items = []
        self._parallel_process_data(image_dirs)
        
        logger.info(
            f"Initialized dataset with {len(self)} samples using {self.num_workers} CPU threads\n"
            f"Bucket stats: {self.bucket_manager.get_stats()}"
        )

    def get_sampler(self, batch_size: Optional[int] = None, shuffle: bool = True, drop_last: bool = False) -> AspectBatchSampler:
        """Create an aspect-aware batch sampler for the dataset."""
        return AspectBatchSampler(
            dataset=self,
            batch_size=batch_size or self.config.batch_size,
            max_image_size=self.config.image_size,
            min_image_size=self.config.min_size,
            max_dim=self.config.max_dim,
            bucket_step=self.config.bucket_step,
            min_bucket_resolution=self.config.min_size[0] * self.config.min_size[1],
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_tolerance=self.config.bucket_tolerance,
            max_aspect_ratio=self.config.max_aspect_ratio,
            min_bucket_length=self.config.min_bucket_size,
            max_consecutive_batch_samples=self.config.max_consecutive_batch_samples
        )

    def _parallel_process_data(self, image_dirs: List[str]):
        """Process data using parallel execution."""
        # Find all valid image files
        image_files = []
        for image_dir in image_dirs:
            files = find_matching_files(
                image_dir,
                extensions={'.jpg', '.jpeg', '.png', '.webp'},
                recursive=True,
                require_text_pair=True
            )
            image_files.extend(files)
            
        total_files = len(image_files)
        if total_files == 0:
            logger.warning("No valid image-text pairs found!")
            return

        # Log system info and calculate optimal chunk size
        log_system_info()
        optimal_workers = get_optimal_workers()
        chunk_size = calculate_chunk_size(total_files, optimal_workers)
        
        logger.info(
            f"Processing configuration:\n"
            f"- Workers: {optimal_workers}\n"
            f"- Chunk size: {chunk_size} images\n"
            f"- Total files: {total_files}"
        )
        
        # Process files in chunks
        self.items, stats = process_in_chunks(
            items=image_files,
            chunk_size=chunk_size,
            process_fn=self._process_chunk,
            num_workers=optimal_workers
        )
        
        logger.info(
            f"Dataset preparation completed:\n"
            f"- Valid items: {len(self.items)}\n"
            f"- Total files: {total_files}\n"
            f"- Success rate: {len(self.items)/total_files*100:.1f}%\n"
            f"Error types: {stats.get('error_types', {})}"
        )

    def _process_chunk(self, image_files: List[str], chunk_id: int) -> Tuple[List[Dict], Dict[str, int]]:
        """Process a chunk of image files."""
        chunk_items = []
        stats = defaultdict(int)
        error_types = defaultdict(int)
        
        for img_path in image_files:
            stats['total'] += 1
            
            try:
                # Validate text file
                txt_path = Path(img_path).with_suffix('.txt')
                if not txt_path.exists() or txt_path.stat().st_size == 0:
                    stats['skipped'] += 1
                    error_types['missing_or_empty_text'] += 1
                    continue
                
                # Validate and load image
                try:
                    with Image.open(img_path) as img:
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGB')
                        width, height = img.size
                        
                        # Basic image validation
                        if width < 32 or height < 32:
                            stats['skipped'] += 1
                            error_types['image_too_small'] += 1
                            continue
                            
                        if width > 8192 or height > 8192:
                            stats['skipped'] += 1
                            error_types['image_too_large'] += 1
                            continue
                            
                        # Find appropriate bucket
                        bucket = self.bucket_manager.find_bucket(width, height)
                        if bucket is None:
                            stats['skipped'] += 1
                            error_types['no_suitable_bucket'] += 1
                            continue
                            
                        # Get cache paths
                        cache_paths = self.cache_manager.get_cache_paths(img_path)
                        
                        # Add to valid items
                        chunk_items.append({
                            'image_path': img_path,
                            'bucket': bucket,
                            'latent_cache': cache_paths['latent'],
                            'text_cache': cache_paths['text'],
                            'original_size': (height, width),
                            'crop_top_left': (0, 0)
                        })
                        stats['valid'] += 1
                        
                except (IOError, OSError) as e:
                    stats['errors'] += 1
                    error_types['image_load_error'] += 1
                    logger.debug(f"Error loading image {img_path}: {e}")
                    continue
                    
            except Exception as e:
                stats['errors'] += 1
                error_types['other_error'] += 1
                logger.error(f"Unexpected error processing {img_path}: {e}")
                continue
        
        stats['error_types'] = dict(error_types)
        return chunk_items, stats

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