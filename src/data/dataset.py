# src/data/dataset.py
from typing import List, Optional, Dict, Any
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from PIL import Image
import time
from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter
from src.data.image_processor import ImageProcessor, ImageProcessorConfig
from src.data.cache_manager import CacheManager
from src.data.batch_processor import BatchProcessor
from src.data.bucket import BucketManager
from src.data.thread_config import get_optimal_thread_config
from src.data.utils import find_matching_files
from src.config.config import NovelAIDatasetConfig

logger = logging.getLogger(__name__)

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
        
        # Get optimal thread configuration
        thread_config = get_optimal_thread_config()
        
        # Initialize bucket manager first
        self.bucket_manager = BucketManager(
            max_image_size=config.max_image_size,
            min_image_size=config.min_image_size,
            bucket_step=config.bucket_step,
            min_bucket_resolution=config.min_bucket_resolution,
            max_aspect_ratio=config.max_aspect_ratio,
            bucket_tolerance=config.bucket_tolerance
        )

        # Initialize image processor with bucket manager
        self.image_processor = ImageProcessor(
            ImageProcessorConfig(
                dtype=self.dtype,
                device=device,
                max_image_size=config.max_image_size,
                min_image_size=config.min_image_size
            ),
            bucket_manager=self.bucket_manager
        )
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=thread_config.num_threads
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            image_processor=self.image_processor,
            cache_manager=self.cache_manager,
            text_embedder=text_embedder,
            vae=vae,
            device=device,
            max_consecutive_batch_samples=config.max_consecutive_batch_samples,
            num_workers=thread_config.num_threads
        )

        # Process data and initialize items
        self.items = []
        self._process_data(image_dirs)
        
        logger.info(
            f"Initialized dataset with {len(self)} samples\n"
            f"Bucket stats: {self.bucket_manager.get_stats()}"
        )

    def _process_data(self, image_dirs: List[str]) -> None:
        """Process data and assign to buckets efficiently."""
        start_time = time.time()
        
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
            
        if not image_files:
            raise ValueError("No valid image-text pairs found!")
            
        # Process each image and prepare items
        for img_path in image_files:
            try:
                # Load and validate image
                with Image.open(img_path) as img:
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    width, height = img.size
                    
                    # Find appropriate bucket
                    bucket = self.bucket_manager.find_bucket(width, height)
                    if bucket is None:
                        continue
                    
                    # Get cache paths
                    cache_paths = self.cache_manager.get_cache_paths(img_path)
                    
                    # Add to items
                    self.items.append({
                        'image_path': img_path,
                        'width': width,
                        'height': height,
                        'latent_cache': cache_paths['latent'],
                        'text_cache': cache_paths['text'],
                        'bucket_key': f"{bucket.width}x{bucket.height}"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Log processing stats
        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(image_files)} files in {elapsed:.2f}s:\n"
            f"- Valid items: {len(self.items)}\n"
            f"- Success rate: {len(self.items)/len(image_files)*100:.1f}%"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with cached data."""
        item = self.items[idx]
        
        try:
            # Load cached latent
            latent = self.cache_manager.load_latent(item['latent_cache'])
            
            # Load cached text data
            text_data = self.cache_manager.load_text_data(item['text_cache'])
            
            return {
                **item,
                'latent': latent,
                'text_embeds': text_data['embeds'],
                'pooled_embeds': text_data['pooled_embeds'],
                'tags': text_data['tags']
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.items)