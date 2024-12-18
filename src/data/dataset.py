# src/data/dataset.py
from typing import List, Optional, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import logging
import asyncio
import gc
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path

# Internal imports from processors
from .processors.text_processor import TextProcessor
from .processors.batch_processor import BatchProcessor
from .processors.image_processor import ImageProcessor
from .processors.cache_manager import CacheManager
from .processors.bucket import BucketManager
from .processors.sampler import AspectBatchSampler

# Import utilities
from .processors.utils.caption.text_embedder import TextEmbedder
from .processors.utils.caption.tag_weighter import TagWeighter
from .processors.utils.thread_config import get_optimal_thread_config
from .processors.utils import (
    find_matching_files,
    get_optimal_workers,
    calculate_optimal_batch_size,
    get_gpu_memory_usage,
    validate_image_text_pair,
    format_time
)
from .processors.utils.progress_utils import (
    create_progress_tracker,
)
from src.utils.logging.metrics import log_metrics, log_system_info, log_error_with_context

# Config import
from src.config.config import NovelAIDatasetConfig, ImageProcessorConfig, TextProcessorConfig, TextEmbedderConfig, BucketConfig, BatchProcessorConfig

logger = logging.getLogger(__name__)

class NovelAIDataset(Dataset):
    def __init__(
        self,
        image_dirs: List[str],
        config: NovelAIDatasetConfig,
        vae,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize basic attributes only."""
        super().__init__()
        self.config = config
        self.device = device
        self.vae = vae.eval()
        self.image_dirs = image_dirs
        self.items = []
        
        # Initialize empty attributes that will be set in _initialize
        self.text_embedder = None
        self.tag_weighter = None
        self.text_processor = None
        self.bucket_manager = None
        self.image_processor = None
        self.cache_manager = None
        self.batch_processor = None
        self.sampler = None

    @classmethod
    async def create(
        cls,
        image_dirs: List[str],
        config: NovelAIDatasetConfig,
        vae,
        device: torch.device = torch.device('cuda')
    ):
        """Async factory method for initialization."""
        self = cls(image_dirs, config, vae, device)
        await self._initialize()
        return self

    async def _initialize(self):
        """Async initialization of components."""
        try:
            # Get optimal configuration
            thread_config = get_optimal_thread_config()
            optimal_batch_size = calculate_optimal_batch_size(
                device=self.device,
                min_batch_size=self.config.min_bucket_size,
                max_batch_size=32,
                target_memory_usage=0.8
            )
            num_workers = get_optimal_workers(memory_per_worker_gb=1.0)

            # Initialize components
            self.text_embedder = TextEmbedder(
                config=TextEmbedderConfig(
                    model_name=self.config.model_name,
                    device=self.device,
                    max_length=self.config.max_token_length,
                    enable_memory_efficient_attention=True,
                    max_memory_usage=0.8
                )
            )

            self.tag_weighter = TagWeighter(config=self.config.tag_weighting)
            
            self.text_processor = TextProcessor(
                config=TextProcessorConfig(
                    device=self.device,
                    batch_size=optimal_batch_size,
                    max_token_length=self.config.max_token_length,
                    use_caching=self.config.use_caching
                ),
                text_embedder=self.text_embedder,
                tag_weighter=self.tag_weighter
            )

            self.bucket_manager = BucketManager(
                config=BucketConfig(
                    max_image_size=self.config.max_image_size,
                    min_image_size=self.config.min_image_size,
                    bucket_step=self.config.bucket_step,
                    min_bucket_resolution=self.config.min_bucket_resolution,
                    max_aspect_ratio=self.config.max_aspect_ratio,
                    bucket_tolerance=self.config.bucket_tolerance
                )
            )

            self.image_processor = ImageProcessor(
                config=ImageProcessorConfig(
                    dtype=next(self.vae.parameters()).dtype,
                    device=self.device,
                    max_image_size=self.config.max_image_size,
                    min_image_size=self.config.min_image_size,
                    enable_vae_slicing=True,
                    vae_batch_size=optimal_batch_size,
                    num_workers=num_workers,
                    prefetch_factor=thread_config.prefetch_factor,
                    enable_memory_efficient_attention=True,
                    max_memory_usage=0.8
                ),
                bucket_manager=self.bucket_manager,
                vae=self.vae
            )

            self.cache_manager = CacheManager(
                cache_dir=self.config.cache_dir,
                max_workers=num_workers,
                use_caching=self.config.use_caching
            )

            self.batch_processor = BatchProcessor(
                config=BatchProcessorConfig(
                    device=self.device,
                    batch_size=optimal_batch_size,
                    prefetch_factor=thread_config.prefetch_factor,
                    num_workers=num_workers,
                    max_memory_usage=0.8,
                    min_batch_size=1,
                    max_batch_size=optimal_batch_size
                ),
                image_processor=self.image_processor,
                text_processor=self.text_processor,
                cache_manager=self.cache_manager,
                vae=self.vae
            )

            # Process data
            await self._process_data(self.image_dirs)

            # Create sampler
            self.sampler = AspectBatchSampler(
                dataset=self,
                batch_size=optimal_batch_size,
                shuffle=True,
                drop_last=False,
                max_consecutive_batch_samples=2,
                min_bucket_length=self.config.min_bucket_size,
                debug_mode=self.config.debug_mode,
                prefetch_factor=thread_config.prefetch_factor
            )

            logger.info(f"Dataset initialized with {len(self)} samples")

        except Exception as e:
            logger.error(f"Dataset initialization failed: {e}")
            await self.cleanup()
            raise

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with cached data."""
        item = self.items[idx]
        
        try:
            # Load cached latent and ensure it's on CPU
            latent = self.cache_manager.load_latent(item['latent_cache'])
            if latent.device.type != 'cpu':
                latent = latent.cpu()
            
            # Load cached text data and ensure tensors are on CPU
            text_data = self.cache_manager.load_text_data(item['text_cache'])
            text_data = self.cache_manager._ensure_cpu_tensors(text_data)
            
            result = {
                **item,
                'latent': latent,
                'text_embeds': text_data['embeds'],
                'pooled_embeds': text_data['pooled_embeds'],
                'tags': text_data['tags'],
                'tag_weights': text_data.get('tag_weights', None)
            }
            
            # Clear references to original data
            del latent
            del text_data
            
            # Clear cache periodically
            if idx % 100 == 0:
                self.cache_manager.clear_memory_cache()
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

    async def _process_data(self, image_dirs: List[str]):
        """Process all data with improved embedding handling."""
        try:
            # Find all image files
            image_files = await find_matching_files(image_dirs)
            
            # Process images and text in batches
            for batch in chunks(image_files, self.config.batch_size):
                # Get text embeddings
                text_files = [Path(f).with_suffix('.txt') for f in batch]
                texts = [await asyncio.to_thread(f.read_text) for f in text_files if f.exists()]
                
                if texts:
                    # Get embeddings for batch
                    embeddings = await self.text_processor.process_batch(texts)
                    
                    # Process images
                    image_results = await self.image_processor.process_batch(batch)
                    
                    # Combine results
                    for img_result, embedding in zip(image_results, embeddings):
                        if img_result is not None and embedding is not None:
                            self.items.append({
                                'image_path': img_result['image_path'],
                                'latents': img_result['latents'],
                                'prompt_embeds': embedding['prompt_embeds'],
                                'pooled_prompt_embeds': embedding['pooled_prompt_embeds'],
                                'tag_weights': embedding.get('tag_weights')
                            })
                
                # Clear memory periodically
                if len(self.items) % 1000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    async def cleanup(self):
        """Clean up all resources."""
        try:
            # Clean up all processors
            await self.batch_processor.cleanup()
            await self.text_processor.cleanup()
            await self.image_processor.cleanup()
            
            # Clean up cache manager
            if hasattr(self.cache_manager, 'cleanup'):
                await self.cache_manager.cleanup()
            
            # Clear references to items
            if hasattr(self, 'items'):
                self.items.clear()
            
            # Clear sampler
            if hasattr(self, 'sampler'):
                del self.sampler
            
            # Clear CUDA cache and force garbage collection
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
                
            logger.info("Successfully cleaned up all dataset resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when object is deleted."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during dataset deletion: {e}")