# src/data/dataset.py
from typing import List, Optional, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import logging
import asyncio

# Internal imports from processors
from .processors.text_processor import TextProcessor
from .processors.batch_processor import BatchProcessor
from .processors.image_processor import ImageProcessor, ImageProcessorConfig
from .processors.cache_manager import CacheManager
from .processors.bucket import BucketManager
from .processors.sampler import AspectBatchSampler

# Import utilities
from .processors.utils.caption.text_embedder import TextEmbedder
from .processors.utils.caption.tag_weighter import TagWeighter
from .processors.utils.thread_config import get_optimal_thread_config
from .processors.utils import (
    find_matching_files,
    ensure_dir,
    get_file_size,
    get_optimal_workers,
    get_system_resources,
    log_system_info,
    calculate_optimal_batch_size,
    create_progress_stats,
    update_progress_stats,
    format_time,
    get_gpu_memory_usage,
    log_progress
)

# Config import
from src.config.config import NovelAIDatasetConfig, TagWeighterConfig

logger = logging.getLogger(__name__)

class NovelAIDataset(Dataset):
    def __init__(
        self,
        image_dirs: List[str],
        config: NovelAIDatasetConfig,
        vae,  # AutoencoderKL
        device: torch.device = torch.device('cuda')
    ):
        """Initialize NovelAI dataset with optimized components."""
        super().__init__()
        self.config = config
        self.device = device
        self.vae = vae
        
        # Get optimal configuration
        thread_config = get_optimal_thread_config()
        resources = get_system_resources()
        optimal_batch_size = calculate_optimal_batch_size(
            device=device,
            min_batch_size=config.min_bucket_size,
            max_batch_size=64,
            target_memory_usage=0.9
        )
        num_workers = get_optimal_workers(memory_per_worker_gb=1.0)
        
        # Initialize text embedder
        self.text_embedder = TextEmbedder(
            pretrained_model_name_or_path=config.model_name,
            device=device,
            max_length=config.max_token_length,
            enable_memory_efficient_attention=True,
            max_memory_usage=0.9
        )

        # Initialize tag weighter with config
        self.tag_weighter = TagWeighter(config=config.tag_weighting)
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            text_embedder=self.text_embedder,
            tag_weighter=self.tag_weighter,
            num_workers=num_workers,
            device=device
        )
        
        # Initialize bucket manager
        self.bucket_manager = BucketManager(
            max_image_size=config.max_image_size,
            min_image_size=config.min_image_size,
            bucket_step=config.bucket_step,
            min_bucket_resolution=config.min_bucket_resolution,
            max_aspect_ratio=config.max_aspect_ratio,
            bucket_tolerance=config.bucket_tolerance
        )

        # Initialize image processor
        self.image_processor = ImageProcessor(
            config=ImageProcessorConfig(
                dtype=next(vae.parameters()).dtype,
                device=device,
                max_image_size=config.max_image_size,
                min_image_size=config.min_image_size,
                enable_memory_efficient_attention=True,
                enable_vae_slicing=True,
                vae_batch_size=optimal_batch_size,
                num_workers=num_workers,
                prefetch_factor=thread_config.prefetch_factor,
                max_memory_usage=0.9
            ),
            bucket_manager=self.bucket_manager,
            vae=self.vae
        )
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=num_workers,
            use_caching=config.use_caching
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            image_processor=self.image_processor,
            text_processor=self.text_processor,
            cache_manager=self.cache_manager,
            vae=self.vae,
            device=device,
            batch_size=optimal_batch_size,
            prefetch_factor=thread_config.prefetch_factor,
            max_memory_usage=0.9,
            num_workers=num_workers
        )

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Create cache directory
        ensure_dir(config.cache_dir)
        
        # Process data and initialize items
        self.items = []
        try:
            loop.run_until_complete(self._process_data(image_dirs))
        finally:
            # Cleanup resources
            if hasattr(self.batch_processor, 'cleanup'):
                loop.run_until_complete(self.batch_processor.cleanup())
        
        # Create sampler for efficient batch processing
        self.sampler = AspectBatchSampler(
            dataset=self,
            batch_size=optimal_batch_size,
            shuffle=True,
            drop_last=False,
            max_consecutive_batch_samples=2,
            min_bucket_length=config.min_bucket_size,
            debug_mode=config.debug_mode,
            prefetch_factor=thread_config.prefetch_factor
        )
        
        # Log initialization info
        logger.info(
            f"Initialized dataset with {len(self)} samples\n"
            f"System Resources:\n"
            f"- Available Memory: {resources.available_memory_gb:.1f}GB\n"
            f"- GPU Memory: {resources.gpu_memory_total:.1f}GB total, "
            f"{resources.gpu_memory_used:.1f}GB used\n"
            f"- Workers: {num_workers}\n"
            f"- Optimal Batch Size: {optimal_batch_size}\n"
            f"\nComponents:\n"
            f"- Text Processor: Initialized with {self.text_processor.num_workers} workers\n"
            f"- Image Processor: Using {self.image_processor.num_workers} workers\n"
            f"- Cache Manager: {'Enabled' if config.use_caching else 'Disabled'}\n"
            f"- Bucket Manager: {len(self.bucket_manager.buckets)} buckets\n"
            f"\nConfig:\n"
            f"- Max image size: {config.max_image_size}\n"
            f"- Min image size: {config.min_image_size}\n"
            f"- Bucket step: {config.bucket_step}\n"
            f"- Max aspect ratio: {config.max_aspect_ratio}\n"
            f"\nBucket Stats:\n"
            f"{self.bucket_manager.get_stats()}"
        )

    async def _process_data(self, image_dirs: List[str]) -> None:
        """Process data and assign to buckets efficiently."""
        stats = create_progress_stats(0)
        
        try:
            # Find all valid image files and calculate total size
            image_files = []
            total_size = 0
            for image_dir in image_dirs:
                files = find_matching_files(
                    image_dir,
                    extensions={'.jpg', '.jpeg', '.png', '.webp'},
                    recursive=True,
                    require_text_pair=True
                )
                for file in files:
                    file_size = get_file_size(file)
                    total_size += file_size
                image_files.extend(files)
                
            if not image_files:
                raise ValueError("No valid image-text pairs found!")
            
            # Update total in progress stats
            stats.total_items = len(image_files)
            logger.info(
                f"Found {len(image_files)} potential image-text pairs\n"
                f"Total dataset size: {total_size / (1024*1024*1024):.2f}GB"
            )
            
            # Process dataset using batch processor
            processed_items, final_stats = await self.batch_processor.process_dataset(
                items=image_files,
                progress_callback=lambda n, batch_stats: (
                    update_progress_stats(stats, n),
                    log_progress(
                        stats,
                        prefix="Processing Dataset - ",
                        extra_stats={
                            'batch_size': self.batch_processor.batch_size,
                            'gpu_memory': f"{get_gpu_memory_usage(self.device):.1%}",
                            'cache_hits': batch_stats.get('cache_hits', 0),
                            'cache_misses': batch_stats.get('cache_misses', 0)
                        }
                    ) if stats.should_log() else None
                )
            )
            
            # Update items list and assign to buckets
            self.items = processed_items
            
            # Log final statistics
            logger.info(
                f"\nData processing complete:\n"
                f"- Total files: {len(image_files)}\n"
                f"- Valid items: {len(self.items)}\n"
                f"- Success rate: {len(self.items)/len(image_files)*100:.1f}%\n"
                f"- Processing time: {format_time(final_stats.get('elapsed_seconds', 0))}\n"
                f"- Cache hits: {final_stats.get('cache_hits', 0)}\n"
                f"- Cache misses: {final_stats.get('cache_misses', 0)}\n"
                f"- Error types:\n" + "\n".join(
                    f"  - {k}: {v}" for k, v in final_stats.get('error_types', {}).items()
                )
            )
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise

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
                'tags': text_data['tags'],
                'tag_weights': text_data.get('tag_weights', None)
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.items)

    def get_sampler(self) -> AspectBatchSampler:
        """Get the aspect ratio-aware batch sampler."""
        return self.sampler

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
            
            # Clear CUDA cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up all dataset resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")