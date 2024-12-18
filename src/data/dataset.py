# src/data/dataset.py
from typing import List, Optional, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import logging
import asyncio
import gc
from PIL import Image

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
    ensure_dir,
    get_optimal_workers,
    get_system_resources,
    calculate_optimal_batch_size,
    get_gpu_memory_usage,
    log_progress
)
from .processors.utils.progress_utils import (
    create_progress_tracker,
    update_tracker,
    log_progress,
)

# Config import
from src.config.config import NovelAIDatasetConfig, ImageProcessorConfig, TextProcessorConfig, TextEmbedderConfig, BucketConfig, BatchProcessorConfig

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
        self.vae = vae.eval()  # Ensure VAE is in eval mode
        
        # Create progress tracker for initialization
        tracker = create_progress_tracker(
            total_items=1,  # For initialization progress
            device=device
        )
        
        try:
            # Get optimal configuration
            thread_config = get_optimal_thread_config()
            resources = get_system_resources()
            optimal_batch_size = calculate_optimal_batch_size(
                device=device,
                min_batch_size=config.min_bucket_size,
                max_batch_size=32,  # Reduced from 64 to prevent memory issues
                target_memory_usage=0.8  # Reduced from 0.9
            )
            num_workers = get_optimal_workers(memory_per_worker_gb=1.0)
            
            # Initialize components with progress tracking
            self.text_embedder = TextEmbedder(
                config=TextEmbedderConfig(
                    model_name=config.model_name,
                    device=device,
                    max_length=config.max_token_length,
                    enable_memory_efficient_attention=True,
                    max_memory_usage=0.8
                )
            )
            update_tracker(tracker, processed=1)

            # Initialize tag weighter with config
            self.tag_weighter = TagWeighter(config=config.tag_weighting)
            
            # Initialize processors
            self.text_processor = TextProcessor(
                config=TextProcessorConfig(
                    device=device,
                    batch_size=optimal_batch_size,
                    max_token_length=config.max_token_length,
                    use_caching=config.use_caching
                ),
                text_embedder=self.text_embedder,
                tag_weighter=self.tag_weighter
            )
            
            # Initialize bucket manager
            self.bucket_manager = BucketManager(
                config=BucketConfig(
                    max_image_size=config.max_image_size,
                    min_image_size=config.min_image_size,
                    bucket_step=config.bucket_step,
                    min_bucket_resolution=config.min_bucket_resolution,
                    max_aspect_ratio=config.max_aspect_ratio,
                    bucket_tolerance=config.bucket_tolerance
                )
            )
            
            # Log initialization progress
            if tracker.should_log():
                extra_stats = {
                    'batch_size': optimal_batch_size,
                    'workers': num_workers,
                    'memory_usage': f"{get_gpu_memory_usage(device):.1%}"
                }
                log_progress(tracker, prefix="Dataset initialization: ", extra_stats=extra_stats)
                
        except Exception as e:
            logger.error(f"Error during dataset initialization: {e}")
            raise
        
        # Initialize image processor
        self.image_processor = ImageProcessor(
            config=ImageProcessorConfig(
                dtype=next(vae.parameters()).dtype,
                device=device,
                max_image_size=config.max_image_size,
                min_image_size=config.min_image_size,
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
        
        # Initialize cache manager with smaller memory cache
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=num_workers,
            use_caching=config.use_caching
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            config=BatchProcessorConfig(
                device=device,
                batch_size=optimal_batch_size,
                prefetch_factor=thread_config.prefetch_factor,
                max_memory_usage=0.8,
                num_workers=num_workers
            ),
            image_processor=self.image_processor,
            text_processor=self.text_processor,
            cache_manager=self.cache_manager,
            vae=self.vae
        )

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Create cache directory
        ensure_dir(config.cache_dir)
        
        # Process data and initialize items (store only paths and metadata, not tensors)
        self.items = []
        try:
            loop.run_until_complete(self._process_data(image_dirs))
        finally:
            # Cleanup resources
            if hasattr(self.batch_processor, 'cleanup'):
                loop.run_until_complete(self.batch_processor.cleanup())
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
        
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

        # After initializing self.bucket_manager, log how many buckets we got:
        logger.info(f"BucketManager has {len(self.bucket_manager.buckets)} buckets after init")

        # If none are created, consider adjusting min_bucket_resolution, aspect ratio, etc.
        if not self.bucket_manager.buckets:
            logger.warning("No buckets created! Check min_bucket_resolution and max_aspect_ratio settings.")

        # Gather existing image files
        all_image_paths = []
        for image_dir in image_dirs:
            found_files = find_matching_files(image_dir, [".jpg", ".jpeg", ".png"])
            all_image_paths.extend(found_files)

        # Build a quick list of (width, height) for each image
        # If you have a specialized routine for reading dims, call it here
        image_dimensions = []
        for path in all_image_paths:
            # Minimal example: open image with PIL just to get size
            with Image.open(path) as img:
                width, height = img.size
            image_dimensions.append({"width": width, "height": height})

        # Create dynamic buckets from actual image sizes
        self.bucket_manager.create_buckets_from_dataset(
            items=image_dimensions, 
            min_count_for_bucket=1
        )

        # Recalculate image processor buffer now that buckets exist
        self.image_processor.recalc_buffer_size()

        logger.info(
            f"BucketManager has {len(self.bucket_manager.buckets)} buckets after creation "
            f"from real image dimensions"
        )

        if not self.bucket_manager.buckets:
            logger.warning("No buckets created! Check min_bucket_resolution and max_aspect_ratio settings.")

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

    async def _process_data(self, image_dirs: List[str]) -> None:
        """Process all data files asynchronously with progress tracking."""
        try:
            # Find all image files asynchronously
            image_files = []
            for image_dir in image_dirs:
                async for found_file in find_matching_files(
                    image_dir, 
                    ['.jpg', '.jpeg', '.png'],
                    batch_size=self.config.batch_size
                ):
                    image_files.append(found_file)
                    
                    # Yield control periodically
                    if len(image_files) % (self.config.batch_size * 2) == 0:
                        await asyncio.sleep(0)
                
            if not image_files:
                raise ValueError(f"No valid image files found in {image_dirs}")
                
            # Create tracker with total files
            tracker = create_progress_tracker(
                total_items=len(image_files),
                batch_size=self.config.batch_size,
                device=self.device
            )

            # Process files in batches
            processed_items = []
            for i in range(0, len(image_files), self.config.batch_size):
                batch_files = image_files[i:i + self.config.batch_size]
                batch_items = [{'image_path': f} for f in batch_files]
                
                # Process batch
                batch_processed, stats = await self.batch_processor.process_batch(
                    batch_items=batch_items,
                    width=self.config.image_size[0],
                    height=self.config.image_size[1],
                    cache_manager=self.cache_manager
                )
                
                processed_items.extend(batch_processed)
                
                # Update progress
                update_tracker(
                    tracker,
                    processed=len(batch_processed),
                    failed=stats.get('errors', 0)
                )
                
                # Log progress periodically
                if tracker.should_log():
                    extra_stats = {
                        'processed': len(processed_items),
                        'memory': f"{get_gpu_memory_usage(self.device):.1%}",
                        'batch_size': self.config.batch_size
                    }
                    log_progress(tracker, prefix="Processing dataset: ", extra_stats=extra_stats)

                # Allow other tasks to run
                await asyncio.sleep(0)
                
            self.items = processed_items

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
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