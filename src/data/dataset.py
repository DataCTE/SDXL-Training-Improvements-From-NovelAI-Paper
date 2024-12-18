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
                    num_workers=num_workers
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

    async def _process_data(self, image_dirs: List[str]) -> None:
        """Process all data files asynchronously with progress tracking."""
        try:
            # Log system info at start
            log_system_info()
            logger.info("Starting dataset processing...")
            
            # Calculate optimal batch size with logging
            optimal_batch_size = calculate_optimal_batch_size(
                device=self.device,
                min_batch_size=self.config.min_bucket_size,
                max_batch_size=32,
                target_memory_usage=0.8
            )
            logger.info(f"Calculated optimal batch size: {optimal_batch_size}")

            # Find all image files asynchronously with detailed progress
            logger.info("Scanning directories for image files...")
            image_files = []
            scan_stats = {
                'total_files': 0,
                'valid_files': 0,
                'skipped_files': 0,
                'error_files': 0
            }
            
            for image_dir in image_dirs:
                logger.info(f"Scanning directory: {image_dir}")
                dir_stats = {'files': 0, 'errors': 0}
                
                async for found_file in find_matching_files(
                    image_dir, 
                    ['.jpg', '.jpeg', '.png'],
                    batch_size=optimal_batch_size
                ):
                    try:
                        # Validate file
                        is_valid, reason = validate_image_text_pair(found_file)
                        if is_valid:
                            image_files.append(found_file)
                            scan_stats['valid_files'] += 1
                        else:
                            scan_stats['skipped_files'] += 1
                            logger.debug(f"Skipped {found_file}: {reason}")
                            
                        dir_stats['files'] += 1
                        scan_stats['total_files'] += 1
                        
                        # Log progress periodically
                        if dir_stats['files'] % 1000 == 0:
                            logger.info(
                                f"Processed {dir_stats['files']} files in {image_dir}\n"
                                f"- Valid: {scan_stats['valid_files']}\n"
                                f"- Skipped: {scan_stats['skipped_files']}\n"
                                f"- Errors: {scan_stats['error_files']}"
                            )
                            
                    except Exception as e:
                        dir_stats['errors'] += 1
                        scan_stats['error_files'] += 1
                        log_error_with_context(e, f"Error processing file {found_file}")
                    
                    # Yield control periodically
                    if len(image_files) % (optimal_batch_size * 2) == 0:
                        await asyncio.sleep(0)
                
                logger.info(
                    f"Completed scanning {image_dir}:\n"
                    f"- Total files: {dir_stats['files']}\n"
                    f"- Errors: {dir_stats['errors']}"
                )
                
            if not image_files:
                raise ValueError(f"No valid image files found in {image_dirs}")
            
            # Log final scan results
            logger.info(
                f"\nDirectory scan completed:\n"
                f"- Total files found: {scan_stats['total_files']}\n"
                f"- Valid files: {scan_stats['valid_files']}\n"
                f"- Skipped files: {scan_stats['skipped_files']}\n"
                f"- Error files: {scan_stats['error_files']}"
            )
                
            # Create tracker with total files
            tracker = create_progress_tracker(
                total_items=len(image_files),
                batch_size=optimal_batch_size,
                device=self.device
            )

            # Process files in batches with detailed metrics
            processed_items = []
            total_batches = (len(image_files) + optimal_batch_size - 1) // optimal_batch_size
            
            logger.info(
                f"\nStarting batch processing:\n"
                f"- Total files: {len(image_files)}\n"
                f"- Batch size: {optimal_batch_size}\n"
                f"- Total batches: {total_batches}"
            )
            
            for batch_num, i in enumerate(range(0, len(image_files), optimal_batch_size)):
                batch_files = image_files[i:i + optimal_batch_size]
                batch_items = [{'image_path': f} for f in batch_files]
                
                try:
                    # Process batch with metrics
                    batch_start = time.time()
                    batch_processed, stats = await self.batch_processor.process_batch(
                        batch_items=batch_items,
                        width=self.config.image_size[0],
                        height=self.config.image_size[1],
                        cache_manager=self.cache_manager
                    )
                    batch_time = time.time() - batch_start
                    
                    processed_items.extend(batch_processed)
                    
                    # Log detailed batch metrics
                    metrics = {
                        'batch': f"{batch_num + 1}/{total_batches}",
                        'processed': len(batch_processed),
                        'failed': len(batch_files) - len(batch_processed),
                        'batch_time': f"{batch_time:.2f}s",
                        'items_per_second': f"{len(batch_processed)/batch_time:.1f}",
                        'memory': f"{get_gpu_memory_usage(self.device):.1%}",
                        'cache_hits': stats.get('cache_hits', 0),
                        'cache_misses': stats.get('cache_misses', 0)
                    }
                    
                    log_metrics(
                        metrics=metrics,
                        step=batch_num,
                        is_main_process=True,
                        use_wandb=self.config.use_wandb,
                        step_type="batch"
                    )

                except Exception as e:
                    log_error_with_context(
                        error=e,
                        context=f"Error processing batch {batch_num + 1}/{total_batches}"
                    )
                    continue

                # Allow other tasks to run
                await asyncio.sleep(0)
                
            logger.info(
                f"\nDataset processing completed:\n"
                f"- Total files processed: {len(processed_items)}/{len(image_files)}\n"
                f"- Success rate: {(len(processed_items)/len(image_files))*100:.1f}%\n"
                f"- Total time: {format_time(tracker.elapsed)}"
            )
            
            self.items = processed_items

        except Exception as e:
            log_error_with_context(e, "Error in dataset processing")
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