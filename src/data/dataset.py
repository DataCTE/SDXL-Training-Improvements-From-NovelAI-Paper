# src/data/dataset.py
from typing import List, Optional, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from PIL import Image
import time
import asyncio
from tqdm import tqdm

# Internal imports from processors
from .processors.text_embedder import TextEmbedder
from .processors.tag_weighter import TagWeighter
from .processors.image_processor import ImageProcessor, ImageProcessorConfig
from .processors.cache_manager import CacheManager
from .processors.batch_processor import BatchProcessor
from .processors.bucket import BucketManager
from .processors.sampler import AspectBatchSampler

# Internal imports from utils
from .processors.utils import (
    find_matching_files,
    ensure_dir,
    get_file_size,
    validate_image_text_pair,
    create_thread_pool,
    get_optimal_workers,
    get_system_resources,
    log_system_info,
    BatchConfig,
    process_in_chunks,
    calculate_optimal_batch_size,
    create_progress_stats,
    update_progress_stats,
    format_time,
    log_progress,
    load_and_validate_image,
    get_image_stats,
    get_optimal_thread_config
)

# Config import
from src.config.config import NovelAIDatasetConfig, TagWeightingConfig

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
        # Log system info at startup
        log_system_info("Dataset Initialization - ")
        
        self.config = config
        self.device = device
        self.vae = vae
        
        # Initialize text embedder internally
        self.text_embedder = TextEmbedder(
            pretrained_model_name_or_path=config.model_name,
            device=device,
            max_length=config.max_token_length,
            enable_memory_efficient_attention=True,
            max_memory_usage=0.9
        )

        # Initialize tag weighter internally
        tag_weighter_config = TagWeightingConfig(
            enabled=config.tag_weighting.enabled,
            min_weight=config.tag_weighting.min_weight,
            max_weight=config.tag_weighting.max_weight,
            default_weight=config.tag_weighting.default_weight,
            update_frequency=config.tag_weighting.update_frequency,
            smoothing_factor=config.tag_weighting.smoothing_factor
        )
        
        self.tag_weighter = TagWeighter(
            config=tag_weighter_config,
            text_embedder=self.text_embedder
        )
        
        # Get model dtype from VAE
        self.dtype = next(vae.parameters()).dtype
        
        # Get optimal thread configuration and resources
        thread_config = get_optimal_thread_config()
        resources = get_system_resources()
        
        # Calculate optimal batch size based on GPU memory
        optimal_batch_size = calculate_optimal_batch_size(
            device=device,
            min_batch_size=config.min_bucket_size,
            max_batch_size=64,  # Adjustable maximum
            target_memory_usage=0.9
        )
        
        # Get optimal number of workers based on system resources
        num_workers = get_optimal_workers(memory_per_worker_gb=1.0)  # 1GB per worker
        
        # Initialize components with optimized parameters
        self.bucket_manager = BucketManager(
            max_image_size=config.max_image_size,
            min_image_size=config.min_image_size,
            bucket_step=config.bucket_step,
            min_bucket_resolution=config.min_bucket_resolution,
            max_aspect_ratio=config.max_aspect_ratio,
            bucket_tolerance=config.bucket_tolerance
        )

        self.image_processor = ImageProcessor(
            ImageProcessorConfig(
                dtype=self.dtype,
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
            bucket_manager=self.bucket_manager
        )
        
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            max_workers=num_workers,
            use_caching=config.use_caching
        )
        
        # Initialize batch processor with optimized configuration
        self.batch_processor = BatchProcessor(
            image_processor=self.image_processor,
            cache_manager=self.cache_manager,
            text_embedder=self.text_embedder,
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
        loop.run_until_complete(self._process_data(image_dirs))
        
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
        
        # Enhanced logging with system resource information
        logger.info(
            f"Initialized dataset with {len(self)} samples\n"
            f"System Resources:\n"
            f"- Available Memory: {resources.available_memory_gb:.1f}GB\n"
            f"- GPU Memory: {resources.gpu_memory_total:.1f}GB total, "
            f"{resources.gpu_memory_used:.1f}GB used\n"
            f"- Workers: {num_workers}\n"
            f"- Optimal Batch Size: {optimal_batch_size}\n"
            f"\nConfig:\n"
            f"- Max image size: {config.max_image_size}\n"
            f"- Min image size: {config.min_image_size}\n"
            f"- Bucket step: {config.bucket_step}\n"
            f"- Max aspect ratio: {config.max_aspect_ratio}\n"
            f"- Cache enabled: {config.use_caching}\n"
            f"Bucket stats: {self.bucket_manager.get_stats()}"
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
            
            # Create batch configuration for processing
            batch_config = BatchConfig(
                batch_size=self.batch_processor.batch_size,
                device=self.device,
                max_memory_usage=0.9,
                prefetch_factor=2,
                log_interval=5.0
            )
            
            # Process files in chunks
            async def process_chunk(chunk_files: List[str], chunk_id: int) -> Tuple[List[Dict], Dict[str, int]]:
                chunk_items = []
                chunk_stats = {'total': 0, 'errors': 0, 'error_types': {}, 'skipped': 0}
                chunk_progress = create_progress_stats(len(chunk_files))
                
                for img_path in chunk_files:
                    try:
                        # Validate image-text pair
                        valid, reason = validate_image_text_pair(img_path)
                        if not valid:
                            chunk_stats['error_types'][reason] = chunk_stats['error_types'].get(reason, 0) + 1
                            chunk_stats['errors'] += 1
                            continue
                            
                        # Get image stats
                        img = load_and_validate_image(img_path)
                        if img is None:
                            chunk_stats['error_types']['invalid_image'] = chunk_stats['error_types'].get('invalid_image', 0) + 1
                            chunk_stats['errors'] += 1
                            continue
                            
                        img_stats = get_image_stats(img)
                        
                        # Find appropriate bucket
                        bucket = self.bucket_manager.find_bucket(img_stats['width'], img_stats['height'])
                        if bucket is None:
                            chunk_stats['error_types']['no_bucket'] = chunk_stats['error_types'].get('no_bucket', 0) + 1
                            chunk_stats['errors'] += 1
                            continue
                            
                        # Get cache paths
                        cache_paths = self.cache_manager.get_cache_paths(img_path)
                        
                        chunk_items.append({
                            'image_path': img_path,
                            'width': img_stats['width'],
                            'height': img_stats['height'],
                            'latent_cache': cache_paths['latent'],
                            'text_cache': cache_paths['text'],
                            'bucket_key': f"{bucket.width}x{bucket.height}"
                        })
                        chunk_stats['total'] += 1
                        
                        # Update and log progress
                        update_progress_stats(chunk_progress, 1)
                        if chunk_progress.should_log():
                            log_progress(
                                chunk_progress,
                                prefix=f"Chunk {chunk_id} - ",
                                extra_stats={
                                    'file_size': f"{get_file_size(img_path) / (1024*1024):.1f}MB",
                                    'bucket': bucket.width if bucket else "None"
                                }
                            )
                        
                    except Exception as e:
                        logger.error(f"Error processing {Path(img_path).name}: {str(e)[:200]}...")
                        chunk_stats['error_types']['exception'] = chunk_stats['error_types'].get('exception', 0) + 1
                        chunk_stats['errors'] += 1
                        continue
                
                return chunk_items, chunk_stats
            
            # Process chunks in parallel with batch configuration
            processed_items, final_stats = await process_in_chunks(
                items=image_files,
                chunk_size=batch_config.batch_size,
                process_fn=process_chunk,
                progress_callback=lambda n, stats: (
                    update_progress_stats(stats, n),
                    log_progress(stats, prefix="Overall Progress - ")
                    if stats.should_log(batch_config.log_interval)
                    else None
                )
            )
            
            # Update items list
            self.items = processed_items
            
            # Log final statistics
            logger.info(
                f"\nData processing complete:\n"
                f"- Total files: {len(image_files)}\n"
                f"- Valid items: {len(self.items)}\n"
                f"- Success rate: {len(self.items)/len(image_files)*100:.1f}%\n"
                f"- Processing time: {format_time(final_stats['elapsed_seconds'])}\n"
                f"- Error types:\n" + "\n".join(
                    f"  - {k}: {v}" for k, v in final_stats['error_types'].items()
                )
            )
            
        except Exception as e:
            logger.error(f"Fatal error in data processing: {e}")
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
                'tags': text_data['tags']
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.items)

    def get_sampler(self) -> AspectBatchSampler:
        """Get the aspect ratio-aware batch sampler."""
        return self.sampler