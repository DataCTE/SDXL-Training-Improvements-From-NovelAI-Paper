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

# Internal imports from data package
from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter
from src.data.image_processor import ImageProcessor, ImageProcessorConfig
from src.data.cache_manager import CacheManager
from src.data.batch_processor import BatchProcessor
from src.data.bucket import BucketManager
from src.data.thread_config import get_optimal_thread_config
from src.data.utils import (
    find_matching_files,
    create_thread_pool,
    get_optimal_workers,
    calculate_optimal_batch_size,
    get_system_resources,
    log_system_info,
    create_progress_stats,
    process_in_chunks,
    format_time,
    load_and_validate_image,
    get_image_stats,
    validate_image_text_pair,
    BatchConfig,
    BatchProcessor
)
from src.data.utils.system_utils import calculate_chunk_size
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
        # Log system info at startup
        log_system_info("Dataset Initialization - ")
        
        self.config = config
        self.device = device
        self.text_embedder = text_embedder
        self.tag_weighter = tag_weighter
        self.vae = vae
        
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
            config=BatchConfig(
                batch_size=optimal_batch_size,
                device=device,
                dtype=self.dtype,
                max_memory_usage=0.9,
                prefetch_factor=thread_config.prefetch_factor,
                log_interval=5.0
            ),
            executor=create_thread_pool(num_workers),
            name="DatasetBatchProcessor"
        )

        # Process data and initialize items
        self.items = []
        asyncio.run(self._process_data(image_dirs))
        
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
            
            # Update total in progress stats
            stats.total_items = len(image_files)
            logger.info(f"Found {len(image_files)} potential image-text pairs")
            
            # Calculate chunk size for parallel processing
            chunk_size = calculate_chunk_size(
                total_items=len(image_files),
                optimal_workers=get_optimal_workers(),
                min_chunk_size=100
            )
            
            async def process_chunk(chunk_files: List[str], chunk_id: int) -> Tuple[List[Dict], Dict[str, int]]:
                chunk_items = []
                chunk_stats = {'total': 0, 'errors': 0, 'error_types': {}, 'skipped': 0}
                logger.debug(f"Processing chunk {chunk_id} with {len(chunk_files)} files")
                
                for img_path in chunk_files:
                    try:
                        # Check if cache exists
                        cache_paths = self.cache_manager.get_cache_paths(img_path)
                        if all(path.exists() for path in cache_paths.values()):
                            # Add to items without processing
                            img = Image.open(img_path)
                            width, height = img.size
                            img.close()
                            
                            bucket = self.bucket_manager.find_bucket(width, height)
                            if bucket is None:
                                chunk_stats['error_types']['no_bucket'] = \
                                    chunk_stats['error_types'].get('no_bucket', 0) + 1
                                chunk_stats['errors'] += 1
                                continue
                                
                            chunk_items.append({
                                'image_path': img_path,
                                'width': width,
                                'height': height,
                                'latent_cache': cache_paths['latent'],
                                'text_cache': cache_paths['text'],
                                'bucket_key': f"{bucket.width}x{bucket.height}"
                            })
                            chunk_stats['skipped'] += 1
                            chunk_stats['total'] += 1
                            continue
                            
                        # Process uncached images
                        img = load_and_validate_image(img_path)
                        if img is None:
                            chunk_stats['error_types']['invalid_image'] += 1
                            chunk_stats['errors'] += 1
                            continue
                        
                        # Validate text pair
                        valid, reason = validate_image_text_pair(img_path)
                        if not valid:
                            chunk_stats['error_types'][reason] = \
                                chunk_stats['error_types'].get(reason, 0) + 1
                            chunk_stats['errors'] += 1
                            continue
                        
                        # Get image stats
                        img_stats = get_image_stats(img)
                        
                        # Find appropriate bucket
                        bucket = self.bucket_manager.find_bucket(
                            img_stats['width'], 
                            img_stats['height']
                        )
                        if bucket is None:
                            chunk_stats['error_types']['no_bucket'] = \
                                chunk_stats['error_types'].get('no_bucket', 0) + 1
                            chunk_stats['errors'] += 1
                            continue
                        
                        # Get cache paths
                        cache_paths = self.cache_manager.get_cache_paths(img_path)
                        
                        # Generate and cache latents
                        pixel_values = self.image_processor.preprocess(img)
                        latents = self.image_processor.encode_vae(self.vae, pixel_values)
                        await self.cache_manager.save_latent_async(cache_paths['latent'], latents)

                        # Generate and cache text embeddings
                        text_data = self.text_embedder.encode_text(img_path)
                        await self.cache_manager.save_text_data_async(cache_paths['text'], text_data)

                        # Add to items as before
                        chunk_items.append({
                            'image_path': img_path,
                            'width': img_stats['width'],
                            'height': img_stats['height'],
                            'latent_cache': cache_paths['latent'],
                            'text_cache': cache_paths['text'],
                            'bucket_key': f"{bucket.width}x{bucket.height}"
                        })
                        chunk_stats['total'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
                        chunk_stats['error_types']['exception'] = \
                            chunk_stats['error_types'].get('exception', 0) + 1
                        chunk_stats['errors'] += 1
                        continue
                
                return chunk_items, chunk_stats
            
            # Process chunks in parallel with async support
            pbar = tqdm(
                total=len(image_files),
                desc="Processing images",
                unit="img",
                dynamic_ncols=True,
                position=0
            )
            
            processed_items, final_stats = await process_in_chunks(
                items=image_files,
                chunk_size=chunk_size,
                process_fn=process_chunk,
                num_workers=get_optimal_workers(),
                progress_interval=0.1,
                progress_callback=lambda n: pbar.update(n)
            )
            pbar.close()
            
            # Update items list
            self.items = processed_items
            
            # Log final statistics
            logger.info(
                f"\nData processing complete:\n"
                f"- Total files: {len(image_files)}\n"
                f"- Valid items: {len(self.items)}\n"
                f"- Success rate: {len(self.items)/len(image_files)*100:.1f}%\n"
                f"- Processing time: {format_time(final_stats['elapsed_seconds'])}\n"
                f"- Cache hits: {final_stats.get('cache_hits', 0)}\n"
                f"- Cache misses: {final_stats.get('cache_misses', 0)}\n"
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