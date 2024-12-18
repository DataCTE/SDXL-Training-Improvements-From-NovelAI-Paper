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
from .processors.utils.batch_utils import find_matching_files
from .processors.utils.progress_utils import create_progress_tracker, update_tracker
from .processors.utils.system_utils import get_gpu_memory_usage, cleanup_processor

# Config import
from src.config.config import NovelAIDatasetConfig, BucketConfig, TextProcessorConfig, ImageProcessorConfig

logger = logging.getLogger(__name__)

class NovelAIDataset(Dataset):
    """Dataset for training Stable Diffusion XL with aspect ratio bucketing."""
    
    def __init__(
        self,
        config: NovelAIDatasetConfig,
        vae = None,
        text_encoders = None,
        tokenizers = None,
        tag_weighter = None
    ):
        """Initialize dataset - use create() for async initialization."""
        self.config = config
        if not self.config.image_dirs:
            raise ValueError("No image directories specified in dataset config")
            
        # Safely find the device
        if vae is not None and any(p.device.type == "cuda" for p in vae.parameters()):
            self.device = next(vae.parameters()).device
        else:
            # Fallback to CPU or config device if no VAE or GPU found
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.warning(f"No VAE GPU device found; defaulting to: {self.device}")

        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.tag_weighter = tag_weighter
        self.items = []
        
    @classmethod
    async def create(
        cls,
        config: NovelAIDatasetConfig,
        vae = None,
        text_encoders = None,
        tokenizers = None,
        tag_weighter = None
    ) -> "NovelAIDataset":
        """Async factory method for initialization."""
        self = cls(config, vae, text_encoders, tokenizers, tag_weighter)
        await self._initialize()
        return self

    async def _initialize(self):
        """Initialize dataset components and process data."""
        try:
            logger.debug("Starting dataset initialization...")
            
            # Initialize tag weighter if configured
            tag_weighter_loaded = None
            if self.config.use_tag_weighting:
                tag_weighter_loaded = TagWeighter(
                    weight_ranges=self.config.tag_weight_ranges,
                    save_path=self.config.tag_weights_path
                )
                if self.config.tag_weights_path and Path(self.config.tag_weights_path).exists():
                    tag_weighter_loaded = TagWeighter.load(self.config.tag_weights_path)
            
            # Create bucket manager
            bucket_config = BucketConfig(
                max_image_size=self.config.max_image_size,
                min_image_size=self.config.min_image_size,
                bucket_step=self.config.bucket_step,
                min_bucket_resolution=self.config.min_bucket_resolution or min(self.config.min_image_size),
                max_aspect_ratio=self.config.max_aspect_ratio,
                bucket_tolerance=self.config.bucket_tolerance
            )
            self.bucket_manager = BucketManager(bucket_config)
            
            # Create text embedder
            self.text_embedder = TextEmbedder(
                tokenizers=self.tokenizers,
                text_encoders=self.text_encoders,
                config=self.config.text_embedder_config
            )
            
            # Initialize text processor
            text_processor_config = TextProcessorConfig(**self.config.text_processor_config)
            self.text_processor = TextProcessor(
                config=text_processor_config,
                text_embedder=self.text_embedder,
                tag_weighter=tag_weighter_loaded
            )
            
            # Initialize image processor
            image_processor_config = ImageProcessorConfig(**self.config.image_processor_config)
            self.image_processor = ImageProcessor(
                config=image_processor_config,
                bucket_manager=self.bucket_manager,
                vae=self.vae
            )
            
            # Create cache manager
            self.cache_manager = CacheManager(self.config.cache_config)
            
            # Initialize batch processor
            self.batch_processor = BatchProcessor(
                config=self.config.batch_processor_config,
                image_processor=self.image_processor,
                text_processor=self.text_processor,
                cache_manager=self.cache_manager,
                vae=self.vae
            )
            
            # Process all data
            await self._process_data(self.config.image_dirs)
            
            if not self.items:
                raise ValueError("No valid items were loaded from the specified image directories")
            
            # Now create the sampler
            self.sampler = AspectBatchSampler(
                dataset=self,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=self.config.drop_last,
                max_consecutive_batch_samples=self.config.max_consecutive_batch_samples,
                min_bucket_length=self.config.min_bucket_length,
                debug_mode=self.config.debug_mode,
                prefetch_factor=self.config.prefetch_factor,
                bucket_manager=self.bucket_manager
            )
            
            logger.info(f"Dataset initialization completed: loaded {len(self.items)} items")
            
        except Exception as e:
            logger.error(f"Dataset initialization failed: {e}")
            raise

    async def _process_data(self, image_dirs: List[str]):
        """
        Process all data by delegating image loading and
        transformations to ImageProcessor.
        """
        try:
            if not image_dirs:
                raise ValueError("No image directories specified in config")
            
            # Check directory existence
            for dir_path in image_dirs:
                if not Path(dir_path).exists():
                    logger.error(f"Image directory does not exist: {dir_path}")
                    raise ValueError(f"Image directory does not exist: {dir_path}")
                logger.info(f"Processing directory: {dir_path}")

            # Find all image files
            image_files = await find_matching_files(
                image_dirs,
                extensions=['.png', '.jpg', '.jpeg', '.webp']
            )
            
            if not image_files:
                logger.error("No image files found in specified directories")
                raise ValueError("No image files found in specified directories")
            
            logger.info(f"Found {len(image_files)} image files")
            
            # Create progress tracker
            tracker = create_progress_tracker(
                total_items=len(image_files),
                desc="Processing dataset",
                unit="images"
            )
            
            processed_count = 0
            missing_text_count = 0
            file_read_errors = 0
            image_errors = 0
            
            # Batch iteration
            for batch in self._get_batches(image_files, self.config.batch_size):
                # Collect text in parallel
                text_files = [Path(f).with_suffix('.txt') for f in batch]
                texts = []
                for f, img_path in zip(text_files, batch):
                    if f.exists():
                        try:
                            # Read text on a thread to avoid blocking
                            text = await asyncio.to_thread(f.read_text)
                            texts.append(text)
                        except Exception as e:
                            logger.warning(f"Failed to read text file {f}: {e}")
                            file_read_errors += 1
                    else:
                        logger.warning(f"Missing text file for image: {img_path}")
                        missing_text_count += 1
                        texts.append("")  # Optionally allow empty text
                
                # Use ImageProcessor instead of manually opening images
                batch_items = []
                for img_path, text in zip(batch, texts):
                    # Let the processor handle loading and transformations
                    # Optionally pass original_size=None for dynamic inference
                    processed_image = await self.image_processor.process_image(
                        image=img_path,
                        original_size=None  # or manually pass if you have prior info
                    )
                    if processed_image is None:
                        logger.warning(f"Skipping corrupted or missing image: {img_path}")
                        continue

                    batch_items.append({
                        "image_path": img_path,
                        "pixel_values": processed_image["pixel_values"],
                        "latents": processed_image.get("latents"),
                        "original_size": processed_image["original_size"],
                        "text": text
                    })

                if not batch_items:
                    continue

                # Process the batch
                try:
                    processed_items, batch_stats = await self.batch_processor.process_batch(
                        batch_items=batch_items,
                        cache_manager=self.cache_manager
                    )
                    for item in processed_items:
                        if item is not None:
                            self.items.append(item)
                            processed_count += 1
                    
                    # Update tracker
                    update_tracker(tracker, **batch_stats)
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    image_errors += len(batch_items)
                
                # Periodic memory cleanup
                if len(self.items) % 8000 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            
            # Final logging
            logger.info("Dataset processing completed:")
            logger.info(f"- Total image files: {len(image_files)}")
            logger.info(f"- Processed successfully: {processed_count}")
            logger.info(f"- Missing text files: {missing_text_count}")
            logger.info(f"- Text file read errors: {file_read_errors}")
            logger.info(f"- Image processing errors: {image_errors}")
            
            if processed_count == 0:
                raise ValueError("No valid items were processed from the image directories")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _get_batches(self, items: List[Any], batch_size: int):
        """Yield batches of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with lazy loading from cache."""
        item = self.items[idx]
        if isinstance(item.get('latents'), (str, Path)):
            # Potentially run in the current loop or handle properly if the environment is async
            cached = asyncio.run(self.cache_manager.load_cached_item(item['image_path']))
            if cached:
                item.update(cached)
        return item

    def __len__(self) -> int:
        return len(self.items)

    async def cleanup(self):
        """Clean up all resources."""
        try:
            await self.batch_processor.cleanup()
            await self.text_processor.cleanup()
            await self.image_processor.cleanup()
            await self.cache_manager.cleanup()

            self.items.clear()

            if hasattr(self, 'sampler'):
                del self.sampler

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