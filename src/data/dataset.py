# src/data/dataset.py
from typing import List, Optional, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import logging
import asyncio
import gc
from pathlib import Path

# Internal imports from processors
from .processors.text_processor import TextProcessor
from .processors.batch_processor import BatchProcessor
from .processors.image_processor import ImageProcessor
from .processors.cache_manager import CacheManager
from .processors.bucket import BucketManager
from .processors.sampler import AspectBatchSampler

# Import utilities
from .processors.utils.caption.tag_weighter import TagWeighter
from .processors.utils.batch_utils import find_matching_files
from .processors.utils.progress_utils import create_progress_tracker, update_tracker, log_progress


# Config import
from src.config.config import NovelAIDatasetConfig, BucketConfig, TextProcessorConfig, ImageProcessorConfig, BatchProcessorConfig, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS

logger = logging.getLogger(__name__)

class NovelAIDataset(Dataset):
    """
    Dataset for training with aspect ratio bucketing. Integrates updated sampler, batch_processor,
    and the rest of the pipeline for caching, text/image processing, and concurrency.
    """

    def __init__(
        self,
        config: NovelAIDatasetConfig,
        vae=None,
        text_encoders=None,
        tokenizers=None,
        tag_weighter=None
    ):
        """
        Initialize dataset. For full setup of Sampler/BatchProcessor, use the async create() method.
        """
        self.config = config
        if not self.config.image_dirs:
            raise ValueError("No image directories specified in dataset config")

        # Attempt to detect GPU if provided VAE is on cuda
        if vae is not None and any(p.device.type == "cuda" for p in vae.parameters()):
            self.device = next(vae.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.warning(f"No VAE GPU device found; defaulting to: {self.device}")

        # Core references
        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.tag_weighter = tag_weighter

        # Primary dataset storage
        self.items: List[Dict[str, Any]] = []

        # Optional references for processing
        self.bucket_manager: Optional[BucketManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.text_processor: Optional[TextProcessor] = None
        self.image_processor: Optional[ImageProcessor] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.sampler: Optional[AspectBatchSampler] = None

    @classmethod
    async def create(
        cls,
        config: NovelAIDatasetConfig,
        vae=None,
        text_encoders=None,
        tokenizers=None,
        tag_weighter=None
    ) -> "NovelAIDataset":
        """
        Asynchronous creation method. Initializes the dataset, bucket manager,
        cache, text/image processors, and sampler if applicable.
        """
        self = cls(config, vae, text_encoders, tokenizers, tag_weighter)
        await self._initialize()
        return self

    async def _initialize(self):
        """
        Main async initialization—creates managers, finds files, processes them into items,
        and optionally sets up the sampler for training/inference usage.
        """
        try:
            logger.debug("Starting dataset initialization...")

            # Optional TagWeighter
            tag_weighter_loaded = None
            if self.config.use_tag_weighting:
                tag_weighter_loaded = TagWeighter(
                    weight_ranges=self.config.tag_weight_ranges,
                    save_path=self.config.tag_weights_path
                )
                if self.config.tag_weights_path and Path(self.config.tag_weights_path).exists():
                    tag_weighter_loaded = TagWeighter.load(self.config.tag_weights_path)
            """ 
            max_image_size: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
            min_image_size: Tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE
            bucket_step: int = 8
            min_bucket_resolution: int = 65536
            max_aspect_ratio: float = 2.0
            bucket_tolerance: float = 0.2
            target_resolutions: List[Tuple[int, int]] = field(
                default_factory=lambda: DEFAULT_TARGET_RESOLUTIONS
            )
            max_ar_error: float = DEFAULT_MAX_AR_ERROR"""    
            # BucketManager - unpack the configuration
            self.bucket_manager = BucketManager(
                config=BucketConfig(
                    image_size=self.config.image_size,
                    min_size=self.config.min_image_size,
                    max_size=self.config.max_image_size,
                    step=self.config.bucket_step,
                    min_resolution=self.config.min_bucket_resolution,
                    max_ar=self.config.max_aspect_ratio,
                    tolerance=self.config.bucket_tolerance,
                    target_resolutions=self.config.target_resolutions,
                    max_ar_error=self.config.max_ar_error
                )
)

            # CacheManager
            self.cache_manager = CacheManager(
                use_memory_cache=self.config.cache_config.use_memory_cache,
                use_caching=self.config.cache_config.use_caching,
                cache_dir=self.config.cache_config.cache_dir,
                cache_format=self.config.cache_config.cache_format
            )
            """ 
            num_workers: int = DEFAULT_NUM_WORKERS
            batch_size: int = DEFAULT_BATCH_SIZE
            max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
            
            # Tag weighting settings
            enable_tag_weighting: bool = True
            tag_frequency_threshold: int = 5
            tag_weight_smoothing: float = 0.1

    # Add this attribute to allow passing 'prefetch_factor' from your YAML
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR
    proportion_empty_prompts: float = 0.0"""

            # TextProcessor
            self.text_processor = TextProcessor(
                num_workers=self.config.text_processor_config.get('num_workers', DEFAULT_NUM_WORKERS),
                batch_size=self.config.text_processor_config.get('batch_size', DEFAULT_BATCH_SIZE),
                max_token_length=self.config.max_token_length,
                enable_tag_weighting=self.config.use_tag_weighting,
                tag_frequency_threshold=self.config.text_processor_config.get('tag_frequency_threshold', 5),
                tag_weight_smoothing=self.config.text_processor_config.get('tag_weight_smoothing', 0.1),
                prefetch_factor=self.config.prefetch_factor,
                proportion_empty_prompts=self.config.proportion_empty_prompts
            )

            # ImageProcessor
            self.image_processor = ImageProcessor(
                config=ImageProcessorConfig(
                    device=str(self.device),
                    max_image_size=self.config.max_image_size,
                    min_image_size=self.config.min_image_size,
                    enable_vae_slicing=self.config.image_processor_config.get('enable_vae_slicing', True),
                    vae_batch_size=self.config.batch_size,
                    num_workers=self.config.image_processor_config.get('num_workers', DEFAULT_NUM_WORKERS),
                    prefetch_factor=self.config.prefetch_factor
                ),
                bucket_manager=self.bucket_manager,
                vae=self.vae
            )

            # BatchProcessor
            self.batch_processor = BatchProcessor(
                config=BatchProcessorConfig(
                    device=str(self.device),
                    batch_size=self.config.batch_size,
                    prefetch_factor=self.config.prefetch_factor,
                    num_workers=self.config.batch_processor_config.num_workers,
                    max_memory_usage=self.config.batch_processor_config.max_memory_usage,
                    memory_check_interval=self.config.batch_processor_config.memory_check_interval,
                    memory_growth_factor=self.config.batch_processor_config.memory_growth_factor,
                    high_memory_threshold=self.config.batch_processor_config.high_memory_threshold,
                    cleanup_interval=self.config.batch_processor_config.cleanup_interval,
                    retry_count=self.config.batch_processor_config.retry_count,
                    backoff_factor=self.config.batch_processor_config.backoff_factor,
                    min_batch_size=self.config.batch_processor_config.min_batch_size,
                    max_batch_size=self.config.batch_processor_config.max_batch_size
                ),
                image_processor=self.image_processor,
                text_processor=self.text_processor,
                cache_manager=self.cache_manager,
                vae=self.vae
            )

            # Collect image files
            image_files = []
            for img_dir in self.config.image_dirs:
                found = find_matching_files(
                    img_dir, supported_exts=[".png", ".jpg", ".jpeg", ".webp", ".bmp"]
                )
                image_files.extend(found)

            if not image_files:
                raise ValueError("No image files found in the specified directories")

            missing_text_count = 0
            file_read_errors = 0
            image_errors = 0
            processed_count = 0

            # Create a progress tracker for overall dataset processing
            tracker = create_progress_tracker(
                total_items=len(image_files),
                batch_size=self.config.batch_size,
                device=str(self.device)
            )

            # Process data in chunks
            chunk_size = self.config.batch_size
            for start_idx in range(0, len(image_files), chunk_size):
                batch_items = []
                batch = image_files[start_idx:start_idx + chunk_size]

                for img_path in batch:
                    # Prepare dictionary describing each file
                    text_file = Path(img_path).with_suffix(".txt")
                    has_text_file = text_file.exists()
                    if not has_text_file:
                        missing_text_count += 1

                    batch_items.append({
                        "image_path": img_path,
                        "has_text_file": has_text_file
                    })

                try:
                    # Use BatchProcessor to handle chunk
                    processed_items, batch_stats = await self.batch_processor.process_batch(
                        batch_items=batch_items
                    )

                    if processed_items:
                        self.items.extend(processed_items)
                        processed_count += len(processed_items)

                    # Update tracker with new stats
                    update_tracker(
                        tracker,
                        processed=len(processed_items),
                        failed=batch_stats.get('errors', 0),
                        cache_hits=batch_stats.get('cache_hits', 0),
                        cache_misses=batch_stats.get('cache_misses', 0),
                        error_types=batch_stats.get('error_types', None)
                    )

                    log_progress(
                        stats=tracker,
                        prefix="Dataset Processing",
                        extra_stats={"batch_size": len(batch)},
                        log_interval=5
                    )

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    image_errors += len(batch_items)

                # Periodic memory cleanup
                if len(self.items) % 8000 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            logger.info("Dataset processing completed:")
            logger.info(f"- Total image files: {len(image_files)}")
            logger.info(f"- Processed successfully: {processed_count}")
            logger.info(f"- Missing text files: {missing_text_count}")
            logger.info(f"- Text file read errors: {file_read_errors}")
            logger.info(f"- Image processing errors: {image_errors}")

            if processed_count == 0:
                raise ValueError("No valid items were processed from the image directories")

            # If you need to incorporate the Sampler for training:
            self.sampler = AspectBatchSampler(
                dataset=self,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=self.config.drop_last,
                max_consecutive_batch_samples=self.config.max_consecutive_batch_samples,
                min_bucket_length=self.config.min_bucket_length,
                debug_mode=self.config.debug_mode,
                prefetch_factor=self.config.prefetch_factor,
                bucket_manager=self.bucket_manager,
                config=self.config
            )

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _get_batches(self, items: List[Any], batch_size: int):
        """
        Yield batches of items from the list. (If you need manual chunking logic.)
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve item from the dataset (with possible cache lookups).
        """
        item = self.items[idx]

        # Lazy load latents if on disk
        if isinstance(item.get("latents"), (str, Path)):
            cached = asyncio.run(self.cache_manager.load_cached_item(item["image_path"]))
            if cached:
                item.update(cached)
        return item

    def __len__(self) -> int:
        return len(self.items)

    async def cleanup(self):
        """
        Clean up all resources: processors, caches, and GPU memory.
        """
        try:
            if self.batch_processor:
                await self.batch_processor.cleanup()
            if self.text_processor:
                await self.text_processor.cleanup()
            if self.image_processor:
                await self.image_processor.cleanup()
            if self.cache_manager:
                await self.cache_manager.cleanup()

            self.items.clear()

            if hasattr(self, "sampler") and self.sampler is not None:
                await self.sampler.cleanup()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            logger.info("Successfully cleaned up all dataset resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """
        Ensure an async cleanup is performed upon deletion if possible.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during dataset deletion: {e}")