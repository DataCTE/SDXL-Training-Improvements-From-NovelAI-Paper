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
from src.config.config import NovelAIDatasetConfig

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
        self.device = torch.device(config.device)
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
            # Initialize tag weighter if configured
            tag_weighter = None
            if self.config.use_tag_weighting:
                tag_weighter = TagWeighter(
                    weight_ranges=self.config.tag_weight_ranges,
                    save_path=self.config.tag_weights_path
                )
                if self.config.tag_weights_path and Path(self.config.tag_weights_path).exists():
                    tag_weighter = TagWeighter.load(self.config.tag_weights_path)
            
            # Initialize bucket manager
            self.bucket_manager = BucketManager(self.config.bucket_config)
            
            # Initialize text embedder and processor
            self.text_embedder = TextEmbedder(
                tokenizers=self.tokenizers,
                text_encoders=self.text_encoders,
                config=self.config.text_embedder_config
            )
            self.text_processor = TextProcessor(
                config=self.config.text_processor_config,
                text_embedder=self.text_embedder,
                tag_weighter=tag_weighter
            )
            
            # Initialize image processor
            self.image_processor = ImageProcessor(
                config=self.config.image_processor_config,
                bucket_manager=self.bucket_manager,
                vae=self.vae
            )
            
            # Initialize cache manager
            self.cache_manager = CacheManager(self.config.cache_config)
            
            # Initialize batch processor
            self.batch_processor = BatchProcessor(
                config=self.config.batch_processor_config,
                image_processor=self.image_processor,
                text_processor=self.text_processor,
                cache_manager=self.cache_manager,
                vae=self.vae
            )
            
            # Process data
            await self._process_data(self.config.image_dirs)
            
            # Initialize sampler
            self.sampler = AspectBatchSampler(
                dataset=self,
                batch_size=self.config.batch_size,
                bucket_manager=self.bucket_manager,
                batch_processor=self.batch_processor
            )
            
        except Exception as e:
            logger.error(f"Dataset initialization failed: {e}")
            raise

    async def _process_data(self, image_dirs: List[str]):
        """Process all data with improved embedding handling."""
        try:
            # Find all image files with supported extensions
            image_files = await find_matching_files(
                image_dirs,
                extensions=['.png', '.jpg', '.jpeg', '.webp']
            )
            
            # Create progress tracker
            tracker = create_progress_tracker(
                total_items=len(image_files),
                desc="Processing dataset",
                unit="images"
            )
            
            # Process images and text in batches
            for batch in self._get_batches(image_files, self.config.batch_size):
                # Get text embeddings
                text_files = [Path(f).with_suffix('.txt') for f in batch]
                texts = [await asyncio.to_thread(f.read_text) for f in text_files if f.exists()]
                
                if texts:
                    # Process batch
                    batch_items = []
                    for img_path, text in zip(batch, texts):
                        # Get original image size for SDXL conditioning
                        with Image.open(img_path) as img:
                            original_size = (img.height, img.width)
                            
                        batch_items.append({
                            'image_path': img_path,
                            'text': text,
                            'original_size': original_size
                        })
                    
                    # Process batch
                    processed_items, batch_stats = await self.batch_processor.process_batch(
                        batch_items=batch_items,
                        cache_manager=self.cache_manager
                    )
                    
                    # Add successful items to dataset
                    for item in processed_items:
                        if item is not None:
                            self.items.append(item)
                            
                    # Update progress
                    update_tracker(tracker, **batch_stats)
                
                # Clear memory periodically
                if len(self.items) % 1000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
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
        
        # Load from cache if needed
        if isinstance(item.get('latents'), (str, Path)):
            cached = asyncio.run(self.cache_manager.load_cached_item(item['image_path']))
            if cached:
                item.update(cached)
        
        return item

    def __len__(self) -> int:
        return len(self.items)

    async def cleanup(self):
        """Clean up all resources."""
        try:
            # Clean up all processors
            await self.batch_processor.cleanup()
            await self.text_processor.cleanup()
            await self.image_processor.cleanup()
            
            # Clean up cache manager
            await self.cache_manager.cleanup()
            
            # Clear references to items
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