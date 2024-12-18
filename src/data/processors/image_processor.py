# src/data/processors/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
from torchvision import transforms
import numpy as np
import logging
import torch.nn.functional as F
import asyncio
import gc
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from weakref import WeakValueDictionary

# Internal imports from utils
from src.data.processors.utils.system_utils import get_gpu_memory_usage, get_optimal_workers, create_thread_pool, cleanup_processor
from src.data.processors.utils.image_utils import load_and_validate_image, resize_image, get_image_stats
from src.data.processors.utils.image.vae_encoder import VAEEncoder
from src.config.config import ImageProcessorConfig  # Import the consolidated config

# Internal imports from processors
from src.data.processors.bucket import BucketManager
from src.data.processors.utils.progress_utils import (
    create_progress_tracker,
    update_tracker,
    log_progress,
)
from src.data.processors.utils.batch_utils import get_gpu_memory_usage

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Main ImageProcessor class for loading, preprocessing,
    and optionally VAE-encoding images.

    Changes made to:
      - Remove excessive concurrency overhead
      - Combine steps where possible
      - Limit usage of torch.compile
      - Increase default load_batch_size for speed
      - Use fewer dynamic calls
    """

    def __init__(self, config, bucket_manager=None, vae=None):
        import torch
        import logging
        from torchvision import transforms
        from src.data.processors.utils.image.vae_encoder import VAEEncoder
        from src.data.processors.utils.system_utils import get_optimal_workers, create_thread_pool

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.bucket_manager = bucket_manager

        # Some basic Torch optimizations
        VAEEncoder.enable_torch_backend_optimizations(self.config)

        # Optional VAE wrapper
        if vae is not None:
            self.vae_encoder = VAEEncoder(vae, config)
        else:
            self.vae_encoder = None

        # Basic transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        ])

        # Simple resize
        self.resize_op = transforms.Resize(
            config.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        # Some config flags
        self.use_pinned_memory = getattr(config, "use_pinned_memory", False)
        self.device = self.config.device
        self.batch_size = getattr(self.config, "batch_size", 64)  # Larger default
        self.load_batch_size = getattr(self.config, "load_batch_size", 256)  # Larger for speed

        # Thread pool for CPU I/O
        cpu_workers = min(get_optimal_workers(), 8)
        self.thread_pool = create_thread_pool(cpu_workers)
        self.logger.info(f"ImageProcessor using {cpu_workers} CPU workers.")

    def preprocess(self, img):
        """
        Synchronously preprocess: convert to tensor & normalize,
        then move to GPU if needed.
        """
        import torch
        # Resize on CPU first
        img = self.resize_op(img)

        # To tensor & normalize
        tensor = self.transform(img)

        # Optionally pin
        if self.use_pinned_memory:
            tensor = tensor.pin_memory()

        # Move to GPU in half precision
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    async def process_batch(self, images, skip_vae=False, keep_on_gpu=False):
        """
        Load & preprocess images in bigger sub-batches for speed.
        Optionally encode with VAE if skip_vae=False.
        Returns stacked or encoded latents.
        """
        import asyncio
        import torch

        all_tensors = []
        for start in range(0, len(images), self.load_batch_size):
            chunk = images[start : start + self.load_batch_size]
            # Preprocess in parallel
            futures = [self.thread_pool.submit(self._load_and_preprocess, img) for img in chunk]
            results = await asyncio.get_event_loop().run_in_executor(None, lambda: [f.result() for f in futures])
            # Stack
            batch_tensor = torch.stack(results, dim=0)
            all_tensors.append(batch_tensor)

        if not all_tensors:
            return None

        # Combine all into one
        big_tensor = torch.cat(all_tensors, dim=0)
        self.logger.info(f"Collected {big_tensor.size(0)} images in memory.")

        # Optional: VAE encode
        if not skip_vae and self.vae_encoder is not None:
            latents = await self.vae_encoder.encode_images(big_tensor, keep_on_gpu=keep_on_gpu)
            return latents

        # If not encoding, optionally move to CPU
        if not keep_on_gpu:
            big_tensor = big_tensor.cpu()
        return big_tensor

    def _load_and_preprocess(self, img_or_path):
        """
        Inline utility that loads from disk if path, then calls self.preprocess.
        """
        from PIL import Image
        import torch
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            try:
                img_obj = Image.open(img_or_path).convert('RGB')
            except Exception as e:
                self.logger.error(f"Failed to load image {img_or_path}: {e}")
                return torch.zeros(3, *self.config.resolution)
        else:
            # Assume it's already a PIL Image
            img_obj = img_or_path
        return self.preprocess(img_obj)

    async def process_image(self, img_or_path, skip_vae=False, keep_on_gpu=False):
        """
        Single-image version. Usually slower for big datasets.
        Batch calls are recommended for better throughput.
        """
        res = await self.process_batch([img_or_path], skip_vae=skip_vae, keep_on_gpu=keep_on_gpu)
        if res is None or res.shape[0] == 0:
            return {}
        # Return the first
        # If VAE encoded, shape is [1,4,H/8,W/8]. If raw, shape is [1,3,H,W].
        return res[0].unsqueeze(0)

    async def cleanup(self):
        """
        Cleanup any resources. For large dataset usage, you can keep this minimal.
        """
        import torch
        import gc
        self.logger.info("Cleaning up ImageProcessor.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if self.vae_encoder is not None:
            await self.vae_encoder.cleanup()

    def __del__(self):
        import asyncio, logging
        logger = logging.getLogger(__name__)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"ImageProcessor del error: {e}")
