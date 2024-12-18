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
    Advanced ImageProcessor allowing torch.compile on preprocess,
    pinned-memory usage, bigger sub-batches, and minimized concurrency overhead.
    """

    def __init__(self, config: ImageProcessorConfig, bucket_manager=None, vae=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.bucket_manager = bucket_manager

        # Basic Torch optimizations
        VAEEncoder.enable_torch_backend_optimizations(self.config)

        # Optional VAE
        self.vae_encoder = None
        if vae is not None:
            self.vae_encoder = VAEEncoder(vae, config)

        # Build transforms
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
        self.transform = transforms.Compose(base_transforms)

        self.resize_op = transforms.Resize(
            config.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        # Possibly compile the transform if shapes are stable
        if getattr(self.config, "use_torch_compile_preprocess", False) and hasattr(torch, "compile"):
            self.transform = torch.compile(self.transform, mode="default")

        self.use_pinned_memory = getattr(config, "use_pinned_memory", True)
        self.device = self.config.device
        self.batch_size = getattr(self.config, "batch_size", 128)
        self.load_batch_size = getattr(self.config, "load_batch_size", 256)

        # CPU worker thread pool - be careful not to overdo concurrency
        from src.data.processors.utils.system_utils import get_optimal_workers, create_thread_pool
        cpu_workers = min(get_optimal_workers(), 8)
        self.thread_pool = create_thread_pool(cpu_workers)
        self.logger.info(f"ImageProcessor using {cpu_workers} CPU threads.")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Synchronously preprocess (resize -> to tensor -> normalize -> optional pin).
        Then move to GPU in half precision if available.
        """
        # CPU resizing
        img = self.resize_op(img)
        tensor = self.transform(img)

        if self.use_pinned_memory:
            tensor = tensor.pin_memory()

        # Move to GPU half precision
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    async def process_batch(self, images: List[Union[str, Path, Image.Image]], skip_vae=False, keep_on_gpu=False):
        """
        Batches image loading & preprocessing, optionally runs VAE encoding.
        """
        if not images:
            return None

        # We gather in bigger sub-batches
        all_tensors = []
        for start in range(0, len(images), self.load_batch_size):
            chunk = images[start : start + self.load_batch_size]

            # Load + preprocess in synchronous Python threadpool
            futures = [self.thread_pool.submit(self._load_and_preprocess, img) for img in chunk]
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [f.result() for f in futures]
            )
            batch_tensor = torch.stack(results, dim=0)
            all_tensors.append(batch_tensor)

        if not all_tensors:
            return None

        big_tensor = torch.cat(all_tensors, dim=0)

        # Debug rather than info
        self.logger.debug(f"Process_batch: collected {big_tensor.shape[0]} images total.")

        # VAE encode if requested
        if not skip_vae and self.vae_encoder:
            latents = await self.vae_encoder.encode_images(big_tensor, keep_on_gpu=keep_on_gpu)
            return latents

        # Otherwise optionally move to CPU
        if not keep_on_gpu:
            big_tensor = big_tensor.cpu()
        return big_tensor

    def _load_and_preprocess(self, img_or_path):
        """
        Internal method. Load from disk (if string/Path).
        """
        if isinstance(img_or_path, (str, Path)):
            try:
                pil_img = Image.open(img_or_path).convert("RGB")
            except Exception as e:
                self.logger.error(f"Failed to load image {img_or_path}: {e}")
                return torch.zeros(3, *self.config.resolution)
        else:
            pil_img = img_or_path
        return self.preprocess(pil_img)

    async def process_image(self, image, original_size=None, skip_vae=False, keep_on_gpu=False):
        """
        Single-image version returning a dict with pixel_values or latents plus original_size.
        """
        batch_result = await self.process_batch([image], skip_vae=skip_vae, keep_on_gpu=keep_on_gpu)
        if batch_result is None or batch_result.shape[0] == 0:
            return {}

        # shape = [1, c, h, w]
        out_tensor = batch_result[0].unsqueeze(0)
        if skip_vae:
            return {
                "pixel_values": out_tensor,
                "latents": None,
                "original_size": original_size
            }
        else:
            return {
                "pixel_values": None,
                "latents": out_tensor,
                "original_size": original_size
            }

    async def cleanup(self):
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if self.vae_encoder:
            await self.vae_encoder.cleanup()
        self.logger.info("ImageProcessor cleanup finished.")

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"ImageProcessor destructor error: {e}")
