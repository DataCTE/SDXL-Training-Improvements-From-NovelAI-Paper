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
    Main ImageProcessor class that handles loading, preprocessing,
    and optional VAE encoding of images. Incorporates techniques
    for performance optimization and memory efficiency, including
    pinned memory, inference_mode, and dynamic sub-batching.
    """

    def __init__(
        self,
        config: ImageProcessorConfig,
        bucket_manager: Optional[BucketManager] = None,
        vae=None
    ):
        """Initialize image processor with VAE and bucket manager."""
        self.config = config
        self.bucket_manager = bucket_manager
        self.logger = logging.getLogger(__name__)

        # Apply shared Torch optimizations from VAEEncoder
        VAEEncoder.enable_torch_backend_optimizations(self.config)

        self.vae_encoder = None
        if vae is not None:
            self.vae_encoder = VAEEncoder(vae=vae, config=config)

        # Build transform pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        ])

        # Initialize resize transforms
        self.resize = transforms.Resize(
            config.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        self.center_crop = transforms.CenterCrop(config.center_crop)
        self.random_crop = transforms.RandomCrop(config.resolution)

        # Whether to keep data on GPU
        self.always_on_gpu = getattr(config, "always_on_gpu", False)
        self.use_pinned_memory = getattr(config, "use_pinned_memory", False)

        # Add processing pools and queues
        self.num_workers = get_optimal_workers()
        self.thread_pool = create_thread_pool(self.num_workers)
        self.preprocessing_queue = asyncio.Queue(maxsize=32)

        # Compile the resize transform if possible
        if isinstance(self.resize, transforms.Resize) and hasattr(torch, 'compile'):
            self.resize = torch.compile(self.resize)

        # Setup fast transforms
        self._setup_fast_transforms()

        # Initialize hardware-optimized transforms
        self._init_hardware_optimized_transforms()

        # Setup parallel processing
        self._setup_parallel_processing()

        # Initialize memory pools
        self._init_memory_pools()

        self.logger.info(
            f"Initialized ImageProcessor:\n"
            f"- Resolution: {config.resolution}\n"
            f"- Center crop: {config.center_crop}\n"
            f"- Random flip: {config.random_flip}\n"
            f"- VAE: {'Yes' if self.vae_encoder else 'No'}"
        )

    def _setup_fast_transforms(self):
        """Setup optimized transform pipeline."""
        def fast_transform(img):
            # Convert to tensor efficiently
            if not isinstance(img, torch.Tensor):
                img = transforms.functional.to_tensor(img)

            # Apply direct normalization using tensor ops
            if (
                hasattr(self.config, 'normalize_mean') and
                hasattr(self.config, 'normalize_std')
            ):
                mean = torch.tensor(self.config.normalize_mean, device=img.device)[None, :, None, None]
                std = torch.tensor(self.config.normalize_std, device=img.device)[None, :, None, None]
                img = (img - mean) / std

            return img

        self.fast_transform = fast_transform

    def _setup_parallel_processing(self):
        """Setup parallel processing pools and queues."""
        num_workers = min(multiprocessing.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="img_proc_"
        )
        self.processing_queue = asyncio.Queue(maxsize=32)
        self.batch_size = getattr(self.config, 'batch_size', 1)

        logger.info(f"Initialized parallel processing with {num_workers} workers")

    def _init_memory_pools(self):
        """Initialize memory pools for efficient tensor allocation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.tensor_cache = WeakValueDictionary()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats(device=self.config.device)
                torch.cuda.reset_peak_memory_stats()
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except Exception as e:
                self.logger.warning(f"Unable to set per_process_memory_fraction: {e}")
        else:
            self.tensor_cache = WeakValueDictionary()

    def _init_hardware_optimized_transforms(self):
        """Initialize hardware-optimized transforms."""
        if torch.cuda.is_available():
            try:
                import cupy as cp
                self.use_gpu_transforms = True
            except ImportError:
                self.logger.info("CuPy not available, falling back to PyTorch transforms")
                self.use_gpu_transforms = False

            self.gpu_transform = self._create_gpu_transform_pipeline()
        else:
            self.use_gpu_transforms = False

    def _create_gpu_transform_pipeline(self):
        """Create GPU-accelerated transform pipeline."""
        @torch.compile
        def gpu_transform(x: torch.Tensor):
            x = self._gpu_normalize(x)
            x = self._gpu_resize(x)  # <--- We now call a properly defined method below
            return x

        return gpu_transform

    @torch.compile(fullgraph=True)
    def _gpu_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hardware-optimized normalization (optional).
        Adjust or extend logic if you want actual GPU-based mean/std normalization.
        For now, this is a placeholder or no-op.
        """
        return x

    @torch.compile(fullgraph=True)
    def _gpu_resize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dedicated GPU-based resizing using PyTorch F.interpolate.
        Resizes to self.config.resolution.
        If x is [C, H, W], we add a batch dimension (N=1) before interpolation 
        and then remove it afterwards, so that F.interpolate can operate on 4D input.
        """
        # If x is 3D: (C, H, W), add an extra batch dimension -> (1, C, H, W)
        added_batch_dim = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch_dim = True

        # Interpolate now that x is guaranteed to be [N, C, H, W].
        x = F.interpolate(
            x,
            size=(self.config.resolution[0], self.config.resolution[1]),
            mode='bilinear',
            align_corners=False
        )
        
        # If we added a batch dimension, remove it again.
        if added_batch_dim:
            x = x.squeeze(0)

        return x

    async def load_image(self, path_or_str) -> "Image.Image":
        """
        Load and validate an image using a utility function, in a separate thread if needed.
        """
        return await asyncio.to_thread(load_and_validate_image, path_or_str, config=self.config)

    def _resize_image(self, img: "Image.Image", width: int, height: int) -> "Image.Image":
        """
        Basic wrapper for resizing logic. Could be replaced with fast GPU-based ops.
        """
        import torchvision.transforms.functional as TF
        img = TF.resize(
            img,
            (height, width),
            interpolation=self.resize.interpolation,
            antialias=self.resize.antialias
        )
        return img

    def preprocess(self, img: "Image.Image", width: int, height: int) -> torch.Tensor:
        """
        Preprocess a single image with resizing and normalization.
        """
        img = self._resize_image(img, width, height)

        if self.use_pinned_memory:
            img_tensor = self.transform(img).pin_memory()
        else:
            img_tensor = self.transform(img)

        # Move to device in half precision if desired
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=self.config.dtype):
            img_tensor = img_tensor.to(self.config.device, non_blocking=True)

        return img_tensor

    @torch.inference_mode()
    async def process_batch(
        self,
        images: List["Image.Image"],
        width: int,
        height: int,
        keep_on_gpu: bool = False
    ) -> List[torch.Tensor]:
        """
        Optimized batch processing with parallel preprocessing and optional VAE encoding.
        """
        try:
            batch_size = len(images)
            if batch_size == 0:
                return []

            # Process images in parallel using thread pool
            preprocessing_tasks = [
                self.thread_pool.submit(self.preprocess, img, width, height)
                for img in images
            ]

            # Gather results maintaining order
            processed = []
            for future in preprocessing_tasks:
                try:
                    tensor = await asyncio.wrap_future(future)
                    if tensor is not None:
                        processed.append(tensor)
                except Exception as e:
                    self.logger.error(f"Preprocessing error: {e}")

            if not processed:
                return []

            # Stack tensors efficiently
            batch_tensor = torch.stack(processed, dim=0)
            del processed

            # Process through VAE if available
            if self.vae_encoder is not None:
                try:
                    encoded = await self.vae_encoder.encode_images(
                        batch_tensor,
                        keep_on_gpu=keep_on_gpu
                    )
                    del batch_tensor
                    return encoded
                except Exception as e:
                    self.logger.error(f"VAE encoding error: {e}")
                    return None

            return batch_tensor

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            if 'batch_tensor' in locals():
                del batch_tensor
            torch.cuda.empty_cache()
            return None

    async def process_image(self, image: Union[str, Path, "Image.Image"], skip_vae: bool = False, **kwargs):
        """
        Overload process_image for additional GPU transforms and progress tracking.
        Now also supports skipping the VAE encoding step if latents are already cached.
        
        Parameters:
            image: The image path or PIL image to be processed.
            skip_vae: If True, do not run the VAE encoder step.
        """
        try:
            # 1) Load and preprocess the image into a tensor
            image_data = await self._parallel_load_and_preprocess(image)

            # 2) Apply GPU transforms if available, else CPU transforms
            if self.use_gpu_transforms:
                image_tensor = self._gpu_process_image(image_data)
            else:
                image_tensor = self._cpu_process_image(image_data)

            if kwargs.get('keep_on_gpu', False):
                image_tensor = image_tensor.to(
                    self.config.device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )

            # 3) If skip_vae==False and a VAE is available, encode to latents.
            latents = None
            if not skip_vae and self.vae_encoder is not None:
                encoded_list = await self.vae_encoder.encode_images(
                    image_tensor.unsqueeze(0),  # [1, C, H, W]
                    keep_on_gpu=kwargs.get('keep_on_gpu', False)
                )
                if encoded_list and len(encoded_list) > 0:
                    latents = encoded_list[0]

            # 4) Build the final dict. If skip_vae is True, latents remain None here.
            return {
                "pixel_values": image_tensor,
                "latents": latents,
                "width": image_tensor.shape[-1],
                "height": image_tensor.shape[-2],
                "original_size": kwargs.get("original_size", (image_tensor.shape[-2], image_tensor.shape[-1]))
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    async def _parallel_load_and_preprocess(self, image):
        async with asyncio.TaskGroup() as group:
            load_task = group.create_task(self._async_load(image))
            prep_task = group.create_task(self._async_preprocess(await load_task))
        return await prep_task

    @torch.compile
    def _gpu_process_image(self, image_data: torch.Tensor):
        """
        Example GPU transform pipeline (normalization + resizing in this example).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            return self.gpu_transform(image_data)

    def _cpu_process_image(self, image_data: torch.Tensor):
        """
        Basic CPU-based transform pipeline placeholder.
        If GPU transforms are disabled, we land here.
        """
        return image_data

    async def _async_load(self, image_path):
        """Asynchronously load an image."""
        try:
            if isinstance(image_path, (str, Path)):
                return Image.open(image_path).convert('RGB')
            return image_path
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise

    async def _async_preprocess(self, image):
        """Asynchronously preprocess an image."""
        try:
            loop = asyncio.get_event_loop()
            if self.thread_pool:
                return await loop.run_in_executor(
                    self.thread_pool,
                    self.preprocess,
                    image,
                    self.config.resolution[0],
                    self.config.resolution[1]
                )
            return self.preprocess(
                image,
                self.config.resolution[0],
                self.config.resolution[1]
            )
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise

    async def process_batch(self, images: List["Image.Image"], **kwargs):
        """
        Process a batch of images with progress tracking and hardware optimizations.
        """
        try:
            batch_progress = create_progress_tracker(
                total_items=len(images),
                batch_size=self.batch_size,
                device=self.config.device,
                desc="Processing batch",
                unit="batch"
            )

            results = []
            for i, img in enumerate(images):
                result = await self.process_image(img, **kwargs)
                if result is not None:
                    results.append(result)

                update_tracker(
                    batch_progress,
                    processed=1,
                    failed=0 if result is not None else 1,
                    memory_gb=get_gpu_memory_usage(self.config.device)
                    if torch.cuda.is_available() else None
                )
                log_progress(batch_progress, prefix="Batch Processing: ")

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            if batch_progress:
                update_tracker(batch_progress, failed=1, error_type=str(type(e).__name__))
            raise

    def _create_result_dict(self, image_tensor, **kwargs):
        """
        Create the final output dictionary. Example usage for structuring results.
        """
        return {
            "pixel_values": image_tensor,
            "width": image_tensor.shape[-1],
            "height": image_tensor.shape[-2],
            **kwargs
        }

    async def cleanup(self):
        """
        Perform resource cleanup. Reliant on shared Torch cleanup in VAEEncoder
        plus any processor-specific logic.
        """
        try:
            VAEEncoder.perform_basic_cleanup()  # GPU cleanup
            await cleanup_processor(self)
        except Exception as e:
            self.logger.error(f"Error during image processor cleanup: {e}")

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            self.logger.error(f"Error during image processor deletion: {e}")

    def recalc_buffer_size(self) -> None:
        """
        Recalculate buffer dimensions based on newly available buckets,
        then re-allocate the tensor buffer.
        """
        self.buffer_size = self._calculate_initial_buffer_size()
        self.tensor_buffer = self._allocate_tensor_buffer()
        self.logger.info(f"Recalculated buffer size: {self.buffer_size}")

    def _calculate_initial_buffer_size(self) -> int:
        """Calculate optimal buffer size based on available memory and bucket sizes."""
        try:
            if self.bucket_manager and self.bucket_manager.buckets:
                max_dims = max(
                    (bucket.width * bucket.height)
                    for bucket in self.bucket_manager.buckets.values()
                )
            else:
                max_dims = self.config.resolution[0] * self.config.resolution[1]

            # Calculate memory per image (3 channels, float32)
            bytes_per_image = max_dims * 3 * 4

            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(self.config.device).total_memory
                free_memory = int(free_memory * 0.7)  # Leave 30% headroom
            else:
                import psutil
                free_memory = psutil.virtual_memory().available
                free_memory = int(free_memory * 0.7)

            buffer_size = max(1, free_memory // bytes_per_image)
            return min(buffer_size, 2048)

        except Exception as e:
            self.logger.warning(f"Error calculating buffer size: {e}, using default")
            return 512

    def _allocate_tensor_buffer(self) -> Optional[torch.Tensor]:
        """Allocate tensor buffer for batch processing with error handling."""
        try:
            if not self.buffer_size:
                return None

            if self.bucket_manager and self.bucket_manager.buckets:
                max_h = max(b.height for b in self.bucket_manager.buckets.values())
                max_w = max(b.width for b in self.bucket_manager.buckets.values())
            else:
                max_h, max_w = self.config.resolution

            buffer = torch.empty(
                (self.buffer_size, 3, max_h, max_w),
                dtype=self.config.dtype,
                device="cpu",
                pin_memory=self.use_pinned_memory
            )

            self.logger.info(
                f"Allocated tensor buffer: {buffer.shape}, "
                f"Memory: {buffer.element_size() * buffer.nelement() / 1024**2:.1f}MB"
            )
            return buffer

        except Exception as e:
            self.logger.error(f"Failed to allocate tensor buffer: {e}")
            return None
