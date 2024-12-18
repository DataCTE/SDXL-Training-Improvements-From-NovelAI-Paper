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

# Internal imports from utils
from src.data.processors.utils.system_utils import get_gpu_memory_usage, get_optimal_workers, create_thread_pool, cleanup_processor
from src.data.processors.utils.image_utils import load_and_validate_image, resize_image, get_image_stats
from src.data.processors.utils.image.vae_encoder import VAEEncoder
from src.config.config import ImageProcessorConfig  # Import the consolidated config

# Internal imports from processors
from src.data.processors.bucket import BucketManager

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

        # Initialize VAE encoder if provided
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
        
        self.logger.info(
            f"Initialized ImageProcessor:\n"
            f"- Resolution: {config.resolution}\n"
            f"- Center crop: {config.center_crop}\n"
            f"- Random flip: {config.random_flip}\n"
            f"- VAE: {'Yes' if self.vae_encoder else 'No'}"
        )

    async def load_image(self, path_or_str) -> "Image.Image":
        """
        Load and validate an image using a utility function, in a separate thread if needed.
        """
        # Pass the config to load_and_validate_image
        return await asyncio.to_thread(load_and_validate_image, path_or_str, config=self.config)

    async def process_image(
        self,
        image: Union[str, Path, "Image.Image"],
        target_size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        keep_on_gpu: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single image with optional cropping, flipping, normalization, and VAE encoding.
        """
        try:
            if isinstance(image, (str, Path)):
                # Fix: await the load_image directly instead of using asyncio.to_thread
                image = await self.load_image(image)
                if image is None:
                    raise ValueError("Failed to load image")

            # Convert to RGB
            image = image.convert("RGB")

            # Determine target size if using bucket manager
            if not target_size and self.bucket_manager:
                target_size = self.bucket_manager.get_target_size(image.size)
            else:
                target_size = target_size or self.config.resolution

            # Crop mode
            if self.config.crop_mode == "center":
                image = self.center_crop(image)
            elif self.config.crop_mode == "random":
                y1, x1, h, w = self.random_crop.get_params(image, (self.config.resolution[0], self.config.resolution[1]))
                image = transforms.functional.crop(image, y1, x1, h, w)
            # if "none", skip cropping

            # Random flip
            if self.config.random_flip and random.random() < 0.5:
                image = transforms.functional.hflip(image)

            # Convert to tensor and normalize
            image_tensor = self.transform(image).to(self.config.device, non_blocking=True)

            result = {
                "pixel_values": image_tensor.cpu(),  # or keep on GPU if you wish
                "original_size": original_size or (image.height, image.width),
                "target_size": target_size,
            }

            # Encode via VAE if available
            if self.vae_encoder is not None:
                latents = await self.vae_encoder.encode_images(
                    image_tensor,
                    keep_on_gpu=(keep_on_gpu or self.always_on_gpu)
                )
                if latents is not None and not (keep_on_gpu or self.always_on_gpu):
                    latents = latents.to("cpu", non_blocking=True)
                result["latents"] = latents

            return result

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

    @torch.inference_mode()
    async def process_batch(
        self,
        images: List["Image.Image"],
        width: int,
        height: int,
        keep_on_gpu: bool = False
    ) -> List[torch.Tensor]:
        """
        Process a batch of images with sub-batching, optional VAE encoding, and inference_mode.
        Demonstrates dynamic approach for sub-batch sizing if desired.
        """
        processed_tensors = []

        batch_size = len(images)
        if batch_size == 0:
            return processed_tensors

        try:
            # Basic sub-batching example (could be made adaptive)
            sub_batch_size = getattr(self.config, "vae_batch_size", 8)

            for i in range(0, batch_size, sub_batch_size):
                sub_batch = images[i : i + sub_batch_size]

                # Preprocess in parallel threads
                preprocessing_tasks = [
                    asyncio.to_thread(self.preprocess, img, width, height)
                    for img in sub_batch
                ]

                sub_processed = await asyncio.gather(*preprocessing_tasks)
                sub_processed = [t for t in sub_processed if t is not None]

                if sub_processed:
                    # Combine into a single tensor
                    batch_tensor = torch.stack(sub_processed).to(self.config.device, non_blocking=True)
                    del sub_processed

                    # Encode with VAE if available
                    if self.vae_encoder is not None:
                        try:
                            encoded = await self.vae_encoder.encode_images(
                                batch_tensor,
                                keep_on_gpu=(keep_on_gpu or self.always_on_gpu)
                            )
                            if encoded is not None:
                                if not (keep_on_gpu or self.always_on_gpu):
                                    encoded = encoded.to("cpu", non_blocking=True)
                                # Split back into single images
                                if encoded.dim() == 3:
                                    processed_tensors.append(encoded)
                                else:
                                    for single_latent in encoded:
                                        processed_tensors.append(single_latent)
                            else:
                                # Fallback: fill with zeros if there's an error
                                for _ in range(batch_tensor.shape[0]):
                                    processed_tensors.append(
                                        torch.zeros(
                                            (4, height // 8, width // 8),
                                            dtype=self.config.dtype,
                                            device="cpu"
                                        )
                                    )
                            del encoded
                        except Exception as e:
                            self.logger.error(f"VAE encoding error: {str(e)[:200]}...")
                            # Fallback: fill with zeros
                            for _ in range(batch_tensor.shape[0]):
                                processed_tensors.append(
                                    torch.zeros(
                                        (4, height // 8, width // 8),
                                        dtype=self.config.dtype,
                                        device="cpu"
                                    )
                                )
                    else:
                        # No VAE => move preprocessed tensors to CPU
                        for tensor in batch_tensor:
                            processed_tensors.append(tensor.cpu())

                    del batch_tensor

            return processed_tensors

        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)[:200]}...")
            if 'sub_processed' in locals():
                del sub_processed
            if 'batch_tensor' in locals():
                del batch_tensor
            torch.cuda.empty_cache()
            raise

    def preprocess(self, img: "Image.Image", width: int, height: int) -> torch.Tensor:
        """
        Preprocess a single image with resizing and normalization.
        """
        img = self._resize_image(img, width, height)

        # Use pinned memory if configured
        if self.use_pinned_memory:
            img_tensor = self.transform(img).pin_memory()
        else:
            img_tensor = self.transform(img)

        # Move to device in half precision if desired
        with torch.cuda.amp.autocast(dtype=self.config.dtype):
            img_tensor = img_tensor.to(self.config.device, non_blocking=True)

        return img_tensor

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

    async def cleanup(self):
        await cleanup_processor(self)

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
            # Get max dimensions from buckets if available
            if self.bucket_manager and self.bucket_manager.buckets:
                max_dims = max(
                    (bucket.width * bucket.height) 
                    for bucket in self.bucket_manager.buckets.values()
                )
            else:
                # Fallback to config resolution
                max_dims = self.config.resolution[0] * self.config.resolution[1]

            # Calculate memory per image (3 channels, float32)
            bytes_per_image = max_dims * 3 * 4

            # Get available memory (70% of free memory)
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(self.config.device).total_memory
                free_memory = int(free_memory * 0.7)  # Leave 30% headroom
            else:
                import psutil
                free_memory = psutil.virtual_memory().available
                free_memory = int(free_memory * 0.7)

            # Calculate max images that fit in buffer
            buffer_size = max(1, free_memory // bytes_per_image)

            # Cap at reasonable maximum
            return min(buffer_size, 2048)

        except Exception as e:
            self.logger.warning(f"Error calculating buffer size: {e}, using default")
            return 512

    def _allocate_tensor_buffer(self) -> Optional[torch.Tensor]:
        """Allocate tensor buffer for batch processing with error handling."""
        try:
            if not self.buffer_size:
                return None

            # Get dimensions from buckets or config
            if self.bucket_manager and self.bucket_manager.buckets:
                max_h = max(b.height for b in self.bucket_manager.buckets.values())
                max_w = max(b.width for b in self.bucket_manager.buckets.values())
            else:
                max_h, max_w = self.config.resolution

            # Allocate buffer with pinned memory if configured
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