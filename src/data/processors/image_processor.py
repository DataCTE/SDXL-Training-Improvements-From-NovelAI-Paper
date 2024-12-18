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
    def __init__(
        self,
        config: ImageProcessorConfig,
        bucket_manager: Optional[BucketManager] = None,
        vae = None
    ):
        """Initialize image processor with VAE and bucket manager."""
        self.config = config
        self.bucket_manager = bucket_manager
        
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
        
        # Optional config attribute to retain everything on GPU
        self.always_on_gpu = getattr(config, "always_on_gpu", False)
        self.use_pinned_memory = getattr(config, "use_pinned_memory", False)
        
        logger.info(
            f"Initialized ImageProcessor:\n"
            f"- Resolution: {config.resolution}\n"
            f"- Center crop: {config.center_crop}\n"
            f"- Random flip: {config.random_flip}\n"
            f"- VAE: {'Yes' if self.vae_encoder else 'No'}"
        )

    async def process_image(
        self,
        image: Union[str, Path, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        keep_on_gpu: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single image with SDXL-style augmentation."""
        try:
            if isinstance(image, (str, Path)):
                # Use async thread for loading
                image = await asyncio.to_thread(self.load_image, image)
                if image is None:
                    raise ValueError("Failed to load image")

            # Convert to RGB
            image = image.convert("RGB")

            # Determine target size
            if not target_size and self.bucket_manager:
                target_size = self.bucket_manager.get_target_size(image.size)
            else:
                target_size = target_size or self.config.resolution

            # Crop mode selection
            if self.config.crop_mode == "center":
                image = self.center_crop(image)
            elif self.config.crop_mode == "random":
                # random_crop usage
                y1, x1, h, w = self.random_crop.get_params(
                    image, (self.config.resolution[0], self.config.resolution[1])
                )
                image = transforms.functional.crop(image, y1, x1, h, w)
            # If "none", skip cropping entirely

            # Random flip
            if self.config.random_flip and random.random() < 0.5:
                image = transforms.functional.hflip(image)

            # Convert to tensor and normalize
            image_tensor = self.transform(image).to(self.config.device, non_blocking=True)

            result = {
                "pixel_values": image_tensor.cpu(),  # Or keep on GPU if needed
                "original_size": original_size or (image.height, image.width),
                "target_size": target_size,
            }

            if self.vae_encoder is not None:
                latents = await self.vae_encoder.encode_images(
                    image_tensor, 
                    keep_on_gpu=(keep_on_gpu or self.always_on_gpu)
                )
                if latents is not None and not (keep_on_gpu or self.always_on_gpu):
                    latents = latents.cpu()
                result["latents"] = latents

            return result

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def load_image(self, path: str) -> Optional[Image.Image]:
        """Load and validate an image using utility function."""
        return load_and_validate_image(
            path,
            config=self.config,
            required_modes=('RGB', 'RGBA')
        )

    def _resize_image(self, img: Image.Image, width: int, height: int) -> Image.Image:
        """Resize image using utility function."""
        return resize_image(img, (width, height), Image.Resampling.LANCZOS)

    def get_image_stats(self, img: Image.Image) -> Dict:
        """Get image statistics using utility function."""
        return get_image_stats(img)

    def _build_transform(self):
        """Build optimized transform pipeline."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3]),  # Ensure 3 channels
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std),
            transforms.Lambda(lambda x: x.to(self.config.dtype))
        ])

    def _process_single_image(self, img: Image.Image, width: int, height: int) -> torch.Tensor:
        """Process a single image with GPU acceleration."""
        img = self._resize_image(img, width, height)
        
        # Transform directly to GPU with mixed precision
        with torch.cuda.amp.autocast(dtype=self.config.dtype):
            img_tensor = self.transform(img).to(self.config.device, non_blocking=True)
        
        return img_tensor

    def _adjust_buffer_size(self, batch_size: int, width: int, height: int):
        """Adjust tensor buffer size if needed."""
        if (batch_size > self.buffer_size[0] or 
            width > self.buffer_size[2] or 
            height > self.buffer_size[3]):
            
            self.buffer_size = (
                max(batch_size, self.buffer_size[0]), 
                3,
                max(width, self.buffer_size[2]), 
                max(height, self.buffer_size[3])
            )
            
            # Reallocate buffer with new size
            self.tensor_buffer = torch.empty(
                self.buffer_size,
                dtype=self.config.dtype,
                device=self.config.device
            )
            logger.debug(f"Resized tensor buffer to {self.buffer_size}")

    @torch.inference_mode()
    async def process_batch(
        self,
        images: List[Image.Image],
        width: int,
        height: int,
        keep_on_gpu: bool = False
    ) -> List[torch.Tensor]:
        """
        Process a batch of images with optimized speed and memory management.
        Uses inference_mode for reduced autograd overhead and a dynamic approach
        for sub_batch_size selection.
        """
        batch_size = len(images)
        if batch_size == 0:
            return []
        
        processed_tensors = []
        
        try:
            # Check total GPU memory and current usage to compute "available" memory
            total_memory = torch.cuda.get_device_properties(self.config.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.config.device)
            free_memory = max(0, total_memory - allocated_memory)

            # Estimate memory usage per image (width * height * channels * bytes per element)
            # For half-precision (2 bytes/element) with 4 channels, etc.
            bytes_per_element = 2 if self.config.dtype in [torch.float16, torch.bfloat16] else 4
            memory_per_image = width * height * 4 * bytes_per_element

            # Use a dynamic sub-batch size. You can incorporate config constraints as well.
            # For instance, set an upper bound from config if needed: 
            # configured_max = getattr(self.config, 'vae_batch_size', 64)
            # sub_batch_size = min(
            #     max(1, int(free_memory * 0.4 // memory_per_image)),
            #     configured_max,
            #     batch_size
            # )
            
            # In this snippet, we preserve the logic from the original code
            # and just refine it a bit:
            sub_batch_size = min(
                max(8, int(free_memory * 0.4 / memory_per_image)),
                128,
                batch_size
            )

            # Process the images in sub-batches
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
                    # Stack into a single tensor on device
                    batch_tensor = torch.stack(sub_processed).to(
                        self.config.device, non_blocking=True
                    )
                    del sub_processed
                    
                    # Optionally encode via VAE
                    if self.vae_encoder is not None:
                        try:
                            encoded = await self.vae_encoder.encode_images(
                                batch_tensor,
                                keep_on_gpu=(keep_on_gpu or self.always_on_gpu)
                            )
                            if encoded is not None:
                                if not (keep_on_gpu or self.always_on_gpu):
                                    encoded = encoded.cpu()
                                # Split back into single images
                                if encoded.dim() == 3:
                                    processed_tensors.append(encoded)
                                else:
                                    for single_latent in encoded:
                                        processed_tensors.append(single_latent)
                            else:
                                # Fallback: fill with zeros
                                for _ in range(batch_tensor.shape[0]):
                                    processed_tensors.append(
                                        torch.zeros(
                                            (4, height // 8, width // 8),
                                            dtype=self.config.dtype,
                                            device='cpu'
                                        )
                                    )
                            del encoded
                        except Exception as e:
                            logger.error(f"VAE encoding error: {str(e)[:200]}...")
                            # On error, fill with zeros
                            for _ in range(batch_tensor.shape[0]):
                                processed_tensors.append(
                                    torch.zeros(
                                        (4, height // 8, width // 8),
                                        dtype=self.config.dtype,
                                        device='cpu'
                                    )
                                )
                    else:
                        # No VAE, just move the processed tensors to CPU
                        for tensor in batch_tensor:
                            processed_tensors.append(tensor.cpu())
                    
                    del batch_tensor

            return processed_tensors
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)[:200]}...")
            if 'sub_processed' in locals():
                del sub_processed
            if 'batch_tensor' in locals():
                del batch_tensor
            torch.cuda.empty_cache()
            raise

    @torch.inference_mode()
    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Optimized VAE encoding with better memory management, using inference_mode.
        This code is otherwise similar to the original, but we've replaced no_grad
        with inference_mode for performance gains.
        """
        if self.config.enable_memory_efficient_attention:
            vae.enable_xformers_memory_efficient_attention()
        
        if self.config.enable_vae_slicing:
            vae.enable_slicing()
        
        batch_size = pixel_values.shape[0]
        
        # Create CUDA streams for overlapped operations
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        
        try:
            with torch.cuda.stream(compute_stream):
                # Example sub-batch size approach
                sub_batch_size = min(8, batch_size)
                latents_list = []
                
                for i in range(0, batch_size, sub_batch_size):
                    sub_batch = pixel_values[i:i+sub_batch_size]
                    
                    with torch.cuda.amp.autocast(dtype=self.config.dtype):
                        latents = vae.encode(sub_batch).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        
                        with torch.cuda.stream(transfer_stream):
                            latents_list.append(latents)
                    
                    # Optional cleanup for very large batches
                    if i % (sub_batch_size * 4) == 0 and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(self.config.device).total_memory * 0.8:
                        compute_stream.synchronize()
                        torch.cuda.empty_cache()
                
                result = torch.cat(latents_list, dim=0)
                del latents_list
                return result
                
        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)[:200]}...")
            torch.cuda.empty_cache()
            raise

    def _get_optimal_size(self, width: int, height: int) -> Tuple[int, int]:
        """Get optimal size considering bucket constraints if available."""
        if self.bucket_manager is not None:
            bucket = self.bucket_manager.find_bucket(width, height)
            if bucket is not None:
                logger.debug(f"Using bucket {bucket} for image {width}x{height}")
                return bucket.width, bucket.height
            
            # If no bucket found, try to find closest valid bucket
            aspect_ratio = width / height
            target_resolution = width * height
            
            best_bucket = None
            min_diff = float('inf')
            
            for bucket in self.bucket_manager.buckets.values():
                if abs(bucket.aspect_ratio - aspect_ratio) <= self.bucket_manager.bucket_tolerance:
                    res_diff = abs(bucket.resolution - target_resolution)
                    if res_diff < min_diff:
                        min_diff = res_diff
                        best_bucket = bucket
            
            if best_bucket is not None:
                logger.debug(f"Using closest bucket {best_bucket} for image {width}x{height}")
                return best_bucket.width, best_bucket.height
        
        # Default sizing logic if no bucket found
        return self._default_resize(width, height)

    def preprocess(self, img: Image.Image, width: int, height: int) -> torch.Tensor:
        """Preprocess a single image with resizing and normalization.
        
        Args:
            img: PIL Image to process
            width: Target width
            height: Target height
            
        Returns:
            Preprocessed image tensor
        """
        # Potential GPU or faster library-based resizing
        # or keep PIL-SIMD on CPU.
        img = self._resize_image(img, width, height)
        
        # Use pinned memory if configured
        if self.use_pinned_memory:
            img_tensor = self.transform(img).pin_memory()
        else:
            img_tensor = self.transform(img)
        
        # Move to device in half precision.
        with torch.cuda.amp.autocast(dtype=self.config.dtype):
            img_tensor = img_tensor.to(self.config.device, non_blocking=True)
            
        return img_tensor

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
            logger.error(f"Error during image processor deletion: {e}")

    def recalc_buffer_size(self) -> None:
        """
        Recalculate buffer dimensions based on newly available buckets,
        then re-allocate the tensor buffer.
        """
        self.buffer_size = self._calculate_initial_buffer_size()
        self.tensor_buffer = self._allocate_tensor_buffer()
        logger.info(f"Recalculated buffer size: {self.buffer_size}")