# src/data/processors/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List, Optional, Dict
from torchvision import transforms
from dataclasses import dataclass
import numpy as np
import logging
import torch.nn.functional as F
import asyncio

# Internal imports from utils
from src.data.processors.utils.system_utils import get_gpu_memory_usage, get_optimal_workers, create_thread_pool
from src.data.processors.utils.batch_utils import adjust_batch_size
from src.data.processors.utils.image_utils import load_and_validate_image, resize_image, get_image_stats
from src.data.processors.utils.image.vae_encoder import VAEEncoder, VAEEncoderConfig

# Internal imports from processors
from src.data.processors.bucket import BucketManager

logger = logging.getLogger(__name__)

@dataclass
class ImageProcessorConfig:
    dtype: torch.dtype = torch.float16
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5)
    device: torch.device = torch.device('cuda')
    enable_memory_efficient_attention: bool = True
    enable_vae_slicing: bool = True
    vae_batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    max_memory_usage: float = 0.9
    max_image_size: Tuple[int, int] = (8192, 8192)
    min_image_size: Tuple[int, int] = (256, 256)

class ImageProcessor:
    def __init__(
        self,
        config: ImageProcessorConfig,
        bucket_manager: Optional[BucketManager] = None,
        vae = None
    ):
        """Initialize with optional bucket manager for resolution-aware processing."""
        self.config = config
        self.bucket_manager = bucket_manager
        
        # Initialize VAE encoder if VAE is provided
        self.vae_encoder = None
        if vae is not None:
            self.vae_encoder = VAEEncoder(
                vae=vae,
                config=VAEEncoderConfig(
                    device=config.device,
                    dtype=config.dtype,
                    enable_memory_efficient_attention=config.enable_memory_efficient_attention,
                    enable_vae_slicing=config.enable_vae_slicing,
                    vae_batch_size=config.vae_batch_size,
                    max_memory_usage=config.max_memory_usage
                )
            )
        
        # Initialize thread pool for parallel processing
        self.num_workers = min(
            self.config.num_workers,
            get_optimal_workers(memory_per_worker_gb=1.0)
        )
        self.executor = create_thread_pool(self.num_workers)
        
        # Build transform pipeline
        self.transform = self._build_transform()
        
        # Pre-allocate reusable tensors on GPU with dynamic sizing
        self.buffer_size = self._calculate_initial_buffer_size()
        self.tensor_buffer = self._allocate_tensor_buffer()
        
        logger.info(
            f"Initialized ImageProcessor:\n"
            f"- Device: {config.device}\n"
            f"- Dtype: {config.dtype}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Buffer size: {self.buffer_size}\n"
            f"- VAE: {'Yes' if self.vae_encoder is not None else 'No'}\n"
            f"- Bucket Manager: {'Yes' if bucket_manager is not None else 'No'}"
        )

    def _calculate_initial_buffer_size(self) -> Tuple[int, int, int, int]:
        """Calculate initial buffer size based on bucket manager if available."""
        if self.bucket_manager is not None:
            # Use largest bucket dimensions
            max_bucket = max(
                self.bucket_manager.buckets.values(),
                key=lambda b: b.resolution
            )
            return (
                32,  # Initial batch size
                3,   # RGB channels
                max_bucket.width,
                max_bucket.height
            )
        return (32, 3, 8192, 8192)  # Default size

    def _allocate_tensor_buffer(self) -> torch.Tensor:
        """Allocate tensor buffer with memory optimization."""
        try:
            return torch.empty(
                self.buffer_size,
                dtype=self.config.dtype,
                device=self.config.device
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce buffer size and retry
                self.buffer_size = (
                    self.buffer_size[0] // 2,
                    3,
                    self.buffer_size[2],
                    self.buffer_size[3]
                )
                logger.warning(f"Reduced buffer size to {self.buffer_size} due to OOM")
                return self._allocate_tensor_buffer()
            raise

    def load_image(self, path: str) -> Optional[Image.Image]:
        """Load and validate an image using utility function."""
        return load_and_validate_image(
            path,
            min_size=self.config.min_image_size,
            max_size=self.config.max_image_size,
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

    @torch.no_grad()
    def process_batch(self, images: List[Image.Image], width: int, height: int) -> torch.Tensor:
        """Process a batch of images with optimized parallel processing and GPU utilization."""
        batch_size = len(images)
        if batch_size == 0:
            return torch.empty(0, dtype=self.config.dtype, device=self.config.device)
        
        # Adjust buffer size if needed
        self._adjust_buffer_size(batch_size, width, height)
        
        # Get output tensor view
        output = self.tensor_buffer[:batch_size, :, :height, :width]
        
        # Process images in parallel with GPU acceleration
        futures = []
        for i, img in enumerate(images):
            future = self.executor.submit(self._process_single_image, img, width, height)
            futures.append((i, future))
        
        # Collect results efficiently
        for i, future in futures:
            try:
                img_tensor = future.result()
                output[i].copy_(img_tensor, non_blocking=True)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Fill with zeros on error
                output[i].zero_()
        
        return output

    @torch.no_grad()
    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE with optimized memory usage."""
        if self.config.enable_memory_efficient_attention:
            vae.enable_xformers_memory_efficient_attention()
        
        if self.config.enable_vae_slicing:
            vae.enable_slicing()
        
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        # Process in smaller batches to manage memory
        for i in range(0, batch_size, self.config.vae_batch_size):
            try:
                batch_slice = slice(i, min(i + self.config.vae_batch_size, batch_size))
                current_batch = pixel_values[batch_slice]
                
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    latents = vae.encode(current_batch).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents_list.append(latents)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Reduce batch size and retry
                    self.config.vae_batch_size = adjust_batch_size(
                        current_batch_size=self.config.vae_batch_size,
                        max_batch_size=32,
                        min_batch_size=1,
                        current_memory_usage=get_gpu_memory_usage(self.config.device),
                        max_memory_usage=self.config.max_memory_usage
                    )
                    logger.warning(f"GPU OOM, reducing VAE batch size to {self.config.vae_batch_size}")
                    torch.cuda.empty_cache()
                    # Retry with smaller batch size
                    return self.encode_vae(vae, pixel_values)
                raise
        
        # Combine all batches
        return torch.cat(latents_list, dim=0)

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
        # First resize the image
        img = self._resize_image(img, width, height)
        
        # Convert to tensor and normalize
        with torch.cuda.amp.autocast(dtype=self.config.dtype):
            img_tensor = self.transform(img).to(self.config.device, non_blocking=True)
            
        return img_tensor

    async def process_image(self, img: Image.Image) -> Optional[torch.Tensor]:
        """Process a single image asynchronously with VAE encoding and bucket optimization."""
        try:
            # Get image stats
            stats = self.get_image_stats(img)
            
            # Find optimal size using bucket manager if available
            width, height = self._get_optimal_size(stats['width'], stats['height'])
            
            # Log resize operation
            if (width, height) != (stats['width'], stats['height']):
                logger.debug(
                    f"Resizing image from {stats['width']}x{stats['height']} "
                    f"to {width}x{height}"
                )
            
            # Preprocess image
            processed = await asyncio.to_thread(
                self.preprocess,
                img,
                width,
                height
            )
            
            # Process through VAE if available
            if self.vae_encoder is not None:
                try:
                    processed = await self.vae_encoder.encode_image(processed)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Clear cache and retry with reduced batch size
                        torch.cuda.empty_cache()
                        self.vae_encoder.config.vae_batch_size //= 2
                        logger.warning(
                            f"Reduced VAE batch size to {self.vae_encoder.config.vae_batch_size}"
                        )
                        processed = await self.vae_encoder.encode_image(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)[:200]}...")
            return None

    def _default_resize(self, width: int, height: int) -> Tuple[int, int]:
        """Default resize logic when no bucket manager is available."""
        # Keep aspect ratio while fitting within max dimensions
        aspect = width / height
        
        if width > self.config.max_image_size[0]:
            width = self.config.max_image_size[0]
            height = int(width / aspect)
            
        if height > self.config.max_image_size[1]:
            height = self.config.max_image_size[1]
            width = int(height * aspect)
            
        # Ensure minimum dimensions
        width = max(width, self.config.min_image_size[0])
        height = max(height, self.config.min_image_size[1])
        
        return width, height

    async def cleanup(self):
        """Clean up resources asynchronously."""
        try:
            # Clean up thread pool
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clean up VAE encoder
            if self.vae_encoder is not None:
                await self.vae_encoder.cleanup()
            
            # Clear CUDA cache
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up image processor resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Don't re-raise as this is cleanup code