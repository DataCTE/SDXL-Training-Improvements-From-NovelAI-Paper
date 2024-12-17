# src/data/processors/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any
from torchvision import transforms
from dataclasses import dataclass
import numpy as np
import logging
import torch.nn.functional as F
import asyncio
import gc

# Internal imports from utils
from src.data.processors.utils.system_utils import get_gpu_memory_usage, get_optimal_workers, create_thread_pool
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
    max_image_size: Tuple[int, int] = (2048, 2048)
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
        if self.bucket_manager is not None and len(self.bucket_manager.buckets) > 0:
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
            
        # Fallback to default size if no buckets available
        logger.warning("No buckets available, using default buffer size")
        return (
            32,  # Initial batch size
            3,   # RGB channels
            self.config.max_image_size[0],  # Use max size from config
            self.config.max_image_size[1]
        )

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
    async def process_batch(
        self, 
        images: List[Image.Image], 
        width: int, 
        height: int
    ) -> List[torch.Tensor]:
        """Process a batch of images without caching."""
        batch_size = len(images)
        if batch_size == 0:
            return []
        
        processed_tensors = []
        
        # Process in smaller sub-batches
        sub_batch_size = min(4, batch_size)
        
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = images[i:i + sub_batch_size]
            
            try:
                # Process images
                sub_processed = []
                for img in sub_batch:
                    # Process single image
                    tensor = await self.process_image(img)
                    if tensor is not None:
                        sub_processed.append(tensor)
                    # Clear original image reference
                    del img
                
                if sub_processed:
                    # Stack tensors if we have any successful results
                    batch_tensor = torch.stack(sub_processed)
                    # Clear individual tensors after stacking
                    del sub_processed
                    
                    # Move to CPU immediately and append
                    for tensor in batch_tensor:
                        cpu_tensor = tensor.cpu()
                        processed_tensors.append(cpu_tensor)
                        del tensor  # Clear GPU tensor immediately
                    
                    del batch_tensor
                    torch.cuda.empty_cache()
                    gc.collect()
                    await asyncio.sleep(0.01)  # Give other tasks a chance to run
                
            except Exception as e:
                logger.error(f"Error processing sub-batch: {e}")
                # Create zero tensors on CPU directly for failed items
                for _ in range(len(sub_batch)):
                    processed_tensors.append(
                        torch.zeros((4, height//8, width//8), 
                        dtype=self.config.dtype, 
                        device='cpu')
                    )
            
            # Clear memory after each sub-batch
            torch.cuda.empty_cache()
            gc.collect()
        
        return processed_tensors

    @torch.no_grad()
    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through VAE one at a time to minimize memory usage."""
        if self.config.enable_memory_efficient_attention:
            vae.enable_xformers_memory_efficient_attention()
        
        if self.config.enable_vae_slicing:
            vae.enable_slicing()
        
        batch_size = pixel_values.shape[0]
        latents_list = []
        
        # Process one image at a time
        for i in range(batch_size):
            try:
                # Get single image
                current_image = pixel_values[i:i+1]
                
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    latents = vae.encode(current_image).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents_list.append(latents)
                
                # Clear cache after each image
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Error encoding image {i}: {str(e)[:200]}...")
                # Return empty tensor of correct shape for failed image
                latents_list.append(torch.zeros(
                    (1, 4, current_image.shape[2]//8, current_image.shape[3]//8),
                    dtype=self.config.dtype,
                    device=self.config.device
                ))
        
        # Combine all latents
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
                    # Clear cache before VAE encoding
                    torch.cuda.empty_cache()
                    processed = await self.vae_encoder.encode_image(processed)
                    # Clear cache after VAE encoding
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error in VAE encoding: {str(e)[:200]}...")
                    return None
            
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
        """Clean up resources."""
        try:
            # Clean up thread pool
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clean up VAE encoder
            if self.vae_encoder and hasattr(self.vae_encoder, 'cleanup'):
                await self.vae_encoder.cleanup()
            
            # Clear tensor buffer
            if hasattr(self, 'tensor_buffer'):
                del self.tensor_buffer
            
            # Clear transform pipeline
            if hasattr(self, 'transform'):
                del self.transform
            
            # Clear CUDA cache and force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Successfully cleaned up image processor resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Ensure cleanup when object is deleted."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during image processor deletion: {e}")