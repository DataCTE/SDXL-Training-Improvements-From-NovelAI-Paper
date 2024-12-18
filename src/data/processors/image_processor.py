# src/data/processors/image_processor.py
import torch
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any
from torchvision import transforms
import numpy as np
import logging
import torch.nn.functional as F
import asyncio
import gc

# Internal imports from utils
from src.data.processors.utils.system_utils import get_gpu_memory_usage, get_optimal_workers, create_thread_pool, cleanup_processor
from src.data.processors.utils.image_utils import load_and_validate_image, resize_image, get_image_stats
from src.data.processors.utils.image.vae_encoder import VAEEncoder
from src.config.config import VAEEncoderConfig  # Import the consolidated config

# Internal imports from processors
from src.data.processors.bucket import BucketManager

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(
        self,
        config: VAEEncoderConfig,
        bucket_manager: Optional[BucketManager] = None,
        vae = None
    ):
        """Initialize with optional bucket manager for resolution-aware processing."""
        self.config = config
        self.bucket_manager = bucket_manager
        
        # Initialize VAE encoder if VAE is provided
        self.vae_encoder = None
        if vae is not None:
            self.vae_encoder = VAEEncoder(vae=vae, config=config)
        
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

    @torch.no_grad()
    async def process_batch(
        self,
        images: List[Image.Image],
        width: int,
        height: int
    ) -> List[torch.Tensor]:
        """Process a batch of images with optimized speed and memory management."""
        batch_size = len(images)
        if batch_size == 0:
            return []
        
        processed_tensors = []
        
        try:
            available_memory = torch.cuda.get_device_properties(self.config.device).total_memory
            memory_per_image = width * height * 4 * 4
            
            optimal_batch_size = min(
                max(8, int(available_memory * 0.4 / memory_per_image)),
                128,
                batch_size
            )
            
            for i in range(0, batch_size, optimal_batch_size):
                sub_batch = images[i:i + optimal_batch_size]
                
                preprocessing_tasks = [
                    asyncio.to_thread(self.preprocess, img, width, height)
                    for img in sub_batch
                ]
                sub_processed = await asyncio.gather(*preprocessing_tasks)
                sub_processed = [t for t in sub_processed if t is not None]
                
                if sub_processed:
                    batch_tensor = torch.stack(sub_processed).to(
                        self.config.device, 
                        non_blocking=True
                    )
                    del sub_processed
                    
                    if self.vae_encoder is not None:
                        try:
                            encoded = await self.vae_encoder.encode_batch(batch_tensor)
                            for tensor in encoded:
                                processed_tensors.append(tensor.cpu())
                            del encoded
                        except Exception as e:
                            logger.error(f"VAE encoding error: {str(e)[:200]}...")
                            for _ in range(len(batch_tensor)):
                                processed_tensors.append(
                                    torch.zeros((4, height//8, width//8),
                                                dtype=self.config.dtype,
                                                device='cpu')
                                )
                    else:
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

    @torch.no_grad()
    def encode_vae(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        """Optimized VAE encoding with better memory management."""
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
                # Process in optimal sub-batches
                sub_batch_size = min(8, batch_size)
                latents_list = []
                
                for i in range(0, batch_size, sub_batch_size):
                    # Get current sub-batch
                    sub_batch = pixel_values[i:i+sub_batch_size]
                    
                    with torch.cuda.amp.autocast(dtype=self.config.dtype):
                        latents = vae.encode(sub_batch).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        
                        # Async transfer of results
                        with torch.cuda.stream(transfer_stream):
                            latents_list.append(latents)
                    
                    # Optional cleanup for very large batches
                    if i % (sub_batch_size * 4) == 0 and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(self.config.device).total_memory * 0.8:
                        compute_stream.synchronize()
                        torch.cuda.empty_cache()
                
                # Combine results efficiently
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