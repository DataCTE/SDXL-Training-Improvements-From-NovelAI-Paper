from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import traceback
import random
import re
import cv2
import numpy as np
from collections import defaultdict
from .ultimate_upscaler import UltimateUpscaler, USDUMode, USDUSFMode
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import gc
from .tag_weighter import TagBasedLossWeighter
import threading
import time
import multiprocessing as mp
from functools import lru_cache

logger = logging.getLogger(__name__)

# Create our own base classes
class CustomDatasetBase:
    def __init__(self):
        pass
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError


class CustomSamplerBase:
    def __init__(self):
        pass
        
    def __iter__(self):
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError


class CustomDataLoaderBase:
    def __init__(self):
        pass
        
    def __iter__(self):
        raise NotImplementedError


# Then modify your existing classes to use these bases
class CustomDataset(CustomDatasetBase):
    def __init__(self, data_dir, vae=None, tokenizer=None, tokenizer_2=None, text_encoder=None, text_encoder_2=None,
                 cache_dir="latents_cache", no_caching_latents=False, all_ar=False,
                 num_workers=None, prefetch_factor=2,
                 resolution_type="square", enable_bucket_sampler=True,
                 min_size=512, max_size=2048,
                 bucket_reso_steps=64,
                 token_dropout_rate=0.1, caption_dropout_rate=0.1,
                 min_tag_weight=0.1, max_tag_weight=3.0, use_tag_weighting=True,
                 finetune_vae=False, vae_learning_rate=1e-6, vae_train_freq=10,
                 adaptive_loss_scale=False, kl_weight=0.0, perceptual_weight=0.0,
                 **kwargs):
        super().__init__()
        
        # Data directory and paths
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Models and tokenizers
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # Processing settings
        self.no_caching_latents = no_caching_latents
        self.all_ar = all_ar
        self.num_workers = num_workers or min(8, os.cpu_count() or 1)
        self.prefetch_factor = prefetch_factor
        
        # Resolution and bucketing settings
        self.resolution_type = resolution_type
        self.enable_bucket_sampler = enable_bucket_sampler
        self.min_size = min_size
        self.max_size = max_size
        self.bucket_reso_steps = bucket_reso_steps
        
        # Text augmentation settings
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        
        # Tag weighting settings
        self.use_tag_weighting = use_tag_weighting
        self.min_tag_weight = min_tag_weight
        self.max_tag_weight = max_tag_weight
        
        # VAE finetuning settings
        self.finetune_vae = finetune_vae
        self.vae_learning_rate = vae_learning_rate
        self.vae_train_freq = vae_train_freq
        self.adaptive_loss_scale = adaptive_loss_scale
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize tag weighter with proper no_cache handling
        if use_tag_weighting:
            self.tag_weighter = TagBasedLossWeighter(
                min_weight=min_tag_weight,
                max_weight=max_tag_weight,
                no_cache=no_caching_latents
            )
        else:
            self.tag_weighter = None
            
        # Initialize multiprocessing components only if caching is enabled
        if not no_caching_latents:
            self.process_pool = None
            self.task_queue = None
            self.result_queue = None
            self.workers = []
            self.cache_lock = threading.Lock()
            self.latent_cache = {}
            
            # Initialize workers and process latents
            if self.num_workers > 0:
                self._initialize_workers()
                self._batch_process_latents_efficient()
        else:
            logger.info("Latent caching disabled - will process images on-the-fly")
            self.latent_cache = {}
            self.cache_lock = threading.Lock()
        
        # Initialize dataset structure
        self._initialize_dataset()
        
        # Initialize bucketing if enabled
        if self.enable_bucket_sampler:
            self._initialize_buckets()
            
        # Build tag statistics
        self.tag_stats = self._build_tag_statistics()
        
        # Set collate function
        self.collate_fn = self.custom_collate

    def _parse_tags(self, caption):
        """Parse tags using tag weighter class method"""
        if self.tag_weighter:
            # Always use instance method - static method not needed
            return self.tag_weighter.parse_tags(caption)
        return [], {}

    def _format_caption(self, caption):
        """Format caption using tag weighter instance method"""
        if not caption:
            return ""
        if self.tag_weighter:
            try:
                return self.tag_weighter.format_caption(caption)
            except Exception as e:
                logger.error(f"Caption formatting failed: {str(e)}")
                return caption
        return caption

    def _calculate_tag_weights(self, tags, special_tags):
        """Calculate tag weights using tag weighter instance method"""
        if self.tag_weighter:
            return self.tag_weighter.calculate_weights(tags, special_tags)
        return 1.0

    def _initialize_workers(self):
        """Initialize worker processes with proper error handling"""
        if self.no_caching_latents:
            logger.info("Skipping worker initialization - caching disabled")
            return
            
        try:
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()
            
            logger.info(f"Initializing {self.num_workers} worker processes for latent validation")
            
            # Create a copy of necessary attributes for workers
            worker_args = (
                self.task_queue,
                self.result_queue,
                self.vae,
                self.cache_dir
            )
            
            for _ in range(self.num_workers):
                p = mp.Process(
                    target=self._worker_process,
                    args=worker_args
                )
                p.daemon = True  # Ensure process cleanup
                p.start()
                self.workers.append(p)
                
        except Exception as e:
            logger.error(f"Failed to initialize workers: {str(e)}")
            self.cleanup_workers()
            raise

    def _worker_process(self, task_queue, result_queue):
        """Worker process function for parallel processing"""
        try:
            # Initialize worker's VAE copy
            if torch.cuda.is_available():
                device = f'cuda:{torch.cuda.current_device()}'
            else:
                device = 'cpu'
                
            while True:
                try:
                    # Get batch of images to process
                    batch_paths = task_queue.get()
                    if batch_paths is None:  # Poison pill
                        break
                        
                    batch_results = {}
                    batch_images = []
                    
                    # Load images
                    for path in batch_paths:
                        try:
                            image = self.load_and_transform_image(path)
                            if image is not None:
                                batch_images.append((path, image))
                        except Exception as e:
                            logger.warning(f"Worker failed to load {path}: {str(e)}")
                    
                    if not batch_images:
                        continue
                    
                    # Process batch through VAE
                    try:
                        with torch.no_grad():
                            batch_tensor = torch.stack([img for _, img in batch_images])
                            batch_tensor = batch_tensor.to(device)
                            
                            with torch.cuda.amp.autocast(enabled=True):
                                latents = self.vae.encode(batch_tensor).latent_dist.sample()
                            
                            # Store results
                            for idx, (path, _) in enumerate(batch_images):
                                if idx < len(latents):
                                    batch_results[path] = latents[idx].shape
                                    
                    except Exception as e:
                        logger.warning(f"Worker batch processing failed: {str(e)}")
                        
                    # Send results back
                    result_queue.put(batch_results)
                    
                except Exception as e:
                    logger.error(f"Worker process error: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal worker error: {str(e)}")
            logger.error(traceback.format_exc())

    def validate_and_cache_latents(self, image_paths):
        """Validate latent dimensions for a batch of images using multiple workers"""
        uncached_paths = [p for p in image_paths if p not in self.latent_cache]
        if not uncached_paths:
            return
            
        # Split work into batches
        batches = [uncached_paths[i:i + self.batch_size] 
                  for i in range(0, len(uncached_paths), self.batch_size)]
        
        # Submit batches to workers
        for batch in batches:
            self.task_queue.put(batch)
            
        # Collect results
        results = {}
        for _ in range(len(batches)):
            batch_results = self.result_queue.get()
            with self.cache_lock:
                self.latent_cache.update(batch_results)
                
        # Clear GPU memory
        torch.cuda.empty_cache()
        
    def __del__(self):
        """Cleanup worker processes"""
        if hasattr(self, 'process_pool') and self.process_pool:
            # Send poison pills to workers
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            
            # Wait for workers to finish
            for p in self.process_pool:
                p.join()
                
    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.image_paths)

    def _initialize_dataset(self):
        """Initialize dataset with improved image validation"""
        logger.info("Initializing dataset...")
        
        # Collect image paths
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            image_paths.extend(self.data_dir.glob(ext))
        
        # Validate images in parallel
        valid_paths = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for path in image_paths:
                futures.append(executor.submit(self._validate_image, path))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating images"):
                result = future.result()
                if result is not None:
                    valid_paths.append(result)
        
        self.image_paths = sorted(valid_paths)
        self.latent_paths = [self.cache_dir / f"{path.stem}_latents.pt" for path in self.image_paths]
        logger.info(f"Found {len(self.image_paths)} valid images")
        
    def _validate_image(self, img_path):
        """Validate single image with improved error handling"""
        try:
            # Check caption file
            caption_path = img_path.with_suffix('.txt')
            if not caption_path.exists():
                logger.warning(f"Skipping {img_path}: No caption file")
                return None
                
            # Validate image
            with Image.open(img_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                # Get dimensions
                width, height = img.size
                
                # If all_ar is True, accept all sizes
                if self.all_ar:
                    return img_path
                
                # Basic validation
                if width < self.min_size or height < self.min_size:
                    # Instead of skipping, try to upscale
                    target_width = max(width, self.min_size)
                    target_height = max(height, self.min_size)
                    img = self._upscale_image(img, target_width, target_height)
                    width, height = img.size
                    
                # Aspect ratio validation
                if not self.all_ar:
                    aspect_ratio = width / height
                    if self.resolution_type == "square" and not 0.95 <= aspect_ratio <= 1.05:
                        logger.warning(f"Skipping {img_path}: Not square ({aspect_ratio:.2f})")
                        return None
                    elif self.resolution_type == "portrait" and aspect_ratio >= 1:
                        logger.warning(f"Skipping {img_path}: Not portrait ({aspect_ratio:.2f})")
                        return None
                    elif self.resolution_type == "landscape" and aspect_ratio <= 1:
                        logger.warning(f"Skipping {img_path}: Not landscape ({aspect_ratio:.2f})")
                        return None
                
                return img_path
                
        except Exception as e:
            logger.error(f"Error validating {img_path}: {str(e)}")
            return None

    def _preprocess_images_parallel(self):
        """
        Parallel image preprocessing with more robust error handling
        Uses ThreadPoolExecutor for I/O-bound tasks
        """
        from concurrent.futures import ThreadPoolExecutor
        def safe_process_image(img_path):
            try:
                with Image.open(img_path) as image:
                    width, height = image.size
                    
                    # Skip processing if all_ar is True
                    if self.all_ar:
                        return img_path, None
                    
                    # Get target size
                    target_width, target_height = self._get_target_size(width, height)
                    
                    # Only process if resize is needed
                    if width != target_width or height != target_height:
                        processed = self.process_image_size(image, target_width, target_height)
                        output_path = img_path.with_suffix('.png')
                        processed.save(output_path, format='PNG', quality=95)
                        
                        # Remove original if processed
                        if output_path != img_path:
                            os.remove(img_path)
                        
                        return output_path, None
                    
                    return img_path, None
                
            except Exception as e:
                return img_path, str(e)

        # Efficient parallel processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(safe_process_image, path): path for path in self.image_paths}
            
            processed_images = []
            errors = []
            
            for future in as_completed(future_to_path):
                processed_path, error = future.result()
                
                if error:
                    errors.append((processed_path, error))
                else:
                    processed_images.append(processed_path)
            
            # Update image paths
            self.image_paths = processed_images
            
            # Log errors if any
            if errors:
                for path, error in errors:
                    logger.error(f"Error processing {path}: {error}")
        
        logger.info(f"Preprocessing complete: {len(processed_images)} images processed")

    def _batch_process_latents_efficient(self, batch_size=32):
        """Process and cache latents in batches using multiple workers with resource management"""
        logger.info(f"Caching latents for {len(self.image_paths)} images in batches")
        
        # Reduce worker count and chunk size to prevent resource exhaustion
        num_workers = min(8, (os.cpu_count() or 1))  # Limit max workers
        chunk_size = 1000  # Smaller chunk size
        logger.info(f"Using {num_workers} workers with chunk size {chunk_size}")
        
        # Group images by size first to reduce memory fragmentation
        size_groups = {}
        for img_path in self.image_paths:
            try:
                with Image.open(img_path) as img:
                    size_key = f"{img.size[0]}x{img.size[1]}"
                    if size_key not in size_groups:
                        size_groups[size_key] = []
                    size_groups[size_key].append(img_path)
            except Exception as e:
                logger.error(f"Error reading image {img_path}: {str(e)}")
                continue
        
        # Process each size group separately
        for size_key, paths in size_groups.items():
            logger.info(f"Processing {len(paths)} images of size {size_key}")
            
            # Process in smaller chunks
            for i in range(0, len(paths), chunk_size):
                chunk = paths[i:i + chunk_size]
                
                # Create a new process pool for each chunk to prevent resource leaks
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    try:
                        # Process chunk in parallel
                        futures = []
                        for j in range(0, len(chunk), batch_size):
                            batch = chunk[j:j + batch_size]
                            future = executor.submit(self.process_batch_with_vae, batch)
                            futures.append(future)
                        
                        # Wait for all futures to complete
                        for future in as_completed(futures):
                            try:
                                future.result()  # Get result to catch any exceptions
                            except Exception as e:
                                logger.error(f"Batch processing error: {str(e)}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue
                
                # Force garbage collection after each chunk
                gc.collect()
                torch.cuda.empty_cache()
                
                # Small delay to allow system resources to stabilize
                time.sleep(0.1)

    def process_batch_with_vae(self, batch_paths):
        """Process a batch of images through VAE with optimized memory usage"""
        if not batch_paths:
            return
            
        try:
            # Load and transform images
            valid_tensors = []
            valid_paths = []
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            # Process images one at a time to avoid memory spikes
            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        tensor = transform(img)
                        valid_tensors.append(tensor)
                        valid_paths.append(img_path)
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {str(e)}")
                    continue

            if not valid_tensors:
                return

            # Process through VAE in very small chunks to manage memory
            chunk_size = 4  # Smaller chunks for better memory management
            for i in range(0, len(valid_tensors), chunk_size):
                chunk_tensors = valid_tensors[i:i + chunk_size]
                chunk_paths = valid_paths[i:i + chunk_size]
                
                try:
                    # Move tensors to device and generate latents
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            image_tensor = torch.stack(chunk_tensors).to(self.vae.device, dtype=self.vae.dtype)
                            latents = self.vae.encode(image_tensor).latent_dist.sample()
                            latents = latents * self.vae.config.scaling_factor

                    # Save latents to disk immediately and free memory
                    for j, latent in enumerate(latents):
                        latent_path = self.cache_dir / f"{Path(chunk_paths[j]).stem}.pt"
                        torch.save(latent.cpu(), latent_path)
                        del latent
                    
                    # Clean up GPU memory
                    del image_tensor, latents
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error in VAE processing: {str(e)}")
                    continue
                
                # Small delay between chunks
                time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def validate_and_cache_latents(self, image_paths):
        """Validate latent dimensions for a batch of images efficiently"""
        uncached_paths = [p for p in image_paths if p not in self.latent_cache]
        if not uncached_paths:
            return
            
        # Process in batches
        for i in range(0, len(uncached_paths), self.batch_size):
            batch_paths = uncached_paths[i:i + self.batch_size]
            batch_images = []
            
            # Load images in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_images = list(executor.map(self.load_and_transform_image, batch_paths))
            
            # Process batch through VAE encoder
            with torch.no_grad():
                batch_tensor = torch.stack([img for img in batch_images if img is not None])
                if len(batch_tensor) == 0:
                    continue
                    
                batch_tensor = batch_tensor.to(self.vae.device)
                try:
                    # Encode in mixed precision if available
                    with torch.cuda.amp.autocast(enabled=True):
                        latents = self.vae.encode(batch_tensor).latent_dist.sample()
                    
                    # Cache results
                    for idx, path in enumerate(batch_paths):
                        if idx < len(latents):
                            self.latent_cache[path] = latents[idx].shape
                            
                except Exception as e:
                    logger.warning(f"Batch encoding failed: {str(e)}")
                    # Fall back to individual processing
                    for path, img in zip(batch_paths, batch_images):
                        if img is not None:
                            try:
                                with torch.cuda.amp.autocast(enabled=True):
                                    latent = self.vae.encode(img.unsqueeze(0).to(self.vae.device)).latent_dist.sample()
                                self.latent_cache[path] = latent.shape
                            except Exception as e:
                                logger.warning(f"Failed to process {path}: {str(e)}")
                                
                # Clear GPU memory
                torch.cuda.empty_cache()
                
    def check_latent_dimensions(self, image_path):
        """Check if image has correct latent dimensions using cache"""
        if image_path in self.latent_cache:
            return self.latent_cache[image_path] == self.expected_latent_shape
            
        # If not in cache, queue for batch processing
        self.validate_and_cache_latents([image_path])
        return self.latent_cache.get(image_path, None) == self.expected_latent_shape

    def load_and_transform_image(self, image_path):
        """Load and transform image for VAE encoding"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {str(e)}")
            return None

    def _build_tag_statistics(self):
        """Build dataset-wide tag statistics and format captions"""
        stats = {
            'niji_count': 0,
            'quality_6_count': 0,
            'stylize_values': [],
            'chaos_values': [],
            'total_images': len(self.image_paths),
            'formatted_count': 0
        }
        
        for img_path in tqdm(self.image_paths, desc="Processing captions"):
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()
                    
                    # Format caption
                    formatted_caption = self._format_caption(original_caption)
                    
                    # Save if changed
                    if formatted_caption != original_caption:
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(formatted_caption)
                        stats['formatted_count'] += 1
                    
                    # Update tag statistics
                    _, special_tags = self._parse_tags(formatted_caption)
                    if special_tags.get('niji', False):
                        stats['niji_count'] += 1
                    if special_tags.get('version', 0) == 6:
                        stats['quality_6_count'] += 1
                    if 'stylize' in special_tags:
                        stats['stylize_values'].append(special_tags['stylize'])
                    if 'chaos' in special_tags:
                        stats['chaos_values'].append(special_tags['chaos'])
                        
                except Exception as e:
                    logger.warning(f"Error processing caption for {img_path}: {str(e)}")
        
        logger.info(f"Formatted {stats['formatted_count']} captions")
        return stats

    def _get_target_size(self, width, height):
        """Calculate target size preserving aspect ratio without upper limits"""
        aspect_ratio = height / width

        # For extreme aspect ratios, we'll preserve them while keeping reasonable dimensions
        if aspect_ratio > 1:  # Portrait
            # Start with target width and calculate height to maintain AR
            target_width = 1024  # Base width for portrait
            target_height = int(round(target_width * aspect_ratio / self.bucket_reso_steps) * self.bucket_reso_steps)
        else:  # Landscape
            # Start with target height and calculate width to maintain AR
            target_height = 1024  # Base height for landscape
            target_width = int(round(target_height / aspect_ratio / self.bucket_reso_steps) * self.bucket_reso_steps)

        # Scale down if needed while preserving AR
        max_dim = max(target_width, target_height)
        if max_dim > 2048:  # Only scale down if absolutely necessary
            scale = 2048 / max_dim
            target_width = int(round(target_width * scale / self.bucket_reso_steps) * self.bucket_reso_steps)
            target_height = int(round(target_height * scale / self.bucket_reso_steps) * self.bucket_reso_steps)

        # Ensure minimum dimensions
        target_width = max(target_width, self.min_size)
        target_height = max(target_height, self.min_size)

        return target_width, target_height

    def process_image_size(self, image, target_width, target_height):
        """Process image size with advanced resizing"""
        width, height = image.size
        
        # Get target size considering aspect ratio constraints
        target_width, target_height = self._get_target_size(width, height)
        
        # Resize to target size using LANCZOS resampling for better quality
        if width != target_width or height != target_height:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
        return image

    def _upscale_image(self, image, target_width, target_height):
        """Upscale image using Ultimate SD Upscaler"""
        if self.all_ar:
            # Just return original image when all_ar is True
            return image
            
        if self.upscaler is None:
            logger.warning("Upscaler not initialized but _upscale_image called")
            return image
            
        try:
            # Calculate scale factor
            scale_factor = max(target_width / image.width, target_height / image.height)
            
            # Get caption for the image
            caption_path = Path(str(image.filename)).with_suffix('.txt')
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            except:
                prompt = "high quality, detailed image"  # fallback prompt
            
            # Process with upscaler
            upscaled = self.upscaler.upscale(
                image=image,
                prompt=prompt,
                upscale_factor=scale_factor,
                mode=USDUMode.LINEAR,
                tile_width=512,
                tile_height=512,
                padding=32,
                num_steps=20,
                guidance_scale=7.5,
                strength=0.4,
                seam_fix_mode=USDUSFMode.HALF_TILE,
                seam_fix_denoise=0.35,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16
            )
            
            # Final resize to exact target size if needed
            if upscaled.size != (target_width, target_height):
                upscaled = upscaled.resize((target_width, target_height), Image.LANCZOS)
                
            return upscaled
            
        except Exception as e:
            logger.error(f"AI upscaling failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def _downscale_image(self, image, target_width, target_height):
        """High-quality downscaling optimized for performance while maintaining quality"""
        try:
            # Convert to numpy once at the start
            img_np = np.array(image)
            
            # Calculate scale factor
            scale_factor = max(image.width / target_width, image.height / target_height)
            
            # Calculate number of steps - fewer steps for smaller scale factors
            num_steps = max(1, min(3, int(scale_factor // 1.5)))
            
            # Apply initial gaussian blur to prevent aliasing
            # Adjust blur based on scale factor
            blur_radius = min(2.0, 0.6 * scale_factor)
            kernel_size = int(blur_radius * 3) | 1  # ensure odd number
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), blur_radius)
            
            # Apply edge-preserving bilateral filter for large downscaling
            if scale_factor > 2.0:
                d = 9  # diameter of pixel neighborhood
                sigma_color = 75
                sigma_space = 75
                img_np = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
            
            # Progressive downscaling
            for step in range(num_steps):
                # Calculate intermediate size
                progress = (step + 1) / num_steps
                intermediate_width = int(target_width + (image.width - target_width) * (1 - progress))
                intermediate_height = int(target_height + (image.height - target_height) * (1 - progress))
                
                # Use INTER_AREA for downscaling
                img_np = cv2.resize(img_np, (intermediate_width, intermediate_height), 
                                  interpolation=cv2.INTER_AREA)
                
                # Apply light sharpening on final step
                if step == num_steps - 1:
                    kernel = np.array([[-0.5,-0.5,-0.5], 
                                     [-0.5, 5.0,-0.5],
                                     [-0.5,-0.5,-0.5]]) / 2.0
                    img_np = cv2.filter2D(img_np, -1, kernel)
            
            # Convert back to PIL
            return Image.fromarray(img_np)
            
        except Exception as e:
            logger.error(f"Advanced downscaling failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def _initialize_buckets(self):
        """Initialize buckets dynamically based on dataset content with unlimited AR support"""
        from concurrent.futures import ThreadPoolExecutor
        import math
        import logging
        from collections import defaultdict
        import numpy as np
        
        logger = logging.getLogger(__name__)
        
        # If all_ar is True, create individual buckets for each unique size
        if self.all_ar:
            logger.info("all_ar enabled - creating individual buckets for each image size")
            
            # Initialize temporary storage for image sizes
            size_groups = defaultdict(list)
            
            def analyze_image_batch(img_paths):
                results = []
                for img_path in img_paths:
                    try:
                        with Image.open(img_path) as img:
                            if img.mode not in ('RGB', 'RGBA'):
                                img = img.convert('RGB')
                            width, height = img.size
                            # Store exact dimensions for all_ar mode
                            results.append((img_path, (height, width)))
                    except Exception as e:
                        logger.error(f"Error analyzing {img_path}: {str(e)}")
                return results

            # Process images in batches
            batch_size = 1000
            total_images = len(self.image_paths)
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, total_images, batch_size):
                    batch = self.image_paths[i:i + batch_size]
                    futures.append(executor.submit(analyze_image_batch, batch))
                
                # Collect results and group by size
                for future in as_completed(futures):
                    for img_path, size in future.result():
                        size_groups[size].append(img_path)

            # Create buckets for each unique size
            self.buckets = list(size_groups.keys())
            self.bucket_data = size_groups
            
            logger.info(f"Created {len(self.buckets)} unique size buckets for all_ar mode")
            
        else:
            # Initialize temporary storage for image sizes
            image_sizes = []
            total_images = len(self.image_paths)
            logger.info(f"Analyzing {total_images} images using {self.num_workers} workers")

            # Only use cache if not in no_caching mode
            if not self.no_caching_latents:
                get_target_size = lru_cache(maxsize=1024)(self._get_target_size)
            else:
                get_target_size = self._get_target_size

            # Process images in batches for better memory efficiency
            batch_size = 1000
            def analyze_image_batch(img_paths):
                results = []
                for img_path in img_paths:
                    try:
                        with Image.open(img_path) as img:
                            if img.mode not in ('RGB', 'RGBA'):
                                img = img.convert('RGB')
                            width, height = img.size
                            target_h, target_w = get_target_size(width, height)
                            if target_h and target_w:  # Ensure valid dimensions
                                results.append((target_h, target_w))
                    except Exception as e:
                        logger.error(f"Error analyzing {img_path}: {str(e)}")
                return results

            # Process images in batches
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(analyze_image_batch, self.image_paths[i:i + batch_size]): i 
                    for i in range(0, total_images, batch_size)
                }
                
                # Collect results
                for i, future in enumerate(future_to_path):
                    try:
                        batch_results = future.result()
                        image_sizes.extend(batch_results)
                        processed = min((i + 1) * batch_size, total_images)
                        if processed % 1000 == 0:
                            logger.info(f"Analyzed {processed}/{total_images} images")
                    except Exception as e:
                        logger.error(f"Error in batch {i}: {str(e)}")

            if not image_sizes:
                raise RuntimeError("No valid images found in dataset")

            # Convert to numpy array for efficient operations
            sizes_array = np.array(image_sizes)
            
            # Calculate aspect ratios and areas
            aspect_ratios = sizes_array[:, 0] / sizes_array[:, 1]  # height/width ratios
            areas = sizes_array[:, 0] * sizes_array[:, 1]

            # Create bucket steps based on distribution
            size_groups = defaultdict(list)
            
            # Process each image size to create appropriate buckets
            for h, w in sizes_array:
                # Round dimensions to nearest step
                bucket_h = max(self.min_size, round(h / self.bucket_reso_steps) * self.bucket_reso_steps)
                bucket_w = max(self.min_size, round(w / self.bucket_reso_steps) * self.bucket_reso_steps)
                
                # Ensure dimensions don't exceed max_size
                if self.max_size:
                    scale = min(1.0, self.max_size / max(bucket_h, bucket_w))
                    if scale < 1.0:
                        bucket_h = round(bucket_h * scale / self.bucket_reso_steps) * self.bucket_reso_steps
                        bucket_w = round(bucket_w * scale / self.bucket_reso_steps) * self.bucket_reso_steps
                
                size_groups[(bucket_h, bucket_w)].append((h, w))

            # Filter buckets to remove those with too few images
            min_images = max(2, total_images // 1000)  # Adjust threshold based on dataset size
            self.buckets = [k for k, v in size_groups.items() if len(v) >= min_images]
            
            # Sort buckets by area for efficient batching
            self.buckets.sort(key=lambda x: x[0] * x[1])

            # Create bucket data structure
            self.bucket_data = {bucket: [] for bucket in self.buckets}

            # Log bucket information
            logger.info(f"Created {len(self.buckets)} buckets:")
            for bucket in self.buckets:
                bucket_h, bucket_w = bucket
                count = len(size_groups[bucket])
                logger.info(f"  {bucket_h}x{bucket_w}: {count} images")
            
            # Calculate and log statistics
            total_buckets = len(self.buckets)
            total_bucketed_images = sum(len(size_groups[b]) for b in self.buckets)
            coverage = total_bucketed_images / total_images * 100
            
            logger.info(f"Bucket statistics:")
            logger.info(f"  Total buckets: {total_buckets}")
            logger.info(f"  Images in buckets: {total_bucketed_images}/{total_images} ({coverage:.1f}%)")
            logger.info(f"  Aspect ratio range: {aspect_ratios.min():.2f} to {aspect_ratios.max():.2f}")
            logger.info(f"  Area range: {areas.min():.0f} to {areas.max():.0f}")

    def _assign_to_bucket(self, img_path):
        """Assign an image to the appropriate bucket based on mode"""
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                if self.all_ar:
                    # In all_ar mode, use exact dimensions as bucket
                    bucket = (height, width)
                    if bucket not in self.buckets:
                        self.buckets.append(bucket)
                        self.bucket_data[bucket] = []
                    return img_path, bucket
                else:
                    # Original bucket assignment logic for non-all_ar mode
                    # Calculate target dimensions
                    target_h, target_w = self._get_target_size(width, height)
                    
                    # Find best fitting bucket
                    best_bucket = None
                    min_area_diff = float('inf')
                    
                    for bucket_h, bucket_w in self.buckets:
                        # Skip if bucket is too small
                        if bucket_h < target_h or bucket_w < target_w:
                            continue
                        
                        # Calculate area difference
                        area_diff = (bucket_h * bucket_w) - (target_h * target_w)
                        
                        # Update best bucket if this one is better
                        if area_diff < min_area_diff:
                            min_area_diff = area_diff
                            best_bucket = (bucket_h, bucket_w)
                    
                    # If no suitable bucket found, use largest bucket that maintains aspect ratio
                    if best_bucket is None:
                        aspect_ratio = height / width
                        best_area_diff = float('inf')
                        
                        for bucket_h, bucket_w in self.buckets:
                            bucket_ar = bucket_h / bucket_w
                            if abs(bucket_ar - aspect_ratio) < 0.1:  # Allow some AR tolerance
                                area_diff = abs((bucket_h * bucket_w) - (target_h * target_w))
                                if area_diff < best_area_diff:
                                    best_area_diff = area_diff
                                    best_bucket = (bucket_h, bucket_w)
                    
                    # If still no bucket found, use the largest available bucket
                    if best_bucket is None:
                        best_bucket = max(self.buckets, key=lambda x: x[0] * x[1])
                        logger.warning(f"Using largest bucket {best_bucket} for image {img_path} "
                                     f"with dimensions {width}x{height}")
                    
                    return img_path, best_bucket
                
        except Exception as e:
            logger.error(f"Error assigning {img_path} to bucket: {str(e)}")
            return img_path, None

    def _apply_text_transforms(self, caption):
        """Apply text augmentation transforms"""
        if random.random() < self.caption_dropout_rate:
            return ""
        
        if self.token_dropout_rate > 0:
            tokens = caption.split(",")
            tokens = [token.strip() for token in tokens if token.strip()]
            
            # Keep tokens with probability (1 - token_dropout_rate)
            tokens = [token for token in tokens if random.random() > self.token_dropout_rate]
            
            # Ensure at least one token remains
            if not tokens:
                tokens = [random.choice(caption.split(",")).strip()]
            
            return ", ".join(tokens)
        
        return caption

    def _process_uncached_item(self, img_path, caption_path):
        """Process an item that hasn't been cached yet"""
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            
            # Get bucket dimensions based on all_ar setting
            if self.all_ar:
                width, height = image.size
                bucket_h, bucket_w = height, width
            else:
                bucket_h, bucket_w = self._get_bucket_size(image.size)
            
            # Process image with advanced resizing
            processed_image = self._advanced_resize(image, bucket_w, bucket_h)
            
            # Generate latents on-the-fly if no_caching is enabled
            if self.no_caching_latents:
                with torch.no_grad():
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    image_tensor = transform(processed_image).unsqueeze(0)
                    image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents = latents.squeeze(0)
            else:
                # Use cached latents
                latents = self._get_cached_latents(img_path, processed_image)
            
            # Load and process caption
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Apply text augmentation
            caption = self._apply_text_transforms(caption)
            
            # Generate text embeddings
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder.device)
            
            text_inputs_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder_2.device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs_2.input_ids,
                    output_hidden_states=True
                )
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
            
            # Generate time embeddings
            original_size = image.size
            target_size = (bucket_w, bucket_h)
            crop_coords_top_left = (0, 0)
            crop_coords_bottom_right = original_size
            
            add_time_ids = torch.tensor([
                original_size[0],          # original width
                original_size[1],          # original height
                target_size[0],            # target width
                target_size[1],            # target height
                crop_coords_top_left[0],     # crop left
                crop_coords_top_left[1],     # crop top
                crop_coords_bottom_right[0], # crop right
                crop_coords_bottom_right[1]  # crop bottom
            ], dtype=torch.float32, device=self.vae.device)
            
            # Add batch dimension: [1, 6]
            add_time_ids = add_time_ids.unsqueeze(0)
            
            # Prepare cache data
            cache_data = {
                'latents': latents,
                'text_embeddings': text_embeddings,
                'text_embeddings_2': hidden_states,
                'added_cond_kwargs': {
                    'text_embeds': pooled_output.unsqueeze(1).unsqueeze(2),
                    'time_ids': add_time_ids
                },
                'original_caption': caption,
                'bucket_size': (bucket_h, bucket_w)
            }
            
            return cache_data
    
    def _get_bucket_size(self, image_size):
        """Get the most appropriate bucket size for an image"""
        width, height = image_size
        
        if self.all_ar:
            # In all_ar mode, return exact dimensions
            return (height, width)
        
        # Find best fitting bucket
        best_bucket = None
        min_area_diff = float('inf')
        
        for bucket_h, bucket_w in self.buckets:
            if bucket_h < height or bucket_w < width:
                continue
            
            area_diff = (bucket_h * bucket_w) - (height * width)
            if area_diff < min_area_diff:
                min_area_diff = area_diff
                best_bucket = (bucket_h, bucket_w)
        
        if best_bucket is None:
            # If no bucket fits, use the largest bucket
            best_bucket = max(self.buckets, key=lambda x: x[0] * x[1])
        
        return best_bucket
    
    def _advanced_resize(self, image, target_width, target_height):
        """Advanced image resizing with aspect ratio preservation"""
        width, height = image.size
        
        # Calculate target dimensions preserving aspect ratio
        aspect_ratio = width / height
        
        if aspect_ratio > target_width / target_height:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # High-quality downscaling
        if width > new_width or height > new_height:
            # Use Lanczos for downscaling
            image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            # Use bicubic for upscaling
            image = image.resize((new_width, new_height), Image.BICUBIC)
        
        # Create new image with padding
        result = Image.new('RGB', (target_width, target_height))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        result.paste(image, (left, top))
        
        return result

    def custom_collate(self, batch):
        """Custom collate function that ensures all items in a batch are from the same bucket"""
        # Filter out None values from failed __getitem__ calls
        batch = [item for item in batch if item is not None]
        if not batch:
            raise RuntimeError("Empty batch after filtering None values")
        
        # All items should have the same bucket size since we're using BucketSampler
        bucket_size = batch[0]['bucket_size']
        if not all(item['bucket_size'] == bucket_size for item in batch):
            if not self.all_ar:  # Only raise error if not in all_ar mode
                sizes = [item['bucket_size'] for item in batch]
                raise ValueError(f"Inconsistent bucket sizes in batch: {sizes}")
        
        # Stack tensors
        try:
            latents = torch.stack([item['latents'] for item in batch])
            text_embeddings = torch.stack([item['text_embeddings'] for item in batch])
            text_embeddings_2 = torch.stack([item['text_embeddings_2'] for item in batch])
            
            # Handle added_cond_kwargs if present
            added_cond_kwargs = {}
            if 'added_cond_kwargs' in batch[0]:
                for key in batch[0]['added_cond_kwargs']:
                    if torch.is_tensor(batch[0]['added_cond_kwargs'][key]):
                        added_cond_kwargs[key] = torch.stack([
                            item['added_cond_kwargs'][key] for item in batch
                        ])
                    else:
                        added_cond_kwargs[key] = [
                            item['added_cond_kwargs'][key] for item in batch
                        ]
            
            return {
                'latents': latents,
                'text_embeddings': text_embeddings,
                'text_embeddings_2': text_embeddings_2,
                'added_cond_kwargs': added_cond_kwargs,
                'bucket_size': bucket_size
            }
            
        except Exception as e:
            logger.error(f"Error in custom_collate: {str(e)}")
            logger.error(f"Batch sizes: {[item['latents'].shape for item in batch]}")
            logger.error(f"Bucket sizes: {[item.get('bucket_size') for item in batch]}")
            raise

    def __getitem__(self, idx):
        """Get a single item with improved text augmentation and bucketing"""
        try:
            # Get image path and caption path
            img_path = self.image_paths[idx]
            caption_path = Path(img_path).with_suffix('.txt')
            
            # Get bucket dimensions if bucketing is enabled
            if self.enable_bucket_sampler:
                bucket = self.image_to_bucket.get(img_path)
                if bucket is None:
                    if self.all_ar:
                        # For all_ar mode, get original dimensions
                        with Image.open(img_path) as img:
                            width, height = img.size
                            bucket = (height, width)
                    else:
                        raise ValueError(f"No bucket found for image {img_path}")
                bucket_h, bucket_w = bucket
            else:
                bucket_h = bucket_w = None
            
            # Load and process image
            with Image.open(img_path) as image:
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Resize image according to bucket if bucketing is enabled
                if self.enable_bucket_sampler:
                    image = self._advanced_resize(image, bucket_w, bucket_h)
                else:
                    # Apply default resizing if no bucketing
                    width, height = image.size
                    target_w, target_h = self._get_target_size(width, height)
                    image = self._advanced_resize(image, target_w, target_h)
            
            # Load and process caption
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Apply text augmentation
            caption = self._apply_text_transforms(caption)
            
            # Generate text embeddings
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder.device)
            
            text_inputs_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder_2.device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs_2.input_ids,
                    output_hidden_states=True
                )
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
            
            # Generate time embeddings
            original_size = image.size
            target_size = (bucket_w, bucket_h) if self.enable_bucket_sampler else (target_w, target_h)
            crop_coords_top_left = (0, 0)
            crop_coords_bottom_right = original_size
            
            add_time_ids = torch.tensor([
                original_size[0],          # original width
                original_size[1],          # original height
                target_size[0],            # target width
                target_size[1],            # target height
                crop_coords_top_left[0],     # crop left
                crop_coords_top_left[1],     # crop top
                crop_coords_bottom_right[0], # crop right
                crop_coords_bottom_right[1]  # crop bottom
            ], dtype=torch.float32, device=self.vae.device)
            
            # Add batch dimension: [1, 6]
            add_time_ids = add_time_ids.unsqueeze(0)
            
            # Prepare cache data
            cache_data = {
                'latents': None,
                'text_embeddings': text_embeddings,
                'text_embeddings_2': hidden_states,
                'added_cond_kwargs': {
                    'text_embeds': pooled_output.unsqueeze(1).unsqueeze(2),
                    'time_ids': add_time_ids
                },
                'original_caption': caption,
                'bucket_size': (bucket_h, bucket_w) if self.enable_bucket_sampler else (target_w, target_h)
            }
            
            # Handle latents based on caching mode
            if self.no_caching_latents:
                # Always generate latents on-the-fly if caching is disabled
                with torch.no_grad():
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    image_tensor = transform(image).unsqueeze(0)
                    image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                    
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    cache_data['latents'] = latents.squeeze(0).cpu()
            else:
                # Only use cached latents if caching is enabled
                if self.latent_paths[idx].exists():
                    cache_data_latents = torch.load(self.latent_paths[idx], map_location='cpu')
                    cache_data['latents'] = cache_data_latents['latents'].float().cpu()
                else:
                    # Generate and cache if file doesn't exist
                    with torch.no_grad():
                        # Convert image to tensor
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
                        image_tensor = transform(image).unsqueeze(0)
                        image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                        
                        # Generate latents
                        latents = self.vae.encode(image_tensor).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                        cache_data['latents'] = latents.squeeze(0).cpu()
                        if not self.no_caching_latents:  # Only save if caching is enabled
                            torch.save({'latents': cache_data['latents']}, self.latent_paths[idx])
            
            return cache_data
        
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            return None

    def _get_cached_latents(self, img_path, processed_image):
        """Get latents either from cache or generate them"""
        cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
        
        if not self.no_caching_latents and cache_path.exists():
            return torch.load(cache_path, map_location='cpu')['latents']
        
        # Generate latents
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image_tensor = transform(processed_image).unsqueeze(0)
            image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.squeeze(0)
            
            # Cache only if caching is enabled
            if not self.no_caching_latents:
                torch.save({'latents': latents}, cache_path)
            
            return latents


class BucketSampler(CustomSamplerBase):
    """
    Sampler that creates batches of samples from the same bucket to ensure consistent tensor sizes.
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__()
        if not hasattr(dataset, 'bucket_data'):
            raise ValueError("Dataset must have 'bucket_data' attribute for BucketSampler")
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.all_ar = dataset.all_ar
        
        # Store (bucket, index) pairs for all valid samples
        self.samples = []
        
        if self.all_ar:
            # In all_ar mode, each image is its own bucket
            for bucket, img_paths in dataset.bucket_data.items():
                for img_path in img_paths:
                    idx = dataset.image_paths.index(img_path)
                    self.samples.append((bucket, idx))
        else:
            # Original bucket validation logic for non-all_ar mode
            # Validate buckets and create sample list
            for bucket, img_paths in dataset.bucket_data.items():
                bucket_h, bucket_w = bucket
                
                # Skip invalid buckets
                if bucket_h < dataset.min_size or bucket_w < dataset.min_size:
                    logger.warning(f"Skipping bucket {bucket} - dimensions too small")
                    continue
                    
                if dataset.max_size and (bucket_h > dataset.max_size or bucket_w > dataset.max_size):
                    logger.warning(f"Skipping bucket {bucket} - dimensions exceed max_size")
                    continue
                
                # Get indices for images in this bucket
                for img_path in img_paths:
                    try:
                        idx = dataset.image_paths.index(img_path)
                        self.samples.append((bucket, idx))
                    except ValueError:
                        logger.warning(f"Image {img_path} not found in dataset paths")
                        continue
                
                # Log bucket statistics
                if len(img_paths) > 0:
                    logger.debug(f"Bucket {bucket}: {len(img_paths)} images")
            
            # Verify we have valid samples
            if not self.samples:
                raise ValueError("No valid samples found after bucket validation")
                
            # Sort samples by bucket size for potentially better memory usage
            self.samples.sort(key=lambda x: x[0][0] * x[0][1])
            
            logger.info(f"BucketSampler initialized with {len(self.samples)} valid samples "
                       f"across {len(dataset.bucket_data)} buckets")
    
    def __iter__(self):
        if not self.samples:
            raise RuntimeError("No samples available for iteration")
            
        if self.all_ar:
            # In all_ar mode, we can shuffle all samples
            indices = [idx for _, idx in self.samples]
            random.shuffle(indices)
            
            # Create batches (they may have different sizes)
            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            
            # Shuffle batches
            random.shuffle(batches)
            
            # Flatten batches
            return iter([idx for batch in batches for idx in batch])
        else:
            # Original iteration logic for non-all_ar mode
            # Group samples by bucket
            bucket_groups = defaultdict(list)
            for bucket, idx in self.samples:
                bucket_groups[bucket].append(idx)
            
            # Create batches for each bucket
            batches = []
            for bucket, indices in bucket_groups.items():
                # Shuffle indices within each bucket
                indices = indices.copy()
                random.shuffle(indices)
                
                # Create batches of the specified size
                for i in range(0, len(indices), self.batch_size):
                    batch = indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
            
            # Shuffle the batches themselves
            random.shuffle(batches)
            
            # Flatten batches into a single list of indices
            indices = []
            for batch in batches:
                indices.extend(batch)
            
            return iter(indices)
    
    def __len__(self):
        if self.drop_last:
            return len(self.samples) - (len(self.samples) % self.batch_size)
        return len(self.samples)


class CustomDataLoader(CustomDataLoaderBase):
    def __init__(self, dataset, batch_size, sampler=None, num_workers=0, 
                 pin_memory=False, drop_last=False, timeout=0, 
                 worker_init_fn=None, prefetch_factor=2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        
        # Initialize multiprocessing components if using workers
        if self.num_workers > 0:
            self.worker_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.worker_pool = None

    def __iter__(self):
        if self.sampler is None:
            indices = range(len(self.dataset))
        else:
            indices = self.sampler

        # Create batches
        batches = []
        batch = []
        
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
                
        # Handle last batch if not dropping
        if batch and not self.drop_last:
            batches.append(batch)

        # Process batches
        for batch_indices in batches:
            if self.num_workers > 0:
                # Parallel processing
                futures = [
                    self.worker_pool.submit(self.dataset.__getitem__, idx)
                    for idx in batch_indices
                ]
                batch_data = [future.result() for future in futures]
            else:
                # Single thread processing
                batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Filter out None values from failed __getitem__ calls
            batch_data = [item for item in batch_data if item is not None]
            
            if batch_data:  # Only yield if we have valid items
                try:
                    # Use dataset's collate function if available
                    if hasattr(self.dataset, 'custom_collate'):
                        yield self.dataset.custom_collate(batch_data)
                    else:
                        yield batch_data
                except Exception as e:
                    logger.error(f"Error collating batch: {str(e)}")
                    continue

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __del__(self):
        # Cleanup
        if self.worker_pool is not None:
            self.worker_pool.shutdown(wait=True)

def create_dataloader(
    data_dir,
    batch_size,
    num_workers=None,
    tokenizer=None,
    text_encoder=None,
    tokenizer_2=None,
    text_encoder_2=None,
    vae=None,
    enable_bucket_sampler=True,
    no_caching_latents=False,
    all_ar=False,
    **kwargs
):
    """
    Create a dataloader with the specified parameters
    """
    # Initialize dataset
    dataset = CustomDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        vae=vae,
        num_workers=num_workers,
        enable_bucket_sampler=enable_bucket_sampler,
        no_caching_latents=no_caching_latents,
        all_ar=all_ar,
        **kwargs
    )

    # Create sampler if bucket sampling is enabled
    if enable_bucket_sampler:
        sampler = BucketSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True
        )
    else:
        sampler = None

    # Create and return dataloader
    return CustomDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers if num_workers is not None else 0,
        pin_memory=True,
        drop_last=True,
        timeout=0,
        prefetch_factor=2
    )