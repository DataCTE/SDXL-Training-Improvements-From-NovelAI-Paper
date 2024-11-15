from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler
import math
from tqdm import tqdm
import logging
import traceback
import random
import re
import cv2
import numpy as np
from collections import defaultdict
from .ultimate_upscaler import UltimateUpscaler, USDUMode, USDUSFMode
from utils.validation import validate_image_dimensions
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from .tag_weighter import TagBasedLossWeighter
from multiprocessing import Process, Queue
import threading

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae=None, tokenizer=None, tokenizer_2=None, text_encoder=None, text_encoder_2=None,
                 cache_dir="latents_cache", no_caching_latents=False, all_ar=False,
                 num_workers=None, prefetch_factor=2,
                 resolution_type="square", enable_bucket_sampler=True,
                 min_size=512, max_size=2048,  # Global min/max image dimensions
                 bucket_reso_steps=64,  # Resolution steps for bucketing
                 token_dropout_rate=0.1, caption_dropout_rate=0.1,
                 min_tag_weight=0.1, max_tag_weight=3.0, use_tag_weighting=True,
                 finetune_vae=False, vae_learning_rate=1e-6, vae_train_freq=10,
                 adaptive_loss_scale=False, kl_weight=0.0, perceptual_weight=0.0,
                 **kwargs):
        super().__init__()
        
        # Basic initialization
        self.data_dir = Path(data_dir)
        self.vae = vae
        self.cache_dir = Path(cache_dir)
        self.no_caching_latents = no_caching_latents
        self.all_ar = all_ar
        
        # Use provided num_workers or system default
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        self.batch_size = 32  # Process images in batches
        
        # Resolution and bucketing parameters
        self.min_size = min_size
        self.max_size = max_size
        self.resolution_type = resolution_type
        self.enable_bucket_sampler = enable_bucket_sampler
        self.bucket_reso_steps = bucket_reso_steps
        
        # Tag weighting parameters
        self.min_tag_weight = min_tag_weight
        self.max_tag_weight = max_tag_weight
        self.use_tag_weighting = use_tag_weighting
        
        # VAE parameters
        self.finetune_vae = finetune_vae
        self.vae_learning_rate = vae_learning_rate
        self.vae_train_freq = vae_train_freq
        self.adaptive_loss_scale = adaptive_loss_scale
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        
        # Text augmentation parameters
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        
        # Model components
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # Initialize upscaler only if needed
        if not all_ar:
            self.upscaler = UltimateUpscaler(
                model_path="Lykon/DreamShaper",
                device=self.vae.device if self.vae else "cuda",
                dtype=self.vae.dtype if self.vae else torch.float32
            )
        else:
            self.upscaler = None
            logger.info("Skipping upscaler initialization since all_ar is enabled")
        
        # Performance optimization
        self.prefetch_factor = prefetch_factor
        
        # Create cache directory
        if not no_caching_latents:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize image paths
        self._initialize_dataset()
        
        # Initialize bucketing
        if self.enable_bucket_sampler:
            self._initialize_buckets()
            
        # Process latents and build tag statistics
        if not no_caching_latents:
            self._batch_process_latents_efficient()
        self.tag_stats = self._build_tag_statistics()
        
        # Set collate function
        self.collate_fn = self.custom_collate
        
        # Initialize tag weighter
        self.tag_weighter = TagBasedLossWeighter(
            min_weight=min_tag_weight,
            max_weight=max_tag_weight
        )
        
        # Initialize multiprocessing components
        self.process_pool = None
        self.task_queue = None
        self.result_queue = None
        self.workers = []
        self.cache_lock = threading.Lock()
        self.latent_cache = {}
        
        # Start worker processes
        self._initialize_workers()
        
    def _initialize_workers(self):
        """Initialize worker processes for parallel processing"""
        if self.process_pool is not None:
            return
            
        logger.info(f"Initializing {self.num_workers} worker processes for latent validation")
            
        # Create queues for task distribution
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        # Start worker processes
        self.process_pool = []
        for _ in range(self.num_workers):
            p = Process(target=self._worker_process, 
                       args=(self.task_queue, self.result_queue))
            p.daemon = True
            p.start()
            self.process_pool.append(p)
            
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
        """Process and cache latents in batches using multiple workers"""
        # Get list of uncached images
        uncached_images = []
        for img_path in self.image_paths:
            latent_path = self.cache_dir / f"{Path(img_path).stem}.pt"
            if not latent_path.exists():
                uncached_images.append(img_path)

        if not uncached_images:
            logger.info("All latents are already cached")
            return

        logger.info(f"Caching latents for {len(uncached_images)} images in batches")
        
        # Group images by size for more efficient batching
        size_groups = defaultdict(list)
        
        def group_image_by_size(img_path):
            try:
                caption_path = Path(img_path).with_suffix('.txt')
                if not caption_path.exists():
                    return None
                    
                with Image.open(img_path) as img:
                    width, height = img.size
                    if self.all_ar:
                        width = ((width + 7) // 8) * 8
                        height = ((height + 7) // 8) * 8
                    elif max(width, height) > 2048:
                        scale = 2048 / max(width, height)
                        width = int(width * scale)
                        height = int(height * scale)
                    return img_path, f"{width}x{height}"
            except Exception as e:
                logger.error(f"Error reading image {img_path}: {str(e)}")
                return None

        # Process images in parallel for size grouping
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in uncached_images:
                futures.append(executor.submit(group_image_by_size, img_path))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Grouping images"):
                result = future.result()
                if result:
                    img_path, size_key = result
                    size_groups[size_key].append(img_path)

        # Process each size group in parallel
        def process_size_group(size_key, paths):
            try:
                for i in range(0, len(paths), batch_size):
                    batch_paths = paths[i:i + batch_size]
                    self.process_batch_with_vae(batch_paths)
                return len(paths)
            except Exception as e:
                logger.error(f"Error processing size group {size_key}: {str(e)}")
                return 0

        # Use num_workers for size group processing, but don't exceed group count
        max_workers = min(len(size_groups), self.num_workers)
        total_processed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for size_key, paths in size_groups.items():
                futures.append(executor.submit(process_size_group, size_key, paths))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing size groups"):
                total_processed += future.result()

        logger.info(f"Successfully processed and cached latents for {total_processed} images")

    def process_batch_with_vae(self, batch_paths):
        """Process a batch of images through VAE with optimized memory usage"""
        if not batch_paths:
            return
        
        try:
            # Process images in parallel
            def load_and_transform_image(path):
                try:
                    with Image.open(path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        width, height = img.size
                        
                        # Get target size considering aspect ratio constraints
                        target_w, target_h = self._get_target_size(width, height)
                        
                        # Resize to target size
                        if width != target_w or height != target_h:
                            if width * height < target_w * target_h:
                                img = self._upscale_image(img, target_w, target_h)
                            else:
                                img = self._downscale_image(img, target_w, target_h)
                        
                        # Transform to tensor
                        image_tensor = transforms.ToTensor()(img)
                        image_tensor = transforms.Normalize([0.5], [0.5])(image_tensor)
                        return image_tensor, None
                except Exception as e:
                    return None, str(e)

            # Load images in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(load_and_transform_image, batch_paths))
            
            # Filter valid results
            valid_tensors = []
            valid_paths = []
            for path, (tensor, error) in zip(batch_paths, results):
                if tensor is not None:
                    valid_tensors.append(tensor)
                    valid_paths.append(path)
                elif error:
                    logger.error(f"Error processing {path}: {error}")

            if not valid_tensors:
                return

            # Process through VAE in chunks to manage memory
            chunk_size = 8  # Smaller chunks for better memory management
            for i in range(0, len(valid_tensors), chunk_size):
                chunk_tensors = valid_tensors[i:i + chunk_size]
                chunk_paths = valid_paths[i:i + chunk_size]
                
                # Generate latents
                with torch.no_grad():
                    image_tensor = torch.stack(chunk_tensors).to(self.vae.device, dtype=self.vae.dtype)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # Process text embeddings in parallel
                def process_text_embeddings(path):
                    try:
                        caption_path = Path(path).with_suffix('.txt')
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        # Generate embeddings
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
                            
                            # Reshape pooled output
                            pooled_output = pooled_output.unsqueeze(1).unsqueeze(2)
                            
                            return {
                                "text_embeddings": text_embeddings,
                                "text_embeddings_2": hidden_states,
                                "pooled_output": pooled_output
                            }
                    except Exception as e:
                        logger.error(f"Error processing text for {path}: {str(e)}")
                        return None

                # Process text embeddings in parallel
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    embedding_results = list(executor.map(process_text_embeddings, chunk_paths))

                # Save results
                for idx, (path, embeddings) in enumerate(zip(chunk_paths, embedding_results)):
                    if embeddings is not None:
                        cache_path = self.cache_dir / f"{Path(path).stem}_latents.pt"
                        try:
                            torch.save({
                                "latents": latents[idx].cpu(),
                                "text_embeddings": embeddings["text_embeddings"].cpu(),
                                "text_embeddings_2": embeddings["text_embeddings_2"].cpu(),
                                "added_cond_kwargs": {
                                    "text_embeds": embeddings["pooled_output"].cpu(),
                                    "time_ids": self._get_add_time_ids(image_tensor[idx:idx+1]).cpu()
                                }
                            }, cache_path)
                        except Exception as e:
                            logger.error(f"Error saving cache for {path}: {str(e)}")

                # Clear GPU memory
                del image_tensor, latents
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            logger.error(traceback.format_exc())

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

    def _parse_tags(self, caption):
        """Parse tags using tag weighter"""
        return self.tag_weighter.parse_tags(caption)

    def _format_caption(self, caption):
        """Format caption using tag weighter"""
        return self.tag_weighter.format_caption(caption)

    def _calculate_tag_weights(self, tags, special_tags):
        """Calculate tag weights using tag weighter"""
        return self.tag_weighter.calculate_weights(tags, special_tags)

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
            target_height = int(target_width * aspect_ratio)
        else:  # Landscape
            # Start with target height and calculate width to maintain AR
            target_height = 1024  # Base height for landscape
            target_width = int(target_height / aspect_ratio)

        # Scale down if needed while preserving AR
        max_dim = max(target_width, target_height)
        if max_dim > 2048:  # Only scale down if absolutely necessary
            scale = 2048 / max_dim
            target_width = int(target_width * scale)
            target_height = int(target_height * scale)

        return target_height, target_width

    def process_image_size(self, image, target_width, target_height):
        """Process image size with advanced resizing"""
        width, height = image.size
        
        # Get target size considering aspect ratio constraints
        target_width, target_height = self._get_target_size(target_width, target_height)
        
        # Resize to target size
        if width != target_width or height != target_height:
            if width * height < target_width * target_height:
                image = self._upscale_image(image, target_width, target_height)
            else:
                image = self._downscale_image(image, target_width, target_height)
                
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
        from functools import lru_cache
        
        logger = logging.getLogger(__name__)
        
        # Initialize temporary storage for image sizes
        image_sizes = []
        total_images = len(self.image_paths)
        logger.info(f"Analyzing {total_images} images using {self.num_workers} workers")
        
        # Cache target size calculations
        @lru_cache(maxsize=1024)
        def get_cached_target_size(width, height):
            return self._get_target_size(width, height)
        
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
                        target_h, target_w = get_cached_target_size(width, height)
                        if target_h and target_w:  # Ensure valid dimensions
                            results.append((target_h, target_w))
                except Exception as e:
                    logger.error(f"Error analyzing {img_path}: {str(e)}")
            return results
        
        # Process images in batches
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(analyze_image_batch, self.image_paths[i:i + batch_size]): i for i in range(0, total_images, batch_size)}
            
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
        
        # Use numpy for faster calculations
        sizes_array = np.array(image_sizes)
        aspect_ratios = sizes_array[:, 0] / sizes_array[:, 1]  # height/width ratios
        
        # Determine AR ranges for adaptive step sizes
        ar_percentiles = np.percentile(aspect_ratios, [5, 95])
        
        # Group similar sizes to create buckets, using numpy for efficiency
        size_groups = defaultdict(list)
        
        # Vectorized operations for bucket assignment
        for h, w in sizes_array:
            ar = h / w
            
            # Adaptive step sizes based on aspect ratio distribution
            if ar < ar_percentiles[0] or ar > ar_percentiles[1]:
                # Extreme aspect ratios get larger steps
                h_step = max(64, self.bucket_reso_steps * 2)
                w_step = max(64, self.bucket_reso_steps * 2)
            else:
                # Normal aspect ratios use standard steps
                h_step = self.bucket_reso_steps
                w_step = self.bucket_reso_steps
            
            # Round to nearest step while preserving aspect ratio
            if h > w:
                bucket_w = max(self.min_size, round(w / w_step) * w_step)
                bucket_h = max(self.min_size, round(h / h_step) * h_step)
            else:
                bucket_h = max(self.min_size, round(h / h_step) * h_step)
                bucket_w = max(self.min_size, round(w / w_step) * w_step)
            
            size_groups[(bucket_h, bucket_w)].append((h, w))
        
        # Filter out buckets with too few images to avoid memory fragmentation
        min_images_per_bucket = max(5, total_images // 1000)  # Adaptive threshold
        size_groups = {k: v for k, v in size_groups.items() if len(v) >= min_images_per_bucket}
        
        # Create buckets from groups
        self.buckets = sorted(size_groups.keys(), key=lambda x: x[0] * x[1])
        self.bucket_data = {bucket: [] for bucket in self.buckets}
        
        # Log bucket statistics
        logger.info(f"Created {len(self.buckets)} dynamic buckets")
        logger.info(f"Aspect ratio range: {ar_percentiles[0]:.2f} to {ar_percentiles[1]:.2f}")
        
        # Prepare lookup arrays for faster matching
        bucket_arrays = np.array(self.buckets)
        bucket_ars = bucket_arrays[:, 0] / bucket_arrays[:, 1]
        bucket_areas = bucket_arrays[:, 0] * bucket_arrays[:, 1]
        
        def process_image_batch(img_paths):
            results = []
            for img_path in img_paths:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                        # Calculate target size first
                        target_h, target_w = get_cached_target_size(width, height)
                        target_area = target_h * target_w
                        
                        # Vectorized bucket matching
                        target_ar = target_h / target_w
                        area_scores = np.abs(bucket_areas - target_area) / target_area
                        
                        # Combined score with more weight on aspect ratio
                        scores = np.abs(np.log(bucket_ars / target_ar))  # Log scale for better AR comparison
                        
                        # Find best bucket that's large enough
                        valid_buckets = (bucket_arrays[:, 0] >= target_h) & (bucket_arrays[:, 1] >= target_w)
                        if not np.any(valid_buckets):
                            continue
                            
                        scores[~valid_buckets] = np.inf
                        best_idx = np.argmin(scores)
                        best_bucket = self.buckets[best_idx]
                        
                        results.append((img_path, best_bucket))
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
            return results
        
        # Process final assignment in batches
        logger.info(f"Assigning {total_images} images to buckets using {self.num_workers} workers...")
        
        # Process in parallel with specified number of workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, total_images, batch_size):
                batch = self.image_paths[i:i + batch_size]
                futures.append(executor.submit(process_image_batch, batch))
            
            # Process results as they come in
            for i, future in enumerate(futures):
                try:
                    batch_results = future.result()
                    for img_path, bucket in batch_results:
                        self.bucket_data[bucket].append(img_path)
                    
                    processed = min((i + 1) * batch_size, total_images)
                    if processed % 5000 == 0:
                        logger.info(f"Assigned {processed}/{total_images} images to buckets")
                except Exception as e:
                    logger.error(f"Error in batch {i}: {str(e)}")
        
        # Clean up empty buckets
        empty_buckets = [bucket for bucket, images in self.bucket_data.items() if not images]
        for bucket in empty_buckets:
            del self.bucket_data[bucket]
            self.buckets.remove(bucket)
        
        # Final statistics
        num_buckets = len(self.bucket_data)
        total_assigned = sum(len(imgs) for imgs in self.bucket_data.values())
        logger.info(f"Final bucket count: {num_buckets} with {total_assigned}/{total_images} images assigned")
        
        # Log distribution
        bucket_sizes = {bucket: len(imgs) for bucket, imgs in self.bucket_data.items()}
        top_buckets = sorted(bucket_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 bucket sizes:")
        for (h, w), count in top_buckets:
            aspect_ratio = f"{w/math.gcd(w,h)}:{h/math.gcd(w,h)}"
            logger.info(f"  {w}x{h} ({aspect_ratio}): {count} images")

    def _assign_to_bucket(self, img_path):
        """Assign an image to the most appropriate bucket with optimized comparison"""
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Calculate target size first
                target_h, target_w = self._get_target_size(width, height)
                
                # Find the best fitting bucket using area-based comparison
                best_bucket = None
                min_area_diff = float('inf')
                
                for bucket_h, bucket_w in self.buckets:
                    # Skip if bucket is too small
                    if bucket_h < target_h or bucket_w < target_w:
                        continue
                    
                    # Calculate area difference
                    area_diff = abs((bucket_h * bucket_w) - (target_h * target_w))
                    if area_diff < min_area_diff:
                        min_area_diff = area_diff
                        best_bucket = (bucket_h, bucket_w)
            
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
        # Load and process image
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            
            # Get bucket dimensions
            bucket_h, bucket_w = self._get_bucket_size(image.size)
            
            # Process image with advanced resizing
            processed_image = self._advanced_resize(image, bucket_w, bucket_h)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image_tensor = transform(processed_image).unsqueeze(0)
            
            # Generate VAE latents
            with torch.no_grad():
                image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.squeeze(0)
            
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
            
            # Load cached latents if available
            if not self.no_caching_latents and self.latent_paths[idx].exists():
                cache_data_latents = torch.load(self.latent_paths[idx], map_location='cpu')  # Load to CPU first
                # Ensure tensors are on CPU and in float32
                cache_data['latents'] = cache_data_latents['latents'].float().cpu()
            
            return cache_data
        
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            return None


class BucketSampler(Sampler):
    """
    Sampler that creates batches of samples from the same bucket to ensure consistent tensor sizes.
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__(dataset)
        if not hasattr(dataset, 'bucket_data'):
            raise ValueError("Dataset must have 'bucket_data' attribute for BucketSampler")
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Store (bucket, index) pairs for all valid samples
        self.samples = []
        for bucket, img_paths in dataset.bucket_data.items():
            for img_path in img_paths:
                idx = dataset.image_paths.index(img_path)
                self.samples.append((bucket, idx))
                
        if not self.samples:
            raise ValueError("No valid samples found in dataset buckets")
        
        # Pre-validate all buckets
        self._validate_buckets(dataset)
    
    def _validate_buckets(self, dataset):
        """Validate bucket configurations to prevent runtime errors"""
        bucket_sizes = {}
        for bucket, img_paths in dataset.bucket_data.items():
            if not img_paths:  # Skip empty buckets
                continue
            # Sample an image from each bucket to verify size
            sample_path = img_paths[0]
            try:
                idx = dataset.image_paths.index(sample_path)
                sample = dataset[idx]
                if sample is None:
                    raise ValueError(f"Invalid sample at index {idx} in bucket {bucket}")
                bucket_sizes[bucket] = sample['bucket_size']
            except Exception as e:
                raise ValueError(f"Error validating bucket {bucket}: {str(e)}")
    
    def __iter__(self):
        """Return batches of indices, where each batch contains images from the same bucket"""
        if not self.samples:
            raise RuntimeError("No samples available for iteration")
            
        # Group samples by bucket
        bucket_samples = defaultdict(list)
        for bucket, idx in self.samples:
            bucket_samples[bucket].append(idx)
        
        # Create complete batches from each bucket
        batches = []
        for bucket, indices in bucket_samples.items():
            # Shuffle indices within the bucket
            indices = indices.copy()  # Create a copy to prevent modifying original
            random.shuffle(indices)
            
            # Create full batches
            complete_batches = len(indices) // self.batch_size
            for i in range(complete_batches):
                start_idx = i * self.batch_size
                batch = indices[start_idx:start_idx + self.batch_size]
                batches.append(batch)
            
            # Handle remaining samples if not dropping last
            if not self.drop_last and len(indices) % self.batch_size > 0:
                last_batch = indices[complete_batches * self.batch_size:]
                if len(last_batch) >= self.batch_size // 2:  # Only keep if reasonable size
                    batches.append(last_batch)
        
        if not batches:
            raise RuntimeError("No valid batches could be created")
            
        # Shuffle the batches themselves
        random.shuffle(batches)
        
        # Flatten batches into single list of indices
        indices = []
        for batch in batches:
            indices.extend(batch)
            
        return iter(indices)
    
    def __len__(self):
        if self.drop_last:
            return len(self.samples) - (len(self.samples) % self.batch_size)
        return len(self.samples)


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
    **kwargs
):
    """Create a DataLoader with bucketing support"""
    # Initialize dataset
    dataset = CustomDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        vae=vae,
        num_workers=num_workers,  # Pass through num_workers
        enable_bucket_sampler=enable_bucket_sampler,
        **kwargs
    )

    if enable_bucket_sampler and hasattr(dataset, 'bucket_data'):
        sampler = BucketSampler(dataset, batch_size)
        shuffle = False  # Bucket sampler handles shuffling
    else:
        sampler = None
        shuffle = True

    # Create DataLoader with the same num_workers as dataset
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataset.num_workers,  # Use dataset's num_workers for consistency
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
