import logging
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union, Any
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from .bucket_manager import BucketManager

from ..tag_weighter import TagBasedLossWeighter
from .base import CustomDatasetBase

logger = logging.getLogger(__name__)

class CustomDataset(CustomDatasetBase):
    """Custom dataset implementation for SDXL training with advanced features.

    This dataset handles image-caption pairs for SDXL training with support for:
    - Efficient latent caching and processing
    - Dynamic resolution bucketing (following NovelAI paper)
    - Tag-based loss weighting
    - Parallel data processing
    - Multi-worker data loading
    - Memory-efficient batch processing

    Args:
        data_dir (str): Directory containing image-caption pairs
        vae (nn.Module, optional): VAE model for latent computation. Defaults to None.
        tokenizer (transformers.PreTrainedTokenizer, optional): Primary tokenizer. Defaults to None.
        tokenizer_2 (transformers.PreTrainedTokenizer, optional): Secondary tokenizer. Defaults to None.
        text_encoder (nn.Module, optional): Primary text encoder. Defaults to None.
        text_encoder_2 (nn.Module, optional): Secondary text encoder. Defaults to None.
        cache_dir (str, optional): Directory for caching latents. Defaults to "latents_cache".
        no_caching_latents (bool, optional): Disable latent caching. Defaults to False.
        all_ar (bool, optional): Use aspect ratio for all images. Defaults to False.
        num_workers (int, optional): Number of worker processes. Defaults to min(8, cpu_count).
        prefetch_factor (int, optional): Number of batches to prefetch. Defaults to 2.
        min_size (int, optional): Minimum image size. Defaults to 512.
        max_size (int, optional): Maximum image size. Defaults to 2048.
        bucket_step_size (int, optional): Resolution step size for buckets. Defaults to 64.
        max_bucket_area (int, optional): Maximum area for buckets. Defaults to 1024*1024.
        token_dropout_rate (float, optional): Token dropout probability. Defaults to 0.1.
        caption_dropout_rate (float, optional): Caption dropout probability. Defaults to 0.1.
        min_tag_weight (float, optional): Minimum weight for tags. Defaults to 0.1.
        max_tag_weight (float, optional): Maximum weight for tags. Defaults to 3.0.
        use_tag_weighting (bool, optional): Enable tag-based loss weighting. Defaults to True.
    """
    def __init__(self, data_dir, vae=None, tokenizer=None, tokenizer_2=None,
                 text_encoder=None, text_encoder_2=None,
                 cache_dir="latents_cache", no_caching_latents=False,
                 all_ar=False, num_workers=None, prefetch_factor=2,
                 min_size=512, max_size=2048, bucket_step_size=64,
                 max_bucket_area=1024*1024, token_dropout_rate=0.1,
                 caption_dropout_rate=0.1, min_tag_weight=0.1,
                 max_tag_weight=3.0, use_tag_weighting=True):
        super().__init__()
        # Store models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        
        # Set device based on CUDA availability and VAE device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.vae is not None:
            self.device = next(self.vae.parameters()).device
        
        # Initialize flags first before any processing
        self.collate_fn = self.custom_collate
        self.all_ar = all_ar
        self.no_caching_latents = no_caching_latents
        self.min_size = min_size
        self.max_size = max_size
        self.bucket_step_size = bucket_step_size
        self.max_bucket_area = max_bucket_area
        self.use_tag_weighting = use_tag_weighting
        
        # Initialize bucket manager
        self.bucket_manager = BucketManager(
            min_size=min_size,
            max_size=max_size,
            step_size=bucket_step_size,
            max_area=max_bucket_area,
            add_square=True
        )
        
        # Set up workers
        self.num_workers = num_workers or min(8, os.cpu_count() or 1)
        self.prefetch_factor = prefetch_factor
        
        # Initialize latents cache
        self.latents_cache = {}
        
        # Initialize tag weight cache
        self.tag_weight_cache = {}
        
        # Store initialization parameters for workers
        self.init_params = {
            'vae': vae,
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
            'text_encoder': text_encoder,
            'text_encoder_2': text_encoder_2,
            'cache_dir': cache_dir,
            'no_caching_latents': no_caching_latents,
            'all_ar': all_ar,
            'num_workers': self.num_workers,
            'prefetch_factor': prefetch_factor,
            'min_size': min_size,
            'max_size': max_size,
            'bucket_step_size': bucket_step_size,
            'max_bucket_area': max_bucket_area,
            'token_dropout_rate': token_dropout_rate,
            'caption_dropout_rate': caption_dropout_rate,
            'min_tag_weight': min_tag_weight,
            'max_tag_weight': max_tag_weight,
            'use_tag_weighting': use_tag_weighting
        }
        
        # Basic initialization
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize multiprocessing components
        if not no_caching_latents:
            self.process_pool = None
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()
            self.workers = []
            
        # Initialize worker models
        self.worker_models = None
            
        # Initialize tag weighter if needed
        if use_tag_weighting:
            self.tag_weighter = TagBasedLossWeighter(
                config={
                    'min_weight': min_tag_weight,
                    'max_weight': max_tag_weight,
                    'no_cache': no_caching_latents
                }
            )
        else:
            self.tag_weighter = None
            
        # Convert image paths to strings once and validate caption files
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
        all_image_paths = []
        for ext in image_extensions:
            all_image_paths.extend([str(p) for p in Path(data_dir).glob(ext)])
            all_image_paths.extend([str(p) for p in Path(data_dir).glob(ext.upper())])

        if not all_image_paths:
            raise RuntimeError(f"No images found in {data_dir}. Supported formats: {', '.join(image_extensions)}")

        # Validate image-caption pairs and only keep valid ones
        self.image_paths = []
        skipped_count = 0
        
        for img_path in all_image_paths:
            caption_path = Path(img_path).with_suffix('.txt')
            try:
                # Verify the image can be opened
                with Image.open(img_path) as img:
                    img.verify()
                # Verify the caption file exists and is not empty
                if caption_path.exists():
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    if caption:  # Only add if caption is not empty
                        self.image_paths.append(img_path)
                        continue
            except Exception as e:
                logger.debug(f"Skipping {img_path}: {str(e)}")
            
            skipped_count += 1
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} images due to missing or invalid caption files")
        
        if not self.image_paths:
            raise RuntimeError(f"No valid image-caption pairs found in {data_dir}. Found {len(all_image_paths)} images but none had valid caption files.")
        
        logger.info(f"Found {len(self.image_paths)} valid image-caption pairs out of {len(all_image_paths)} total images")
        
        # Initialize dataset structure
        self.bucket_data = defaultdict(list)  # Initialize bucket_data in __init__
        self._initialize_dataset()
        
    def _initialize_dataset(self):
        """Initialize dataset structure with proper error handling"""
        try:
            # Pre-compute bucket data efficiently using ThreadPoolExecutor
            if self.init_params['all_ar']:
                logger.info("Pre-computing bucket data...")
                self._precompute_bucket_data()
                
            # Pre-compute tag weights if enabled
            if self.use_tag_weighting:
                logger.info("Pre-computing tag weights...")
                self._precompute_tag_weights()
                
            # Initialize workers and process latents only if caching is enabled
            if not self.init_params['no_caching_latents']:
                logger.info("Pre-computing latents...")
                self._initialize_workers()
                self._batch_process_latents_efficient()
            else:
                logger.info("Latent caching disabled, will process latents on-the-fly")
                
        except Exception as e:
            logger.error("Dataset initialization failed: %s", str(e))
            logger.error(traceback.format_exc())
            self.cleanup()
            raise

    def _precompute_bucket_data(self):
        """Pre-compute bucket data using thread pool"""
        def process_image_batch(paths):
            results = []
            for path in paths:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        results.append((path, height, width))
                except Image.UnidentifiedImageError as e:
                    logger.error("Invalid or corrupted image file %s: %s", path, str(e))
                except Image.DecompressionBombError as e:
                    logger.error("Image %s is too large to process: %s", path, str(e))
                except ValueError as e:
                    logger.error("Invalid image format in %s: %s", path, str(e))
                except (IOError, OSError) as e:
                    logger.error("Failed to read image file %s: %s", path, str(e))
                except MemoryError as e:
                    logger.error("Out of memory while processing %s: %s", path, str(e))
                    logger.error(traceback.format_exc())
                except (AttributeError, TypeError) as e:
                    logger.error("Malformed image data in %s: %s", path, str(e))
                    logger.error(traceback.format_exc())
            return results
            
        # Process in parallel using ThreadPoolExecutor
        chunk_size = max(1, len(self.image_paths) // (self.init_params['num_workers'] * 4))
        # Clear existing bucket data before recomputing
        self.bucket_data.clear()
        
        # Collect image dimensions
        image_dimensions = []
        with ThreadPoolExecutor(max_workers=self.init_params['num_workers']) as executor:
            futures = []
            for i in range(0, len(self.image_paths), chunk_size):
                chunk = self.image_paths[i:i + chunk_size]
                futures.append(executor.submit(process_image_batch, chunk))
                
            # Combine results
            for future in tqdm(futures, desc="Processing image dimensions"):
                for path, height, width in future.result():
                    image_dimensions.append((height, width))
                    
        # Assign images to buckets using bucket manager
        bucket_assignments = self.bucket_manager.assign_to_buckets(image_dimensions)
        
        # Convert assignments to paths
        for bucket, indices in bucket_assignments.items():
            self.bucket_data[bucket] = [self.image_paths[i] for i in indices]
            
        # Log bucket statistics
        total_images = sum(len(paths) for paths in self.bucket_data.values())
        logger.info(f"Assigned {total_images} images to {len(self.bucket_data)} buckets")
        for bucket, paths in self.bucket_data.items():
            logger.info(f"Bucket {bucket}: {len(paths)} images")
            
    def _precompute_tag_weights(self):
        """Pre-compute tag weights for all captions using thread pool"""
        if not self.use_tag_weighting or not hasattr(self, 'tag_weighter'):
            logger.warning("Tag weighting is disabled or tag weighter not initialized")
            return
            
        # Ensure tag_weight_cache is initialized
        if self.tag_weight_cache is None:
            self.tag_weight_cache = {}
            
        def process_caption_batch(paths):
            results = {}
            for path in paths:
                try:
                    caption_path = Path(path).with_suffix('.txt')
                    if not caption_path.exists():
                        logger.debug(f"Caption file not found for {path}, using default weight")
                        results[path] = 1.0
                        continue

                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    if not caption:
                        logger.debug(f"Empty caption file for {path}, using default weight")
                        results[path] = 1.0
                        continue

                    # Split caption into tags and calculate weights for each
                    tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
                    if not tags:
                        logger.debug(f"No valid tags found in caption for {path}, using default weight")
                        results[path] = 1.0
                        continue

                    tag_weights = [self.tag_weighter.calculate_tag_weight(tag) for tag in tags]
                    # Use mean of tag weights as the overall weight
                    weight = sum(tag_weights) / len(tag_weights) if tag_weights else 1.0
                    results[path] = weight

                except Exception as e:
                    logger.debug(f"Failed to calculate tag weight for {path}: {str(e)}, using default weight")
                    results[path] = 1.0  # Default weight on error
            return results

        # Process in parallel using ThreadPoolExecutor
        chunk_size = max(1, len(self.image_paths) // (self.init_params['num_workers'] * 4))
        processed_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=self.init_params['num_workers']) as executor:
            futures = []
            for i in range(0, len(self.image_paths), chunk_size):
                chunk = self.image_paths[i:i + chunk_size]
                futures.append(executor.submit(process_caption_batch, chunk))
                
            # Combine results
            for future in tqdm(futures, desc="Processing tag weights"):
                try:
                    results = future.result()
                    self.tag_weight_cache.update(results)
                    processed_count += len(results)
                except Exception as e:
                    logger.error(f"Failed to process tag weight batch: {str(e)}")
                    error_count += 1
                    continue

        if error_count > 0:
            logger.warning(f"{error_count} tag weight batches failed to process")
        logger.info(f"Successfully processed tag weights for {processed_count} images")

    def write_captions(self, formatted_paths):
        """Write formatted captions using thread pool"""
        def write_caption(path_caption):
            path, caption = path_caption
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(caption)
            except PermissionError as e:
                logger.error("Permission denied writing to %s: %s", path, str(e))
            except (IOError, OSError) as e:
                logger.error("File I/O error writing to %s: %s", path, str(e))
            except UnicodeEncodeError as e:
                logger.error("Unicode encoding error writing to %s: %s", path, str(e))
                
        with ThreadPoolExecutor(max_workers=self.init_params['num_workers']) as executor:
            list(executor.map(write_caption, formatted_paths))

    def _initialize_worker(self):
        """Initialize worker process state"""
        if torch.cuda.is_available():
            device = f'cuda:{torch.cuda.current_device()}'
        else:
            device = 'cpu'
            
        # Initialize models in worker process
        self.worker_models = {
            'vae': self.init_params['vae'].to(device),
            'text_encoder': self.init_params['text_encoder'].to(device),
            'text_encoder_2': self.init_params['text_encoder_2'].to(device),
            'tokenizer': self.init_params['tokenizer'],
            'tokenizer_2': self.init_params['tokenizer_2']
        }

    def _worker_process(self, task_queue, result_queue):
        """Worker process function"""
        try:
            self._initialize_worker()
            
            while True:
                task = task_queue.get()
                if task is None:  # Poison pill
                    break
                    
                try:
                    result = self._process_task(task)
                    result_queue.put(result)
                except Exception as e:
                    logger.error("Error processing task: %s", str(e))
                    result_queue.put(None)
                    
        except Exception as e:
            logger.error("Worker process error: %s", str(e))
            logger.error(traceback.format_exc())
        finally:
            # Cleanup worker resources
            if hasattr(self, 'worker_models'):
                del self.worker_models

    def _process_task(self, task):
        """Process a task in the worker process
        
        Args:
            task: Dictionary containing task information
                - 'type': Type of task ('process_latent', etc.)
                - 'data': Data needed for the task
                
        Returns:
            Result of the task processing
        """
        if task['type'] == 'process_latent':
            return self._process_single_latent(task['data'])
        else:
            logger.warning("Unknown task type: %s", task['type'])
            return None

    def _parse_tags(self, caption):
        """Optimized tag parsing with caching"""
        from functools import lru_cache
        
        @lru_cache(maxsize=10000)
        def cached_parse_tags(caption_text):
            """Cache tag parsing results for repeated captions"""
            if not self.tag_weighter:
                return [], {}
            # Split caption into tags and process them
            tags = [tag.strip() for tag in caption_text.split(",") if tag.strip()]
            return tags, self.tag_weighter.process_tag_batch(tags)
        
        if not caption:
            return [], {}
            
        return cached_parse_tags(caption)
    
    def parse_tags_batch(self, captions, num_workers=None):
        """Parallel tag parsing for multiple captions"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if not captions:
            return [], {}
            
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        def process_batch(caption_batch):
            return [self._parse_tags(caption) for caption in caption_batch]
            
        # Process in parallel for large batches
        if len(captions) > 100:
            batch_size = max(50, len(captions) // (num_workers * 4))
            batches = np.array_split(captions, max(1, len(captions) // batch_size))
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                results = [r for f in futures for r in f.result()]
            return results
            
        # Process sequentially for small batches
        return process_batch(captions)

    def _format_caption(self, caption):
        """Optimized caption formatting with caching"""
        from functools import lru_cache
        
        @lru_cache(maxsize=10000)
        def cached_format_caption(caption_text):
            """Cache formatted captions"""
            if not caption_text:
                return ""
            if not self.tag_weighter:
                return caption_text
                
            try:
                return self.tag_weighter.format_caption(caption_text)
            except ValueError as e:
                logger.error("Invalid caption format: %s", str(e))
                return caption_text
            except AttributeError as e:
                logger.error("Tag weighter not properly initialized: %s", str(e))
                return caption_text
            except TypeError as e:
                logger.error("Invalid caption type: %s", str(e))
                return caption_text
                
        return cached_format_caption(caption)
    
    def format_captions_batch(self, captions, num_workers=None):
        """Parallel caption formatting for multiple captions"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if not captions:
            return []
            
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        def process_batch(caption_batch):
            return [self._format_caption(caption) for caption in caption_batch]
            
        # Process in parallel for large batches
        if len(captions) > 100:
            batch_size = max(50, len(captions) // (num_workers * 4))
            batches = np.array_split(captions, max(1, len(captions) // batch_size))
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                results = [r for f in futures for r in f.result()]
            return results
            
        # Process sequentially for small batches
        return process_batch(captions)

    def _calculate_tag_weights(self, tags, special_tags):
        """Optimized tag weight calculation with caching"""
        from functools import lru_cache
        
        @lru_cache(maxsize=1024)
        def cached_calculate_weights(tags_tuple, special_tags_tuple):
            # Convert special tags back to regular tags with weights
            weighted_tags = list(tags_tuple)
            for tag, weight in special_tags_tuple:
                if isinstance(weight, (int, float)):
                    weighted_tags.append(f"{tag}::{weight}")
            # Process tags in batch to get their weights
            return self.tag_weighter.process_tag_batch(weighted_tags)
        
        # Convert inputs to hashable types for caching
        tags_tuple = tuple(sorted(tags))
        # Ensure special_tags is a dictionary before calling items()
        if not isinstance(special_tags, dict):
            special_tags = {tag: 1.0 for tag in special_tags}
        special_tags_tuple = tuple(sorted(special_tags.items()))
        
        return cached_calculate_weights(tags_tuple, special_tags_tuple)
    
    def calculate_weights_batch(self, tag_pairs, num_workers=None):
        """Parallel weight calculation for multiple tag pairs"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if not tag_pairs:
            return []
            
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        def process_batch(pairs_batch):
            return [self._calculate_tag_weights(tags, special_tags) 
                   for tags, special_tags in pairs_batch]
            
        # Process in parallel for large batches
        if len(tag_pairs) > 100:
            batch_size = max(50, len(tag_pairs) // (num_workers * 4))
            batches = np.array_split(tag_pairs, max(1, len(tag_pairs) // batch_size))
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                results = [r for f in futures for r in f.result()]
            return results
            
        # Process sequentially for small batches
        return process_batch(tag_pairs)

    def _initialize_workers(self) -> None:
        """Initialize worker processes for parallel latent computation"""
        if not self.no_caching_latents:
            # Use ProcessPoolExecutor instead of multiprocessing for better GPU handling
            self.process_pool = ThreadPoolExecutor(max_workers=1)  # Single worker for GPU
            logger.info("Initialized latent processing worker")

    def _process_single_latent(self, image_path: str) -> Optional[Tuple[str, torch.Tensor]]:
        """Process a single image into latent space"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(
                np.array(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            
            # Move to GPU and process through VAE encoder
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image_tensor = image_tensor.to(self.device)
                    latent = self.vae.encode(image_tensor).latent_dist.sample()
                    latent = latent.cpu()  # Move back to CPU for storage
                    
            # Clean up GPU memory
            del image_tensor
            torch.cuda.empty_cache()
                
            return image_path, latent
            
        except Exception as e:
            logger.error("Failed to process latent for %s: %s", image_path, str(e))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def _batch_process_latents_efficient(self):
        """Process latents in batches efficiently using worker pool"""
        try:
            total_images = len(self.image_paths)
            batch_size = 32  # Process larger batches for GPU efficiency
            num_batches = (total_images + batch_size - 1) // batch_size
            
            logger.info("Processing %d images in %d batches", total_images, num_batches)
            
            for batch_idx in tqdm(range(num_batches), desc="Processing latents"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_images)
                batch_paths = self.image_paths[start_idx:end_idx]
                
                # Process batch of images
                try:
                    # Load and preprocess all images in batch
                    images = []
                    valid_paths = []
                    
                    for image_path in batch_paths:
                        try:
                            with Image.open(image_path) as img:
                                img = img.convert('RGB')
                                # Get original dimensions
                                orig_height, orig_width = img.size[1], img.size[0]
                                
                                # Find closest bucket resolution
                                target_height, target_width = self.bucket_manager.find_closest_bucket(orig_height, orig_width)
                                
                                # Resize image to target bucket resolution
                                if (orig_height, orig_width) != (target_height, target_width):
                                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                                
                                # Optimize image processing for GPU
                                image = torch.from_numpy(
                                    np.array(img, dtype=np.float32)
                                ).permute(2, 0, 1) / 127.5 - 1
                                images.append(image)
                                valid_paths.append(image_path)
                        except Exception as e:
                            logger.debug(f"Failed to load/resize image {image_path}: {str(e)}")
                            continue
                    
                    if not images:
                        continue
                        
                    # Stack images into a single batch tensor
                    image_batch = torch.stack(images, dim=0)
                    
                    # Process batch through VAE with mixed precision
                    with torch.amp.autocast('cuda'):
                        with torch.no_grad():
                            image_batch = image_batch.to(self.device, non_blocking=True)
                            latents = self.vae.encode(image_batch).latent_dist.sample()
                            latents = latents.cpu()  # Move to CPU immediately
                            
                    # Store latents in cache
                    for path, latent in zip(valid_paths, latents):
                        self.latents_cache[path] = latent
                    
                    # Clean up GPU memory
                    del image_batch, latents, images
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_idx}: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            logger.info("Completed latent processing")
            
        except Exception as e:
            logger.error("Failed to process latents: %s", str(e))
            logger.error(traceback.format_exc())
            raise

    def custom_collate(self, batch):
        """Custom collate function for DataLoader to handle batched data.
        
        Args:
            batch: List of data items from dataset
            
        Returns:
            Dict containing batched tensors and metadata
        """
        batch_dict = {
            'latents': [],
            'captions': [],
            'tag_weights': [],
            'metadata': []
        }
        
        for item in batch:
            batch_dict['latents'].append(item.get('latent'))
            batch_dict['captions'].append(item.get('caption', ''))
            batch_dict['tag_weights'].append(item.get('tag_weight', 1.0))
            batch_dict['metadata'].append(item.get('metadata', {}))
            
        # Stack tensors and convert to appropriate formats
        batch_dict['latents'] = torch.stack(batch_dict['latents'])
        batch_dict['tag_weights'] = torch.tensor(batch_dict['tag_weights'], dtype=torch.float32)
        
        return batch_dict

    def __len__(self) -> int:
        """Return the total number of items in the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: Union[int, slice]) -> Any:
        """Get a single item or slice of items from the dataset"""
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        # Add safety counter to prevent infinite loops
        max_attempts = len(self)
        attempts = 0
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self)
            attempts += 1
            
            image_path = self.image_paths[current_idx]
            
            # Get caption and process it first (faster to check)
            caption_path = Path(image_path).with_suffix('.txt')
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                if not caption:
                    continue
            except (IOError, OSError) as e:
                logger.debug(f"Failed to read caption file %s: %s", caption_path, str(e))
                continue
            
            # Always get tag weight from cache since tag caching is always enabled
            tag_weight = self.tag_weight_cache.get(image_path, 1.0) if self.use_tag_weighting else 1.0
            
            # Process latent based on caching setting
            latent = None
            if not self.no_caching_latents:
                # Try to get from cache first if caching is enabled
                latent = self.latents_cache.get(image_path)
            
            if latent is None:
                # Process latent on-the-fly if not in cache or caching is disabled
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        
                        # Get original dimensions
                        orig_height, orig_width = img.size[1], img.size[0]
                        
                        # Find closest bucket resolution
                        target_height, target_width = self.bucket_manager.find_closest_bucket(orig_height, orig_width)
                        
                        # Resize image to target bucket resolution
                        if (orig_height, orig_width) != (target_height, target_width):
                            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        # Optimize image processing for GPU
                        image = torch.from_numpy(
                            np.array(img, dtype=np.float32)
                        ).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
                        
                        # Move to GPU and use automatic mixed precision
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                image = image.to(self.device, non_blocking=True)
                                latent = self.vae.encode(image).latent_dist.sample()
                                
                                if not self.no_caching_latents:
                                    # Only cache if caching is enabled
                                    self.latents_cache[image_path] = latent.cpu()
                                    latent = latent.cpu()  # Move to CPU for storage
                                
                        # Clean up GPU memory
                        del image
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.debug("Failed to process image %s: %s", image_path, str(e))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            # Ensure latent is on CPU for return
            if latent.device.type == 'cuda':
                latent = latent.cpu()
                
            return {
                'latent': latent,
                'caption': caption,
                'tag_weight': tag_weight,
                'metadata': {
                    'image_path': image_path,
                    'caption_path': str(caption_path)
                }
            }
            
        raise RuntimeError(
            f"No valid items found in dataset after trying {max_attempts} items. "
            "Check that your dataset contains valid images and captions."
        )