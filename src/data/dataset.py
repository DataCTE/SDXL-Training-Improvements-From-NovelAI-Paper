from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import traceback
import random
from threading import Thread
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
import itertools
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Sequence, Union, Dict
from torch.utils.data import Dataset, Sampler, DataLoader
from queue import Queue

from queue import Queue
import threading
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Sampler
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Create our own base classes
class CustomDatasetBase(ABC):
    """Abstract base class for custom datasets with enhanced functionality"""
    
    def __init__(self):
        self._length: Optional[int] = None
        self._initialized: bool = False
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset"""
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx: Union[int, slice]) -> Any:
        """Get a single item or slice of items from the dataset"""
        raise NotImplementedError
    
    def initialize(self) -> None:
        """Optional initialization method for lazy loading"""
        self._initialized = True
    
    @property
    def is_initialized(self) -> bool:
        """Check if dataset has been initialized"""
        return self._initialized
    
    def get_batch(self, indices: Sequence[int]) -> list:
        """Efficiently get multiple items at once"""
        return [self[idx] for idx in indices]
    
    def prefetch(self, indices: Sequence[int]) -> None:
        """Optional prefetch method for optimization"""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method"""
        pass


class CustomSamplerBase(ABC):
    """Abstract base class for custom samplers with enhanced functionality"""
    
    def __init__(self, data_source: CustomDatasetBase):
        self.data_source = data_source
        self._iterator: Optional[Iterator] = None
        self._epoch: int = 0
    
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Return iterator over dataset indices"""
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the sampler"""
        raise NotImplementedError
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducibility"""
        self._epoch = epoch
    
    @property
    def epoch(self) -> int:
        """Get current epoch"""
        return self._epoch
    
    def state_dict(self) -> Dict:
        """Get sampler state for checkpointing"""
        return {
            'epoch': self._epoch,
            'iterator_state': getattr(self._iterator, 'state', None)
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load sampler state from checkpoint"""
        self._epoch = state_dict.get('epoch', 0)
        if hasattr(self._iterator, 'load_state'):
            self._iterator.load_state(state_dict.get('iterator_state'))


class CustomDataLoaderBase(ABC):
    """Abstract base class for custom data loaders with enhanced functionality"""
    
    def __init__(self, 
                 dataset: CustomDatasetBase,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 sampler: Optional[CustomSamplerBase] = None,
                 batch_sampler: Optional[Sampler] = None,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: Optional[callable] = None,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Initialize batch_sampler and sampler
        self.batch_sampler = None
        self.sampler = None
        
        # Handle sampler configuration
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with other parameters')
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            self.sampler = sampler
            
            # Create batch sampler if not provided
            self.batch_sampler = torch.utils.data.BatchSampler(
                self.sampler,
                batch_size=batch_size,
                drop_last=drop_last
            )
            
        self._initialized = False
        self._iterator = None
    
    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return iterator over the dataset"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of batches in the dataloader"""
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    def initialize(self) -> None:
        """Initialize the dataloader and its dataset"""
        if not self._initialized:
            if not self.dataset.is_initialized:
                self.dataset.initialize()
            self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._iterator is not None:
            del self._iterator
            self._iterator = None
        self.dataset.cleanup()
    
    def state_dict(self) -> Dict:
        """Get dataloader state for checkpointing"""
        return {
            'initialized': self._initialized,
            'sampler_state': self.sampler.state_dict() if hasattr(self.sampler, 'state_dict') else None
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load dataloader state from checkpoint"""
        self._initialized = state_dict.get('initialized', False)
        if hasattr(self.sampler, 'load_state_dict'):
            self.sampler.load_state_dict(state_dict.get('sampler_state', {}))


# Then modify your existing classes to use these bases
class CustomDataset(CustomDatasetBase):
    def __init__(self, data_dir, vae=None, tokenizer=None, tokenizer_2=None, 
                 text_encoder=None, text_encoder_2=None,
                 cache_dir="latents_cache", no_caching_latents=False, 
                 all_ar=False, num_workers=None, prefetch_factor=2,
                 resolution_type="square", enable_bucket_sampler=True,
                 min_size=512, max_size=2048, bucket_reso_steps=64,
                 token_dropout_rate=0.1, caption_dropout_rate=0.1,
                 min_tag_weight=0.1, max_tag_weight=3.0, 
                 use_tag_weighting=True, **kwargs):
        super().__init__()
        
        # Initialize flags first before any processing
        self.collate_fn = self.custom_collate 
        self.all_ar = all_ar
        self.no_caching_latents = no_caching_latents
        self.enable_bucket_sampler = enable_bucket_sampler
        self.resolution_type = resolution_type
        self.min_size = min_size
        self.max_size = max_size
        self.bucket_reso_steps = bucket_reso_steps
        self.use_tag_weighting = use_tag_weighting
        
        # Set up workers
        self.num_workers = num_workers or min(8, os.cpu_count() or 1)
        self.prefetch_factor = prefetch_factor
        
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
            'resolution_type': resolution_type,
            'enable_bucket_sampler': enable_bucket_sampler,
            'min_size': min_size,
            'max_size': max_size,
            'bucket_reso_steps': bucket_reso_steps,
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
            
        # Initialize tag weighter if needed
        if use_tag_weighting:
            self.tag_weighter = TagBasedLossWeighter(
                min_weight=min_tag_weight,
                max_weight=max_tag_weight,
                no_cache=no_caching_latents
            )
        else:
            self.tag_weighter = None
            
        # Convert image paths to strings once
        self.image_paths = [str(p) for p in Path(data_dir).glob("*.png")]
        
        # Initialize dataset structure
        self._initialize_dataset()
        
    def _initialize_dataset(self):
        """Initialize dataset structure with proper error handling"""
        try:
            # Pre-compute bucket data efficiently using ThreadPoolExecutor
            if self.init_params['all_ar']:
                logger.info("Pre-computing bucket data...")
                self._precompute_bucket_data()
                
            # Initialize workers if needed
            if not self.init_params['no_caching_latents']:
                self._initialize_workers()
                self._batch_process_latents_efficient()
                
        except Exception as e:
            logger.error(f"Dataset initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.cleanup()
            raise

    def _precompute_bucket_data(self):
        """Pre-compute bucket data using thread pool"""
        def process_image_batch(paths):
            results = defaultdict(list)
            for path in paths:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                        bucket = (height, width)
                        results[bucket].append(path)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
            return results
            
        # Process in parallel using ThreadPoolExecutor
        chunk_size = max(1, len(self.image_paths) // (self.init_params['num_workers'] * 4))
        self.bucket_data = defaultdict(list)
        
        with ThreadPoolExecutor(max_workers=self.init_params['num_workers']) as executor:
            futures = []
            for i in range(0, len(self.image_paths), chunk_size):
                chunk = self.image_paths[i:i + chunk_size]
                futures.append(executor.submit(process_image_batch, chunk))
                
            # Combine results
            for future in tqdm(futures, desc="Processing image buckets"):
                for bucket, paths in future.result().items():
                    self.bucket_data[bucket].extend(paths)

    def write_captions(self, formatted_paths):
        """Write formatted captions using thread pool"""
        def write_caption(path_caption):
            path, caption = path_caption
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(caption)
            except Exception as e:
                logger.error(f"Error writing to {path}: {str(e)}")
                
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
                    logger.error(f"Error processing task: {str(e)}")
                    result_queue.put(None)
                    
        except Exception as e:
            logger.error(f"Worker process error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Cleanup worker resources
            if hasattr(self, 'worker_models'):
                del self.worker_models

    def _parse_tags(self, caption):
        """Optimized tag parsing with caching"""
        from functools import lru_cache
        
        @lru_cache(maxsize=10000)
        def cached_parse_tags(caption_text):
            """Cache tag parsing results for repeated captions"""
            if not self.tag_weighter:
                return [], {}
            return self.tag_weighter.parse_tags(caption_text)
        
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
            except Exception as e:
                logger.error(f"Caption formatting failed: {str(e)}")
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
        
        @lru_cache(maxsize=10000)
        def cached_calculate_weights(tags_tuple, special_tags_tuple):
            """Cache weight calculations for repeated tag combinations"""
            if not self.tag_weighter:
                return 1.0
            return self.tag_weighter.calculate_weights(list(tags_tuple), dict(special_tags_tuple))
        
        # Convert inputs to hashable types for caching
        tags_tuple = tuple(sorted(tags))
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
        """Highly optimized parallel dataset initialization"""
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        import numpy as np
        from itertools import chain
        from functools import partial
        
        logger.info("Initializing dataset...")
        
        def collect_image_paths():
            """Efficiently collect image paths using set for uniqueness"""
            extensions = {'.png', '.jpg', '.jpeg', '.webp'}
            paths = set()
            
            # Use chain for efficient iteration
            for file in chain.from_iterable(
                self.data_dir.glob(f'*{ext}') for ext in extensions
            ):
                paths.add(file)
            return list(paths)
        
        def process_image_batch(batch, all_ar=False):
            """Process a batch of images with optimized validation"""
            batch_valid = []
            batch_buckets = defaultdict(list) if all_ar else None
            
            # Pre-compile checks for better performance
            def is_valid_image(path):
                try:
                    with Image.open(path) as img:
                        # Fast header checks
                        if img.mode not in {'RGB', 'RGBA'}:
                            return False
                            
                        # Quick size validation
                        w, h = img.size
                        if w < 256 or h < 256:
                            return False
                            
                        if not self.all_ar and (w > 4096 or h > 4096):
                            return False
                            
                        # Verify caption exists
                        if not path.with_suffix('.txt').exists():
                            return False
                            
                        return True
                except:
                    return False
            
            # Process batch efficiently
            for path in batch:
                if is_valid_image(path):
                    if all_ar:
                        with Image.open(path) as img:
                            w, h = img.size
                            batch_buckets[(h, w)].append(path)
                    batch_valid.append(path)
            
            return (batch_valid, batch_buckets) if all_ar else batch_valid
        
        # Collect paths efficiently
        image_paths = collect_image_paths()
        total_images = len(image_paths)
        
        if total_images == 0:
            logger.warning("No images found in data directory")
            self.image_paths = []
            return
        
        logger.info(f"Found {total_images} potential images")
        
        # Optimize worker and batch configuration
        num_workers = min(32, (os.cpu_count() or 1) + 4)
        batch_size = max(100, total_images // (num_workers * 4))
        
        # Create optimally sized batches
        batches = np.array_split(image_paths, 
                               max(1, total_images // batch_size))
        
        # Initialize results containers
        valid_paths = []
        bucket_data = defaultdict(list) if self.all_ar else None
        
        # Process batches in parallel with progress tracking
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            process_func = partial(process_image_batch, all_ar=self.all_ar)
            futures = [executor.submit(process_func, batch) for batch in batches]
            
            with tqdm(total=len(batches), desc="Validating images") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if self.all_ar:
                            batch_valid, batch_buckets = result
                            valid_paths.extend(batch_valid)
                            # Merge bucket data efficiently
                            for bucket, paths in batch_buckets.items():
                                bucket_data[bucket].extend(paths)
                        else:
                            valid_paths.extend(result)
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
                    pbar.update(1)
        
        # Sort paths for consistency
        self.image_paths = sorted(valid_paths)
        
        # Handle latent caching setup
        if not self.no_caching_latents:
            # Pre-calculate all paths at once
            self.latent_paths = [
                self.cache_dir / f"{Path(p).stem}_latents.pt" 
                for p in self.image_paths
            ]
            logger.info(f"Created {len(self.latent_paths)} latent cache paths")
        else:
            logger.info("Latent caching disabled - using on-the-fly processing")
        
        # Set up bucket data if needed
        if self.all_ar:
            self.bucket_data = bucket_data
            logger.info(f"Created {len(bucket_data)} unique size buckets")
        
        logger.info(f"Successfully initialized dataset with {len(self.image_paths)} valid images")

        # Initialize size cache for faster bucket sampling
        if self.enable_bucket_sampler:
            self._size_cache = {}
            
            def cache_image_sizes(paths_batch):
                """Process a batch of images to cache their sizes"""
                batch_sizes = {}
                for path in paths_batch:
                    try:
                        with Image.open(path) as img:
                            batch_sizes[path] = img.size[::-1]  # (h,w)
                    except Exception as e:
                        logger.warning(f"Failed to cache size for {path}: {str(e)}")
                return batch_sizes
            
            # Process in batches
            batch_size = 1000
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, len(self.image_paths), batch_size):
                    batch = self.image_paths[i:i + batch_size]
                    futures.append(executor.submit(cache_image_sizes, batch))
                
                # Collect results
                for future in futures:
                    try:
                        self._size_cache.update(future.result())
                    except Exception as e:
                        logger.error(f"Failed to process size cache batch: {str(e)}")
            
            logger.info(f"Cached sizes for {len(self._size_cache)} images")
        
        # ... rest of initialization code ...

    def _validate_and_process_chunk(self, image_paths, process_ar=False, cache_latents=True):
        """Ultra-fast image validation using batched processing and caching"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from functools import lru_cache
        
        valid_paths = []
        bucket_data = defaultdict(list) if process_ar else None
        
        # Cache image mode checks
        @lru_cache(maxsize=1024)
        def is_valid_mode(mode):
            return mode in ['RGB', 'RGBA']
        
        # Batch process images
        def process_batch(paths_batch):
            results = []
            for path in paths_batch:
                try:
                    if not path.with_suffix('.txt').exists():
                        continue
                        
                    # Fast image header check without full load
                    with Image.open(path) as img:
                        if not is_valid_mode(img.mode):
                            continue
                            
                        width, height = img.size
                        
                        # Quick size validation
                        if width < 256 or height < 256:
                            continue
                            
                        if not self.all_ar and (width > 4096 or height > 4096):
                            continue
                        
                        # Minimal corruption check
                        try:
                            img.verify()
                        except:
                            continue
                        
                        if process_ar:
                            results.append((path, (height, width)))
                        else:
                            results.append(path)
                            
                except Exception:
                    continue
                    
            return results

        # Optimize batch size based on CPU cores
        optimal_batch_size = 1000
        batches = np.array_split(image_paths, 
                               max(1, len(image_paths) // optimal_batch_size))
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            # Collect results efficiently
            for future in futures:
                batch_results = future.result()
                if process_ar:
                    for path, bucket in batch_results:
                        valid_paths.append(path)
                        bucket_data[bucket].append(path)
                else:
                    valid_paths.extend(batch_results)

        if process_ar:
            return valid_paths, bucket_data
        return valid_paths

    
    def _precompute_latents(self, img_path, cache_path, img=None):
        """Precompute and cache latents for an image"""
        if img is None:
            img = Image.open(img_path)
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
        
        # Transform image
        pixel_values = self.image_transforms(img)
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        # Move to device temporarily
        self.vae = self.vae.to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        # Compute latents
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
        # Save latents
        torch.save(latents.cpu(), cache_path)
        
        # Move VAE back to CPU if needed
        self.vae = self.vae.cpu()

    def _upscale_image(self, img, target_width, target_height):
        """Upscale image to target dimensions"""
        return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

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
                        processed = self.process_image_size([image])[0]
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
                
    def check_latent_dimensions(self, image_paths, batch_size=32):
        """Efficiently check latent dimensions for multiple images in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
            
        # Fast path for cached images
        results = {}
        uncached_paths = []
        
        for path in image_paths:
            if path in self.latent_cache:
                results[path] = self.latent_cache[path] == self.expected_latent_shape
            else:
                uncached_paths.append(path)
        
        if not uncached_paths:
            return results if len(image_paths) > 1 else results[image_paths[0]]
        
        # Process uncached images in batches
        num_workers = min(8, (os.cpu_count() or 1))
        
        def process_batch(paths):
            batch_results = {}
            batch_images = []
            valid_paths = []
            
            # Load images in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_path = {
                    executor.submit(self.load_and_transform_image, path): path 
                    for path in paths
                }
                
                for future in future_to_path:
                    path = future_to_path[future]
                    try:
                        img = future.result()
                        if img is not None:
                            batch_images.append(img)
                            valid_paths.append(path)
                    except Exception as e:
                        logger.warning(f"Failed to process {path}: {str(e)}")
                        batch_results[path] = False
            
            if not batch_images:
                return batch_results
            
            # Process batch through VAE
            try:
                with torch.no_grad():
                    batch_tensor = torch.stack(batch_images).to(self.vae.device)
                    
                    # Use mixed precision if available
                    with torch.cuda.amp.autocast(enabled=True):
                        latents = self.vae.encode(batch_tensor).latent_dist.sample()
                    
                    # Update cache and results
                    for idx, path in enumerate(valid_paths):
                        shape = tuple(latents[idx].shape)
                        self.latent_cache[path] = shape
                        batch_results[path] = shape == self.expected_latent_shape
                    
            except Exception as e:
                logger.error(f"Batch encoding failed: {str(e)}")
                for path in valid_paths:
                    batch_results[path] = False
            
            return batch_results
        
        # Process in optimally sized batches
        batches = [
            uncached_paths[i:i + batch_size] 
            for i in range(0, len(uncached_paths), batch_size)
        ]
        
        for batch in batches:
            batch_results = process_batch(batch)
            results.update(batch_results)
        
        return results if len(image_paths) > 1 else results[image_paths[0]]

    def load_and_transform_image(self, image_path):
        """Optimized image loading and transformation"""
        from functools import lru_cache
        
        @lru_cache(maxsize=1)
        def get_transform():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        try:
            # Use context manager for proper cleanup
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Use cached transform
                transform = get_transform()
                return transform(img)
                
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {str(e)}")
            return None

    def _build_tag_statistics(self):
        """Multi-threaded tag statistics builder with optimized processing"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from collections import Counter
        from functools import partial
        
        def process_caption_batch(paths_batch):
            """Process a batch of captions and return stats"""
            batch_stats = {
                'niji_count': 0,
                'quality_6_count': 0,
                'stylize_values': [],
                'chaos_values': [],
                'formatted_count': 0,
                'formatted_paths': []  # Track paths that need writing
            }
            
            for img_path in paths_batch:
                caption_path = img_path.with_suffix('.txt')
                if not caption_path.exists():
                    continue
                    
                try:
                    # Read caption
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()
                    
                    # Format caption
                    formatted_caption = self._format_caption(original_caption)
                    
                    # Track if needs saving
                    if formatted_caption != original_caption:
                        batch_stats['formatted_count'] += 1
                        batch_stats['formatted_paths'].append(
                            (caption_path, formatted_caption)
                        )
                    
                    # Update tag statistics
                    _, special_tags = self._parse_tags(formatted_caption)
                    if special_tags.get('niji', False):
                        batch_stats['niji_count'] += 1
                    if special_tags.get('version', 0) == 6:
                        batch_stats['quality_6_count'] += 1
                    if 'stylize' in special_tags:
                        batch_stats['stylize_values'].append(special_tags['stylize'])
                    if 'chaos' in special_tags:
                        batch_stats['chaos_values'].append(special_tags['chaos'])
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {str(e)}")
                    
            return batch_stats
        
        # Initialize combined stats
        combined_stats = {
            'niji_count': 0,
            'quality_6_count': 0,
            'stylize_values': [],
            'chaos_values': [],
            'total_images': len(self.image_paths),
            'formatted_count': 0
        }
        
        # Calculate optimal batch size and number of workers
        num_workers = min(32, (os.cpu_count() or 1) + 4)
        batch_size = max(100, len(self.image_paths) // (num_workers * 4))
        
        # Create batches
        batches = np.array_split(self.image_paths, 
                               max(1, len(self.image_paths) // batch_size))
        
        # Process batches in parallel
        formatted_paths = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_caption_batch, batch) 
                      for batch in batches]
            
            # Process results with progress bar
            with tqdm(total=len(batches), desc="Processing caption batches") as pbar:
                for future in futures:
                    try:
                        batch_stats = future.result()
                        
                        # Combine statistics
                        combined_stats['niji_count'] += batch_stats['niji_count']
                        combined_stats['quality_6_count'] += batch_stats['quality_6_count']
                        combined_stats['stylize_values'].extend(batch_stats['stylize_values'])
                        combined_stats['chaos_values'].extend(batch_stats['chaos_values'])
                        combined_stats['formatted_count'] += batch_stats['formatted_count']
                        
                        # Collect paths that need writing
                        formatted_paths.extend(batch_stats['formatted_paths'])
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
        
        # Write formatted captions in parallel
        if formatted_paths:
            def write_caption(path_caption):
                path, caption = path_caption
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                except Exception as e:
                    logger.error(f"Error writing to {path}: {str(e)}")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(write_caption, formatted_paths))
        
        logger.info(f"Formatted {combined_stats['formatted_count']} captions")
        return combined_stats

    def _get_target_size(self, width, height):
        """Single image target size calculation"""
        from functools import lru_cache
        import numpy as np
        
        @lru_cache(maxsize=1024)
        def calculate_target_size(w, h, steps, min_size, all_ar):
            if all_ar:
                if steps > 1:
                    target = np.array([w, h], dtype=np.float32)
                    target = np.round(target / steps) * steps
                    target = np.maximum(target, min_size)
                    return int(target[0]), int(target[1])
                return w, h
            
            aspect_ratio = h / w
            if aspect_ratio > 1:
                tw = 1024
                th = int(np.round(tw * aspect_ratio / steps) * steps)
            else:
                th = 1024
                tw = int(np.round(th / aspect_ratio / steps) * steps)
            
            max_dim = max(tw, th)
            if max_dim > 2048:
                scale = 2048 / max_dim
                dims = np.array([tw, th], dtype=np.float32)
                dims = np.round(dims * scale / steps) * steps
                return int(dims[0]), int(dims[1])
            
            return tw, th
        
        return calculate_target_size(
            width, height, 
            self.bucket_reso_steps,
            self.min_size,
            self.all_ar
        )

    def _get_target_sizes_batch(self, image_sizes, num_workers=None):
        """Multi-threaded batch processing of target sizes"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        def process_batch(batch):
            return [self._get_target_size(w, h) for w, h in batch]
        
        # Optimize batch size based on total images
        batch_size = max(1, len(image_sizes) // (num_workers * 4))
        batches = [
            image_sizes[i:i + batch_size] 
            for i in range(0, len(image_sizes), batch_size)
        ]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_batch, batches))
            
        # Flatten results
        return [size for batch_result in results for size in batch_result]

    def _get_target_sizes_vectorized(self, image_sizes):
        """Vectorized processing for large batches"""
        import numpy as np
        
        # Convert to numpy array for vectorized operations
        sizes = np.array(image_sizes, dtype=np.float32)
        widths = sizes[:, 0]
        heights = sizes[:, 1]
        
        if self.all_ar and self.bucket_reso_steps > 1:
            # Vectorized all_ar processing
            target = np.round(sizes / self.bucket_reso_steps) * self.bucket_reso_steps
            target = np.maximum(target, self.min_size)
            return target.astype(np.int32).tolist()
        
        # Calculate aspect ratios
        aspect_ratios = heights / widths
        is_portrait = aspect_ratios > 1
        
        # Initialize target arrays
        target_widths = np.zeros_like(widths)
        target_heights = np.zeros_like(heights)
        
        # Process portrait images
        portrait_mask = is_portrait
        target_widths[portrait_mask] = 1024
        target_heights[portrait_mask] = np.round(
            1024 * aspect_ratios[portrait_mask] / self.bucket_reso_steps
        ) * self.bucket_reso_steps
        
        # Process landscape images
        landscape_mask = ~portrait_mask
        target_heights[landscape_mask] = 1024
        target_widths[landscape_mask] = np.round(
            1024 / aspect_ratios[landscape_mask] / self.bucket_reso_steps
        ) * self.bucket_reso_steps
        
        # Scale down large images
        max_dims = np.maximum(target_widths, target_heights)
        scale_mask = max_dims > 2048
        if np.any(scale_mask):
            scale = 2048 / max_dims[scale_mask]
            target_widths[scale_mask] = np.round(
                target_widths[scale_mask] * scale / self.bucket_reso_steps
            ) * self.bucket_reso_steps
            target_heights[scale_mask] = np.round(
                target_heights[scale_mask] * scale / self.bucket_reso_steps
            ) * self.bucket_reso_steps
        
        return np.stack([target_widths, target_heights], axis=1).astype(np.int32).tolist()

    def get_target_sizes(self, image_sizes, use_vectorized=True):
        """Smart dispatcher for target size calculations"""
        if not image_sizes:
            return []
            
        # Use vectorized version for large batches
        if use_vectorized and len(image_sizes) > 1000:
            return self._get_target_sizes_vectorized(image_sizes)
        
        # Use multi-threaded version for medium batches
        if len(image_sizes) > 32:
            return self._get_target_sizes_batch(image_sizes)
        
        # Use single-threaded version for small batches
        return [self._get_target_size(w, h) for w, h in image_sizes]

    def process_image_size(self, images):
        """Multi-threaded image processing using num_workers"""
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        def process_single_image(image):
            width, height = image.size
            
            if self.all_ar:
                # Minimal processing for all_ar mode
                               # Minimal processing for all_ar mode
                if width < self.min_size or height < self.min_size:
                    scale = max(self.min_size / width, self.min_size / height)
                    new_width = int(round(width * scale / self.bucket_reso_steps) * self.bucket_reso_steps)
                    new_height = int(round(height * scale / self.bucket_reso_steps) * self.bucket_reso_steps)
                    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return image
            # Get target size for non-all_ar mode
            target_width, target_height = self._get_target_size(width, height)
            
            # Only resize if dimensions differ
            if width != target_width or height != target_height:
                return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            return image

        # Process images in parallel using ThreadPoolExecutor
        num_workers = min(self.num_workers, len(images))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_images = list(executor.map(process_single_image, images))

        return processed_images

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
        """Ultra-fast high-quality downscaling with optimized processing"""
        from functools import lru_cache
        import cv2
        import numpy as np
        
        @lru_cache(maxsize=32)
        def get_sharpening_kernel():
            """Cache sharpening kernel"""
            return np.array([[-0.5,-0.5,-0.5], 
                           [-0.5, 5.0,-0.5],
                           [-0.5,-0.5,-0.5]], dtype=np.float32) / 2.0
        
        @lru_cache(maxsize=1024)
        def calculate_steps(scale):
            """Cache step calculations"""
            return max(1, min(3, int(scale // 1.5)))
        
        try:
            # Fast numpy conversion with proper dtype
            img_np = np.asarray(image, dtype=np.uint8)
            
            # Efficient scale calculation
            scale_factor = max(image.width / target_width, 
                             image.height / target_height)
            
            if scale_factor <= 1.0:
                return image
            
            # Get cached number of steps
            num_steps = calculate_steps(scale_factor)
            
            # Optimized initial blur
            blur_radius = min(2.0, 0.6 * scale_factor)
            kernel_size = max(3, int(blur_radius * 3) | 1)
            
            # Apply blur only if necessary
            if scale_factor > 1.5:
                img_np = cv2.GaussianBlur(
                    img_np, 
                    (kernel_size, kernel_size),
                    blur_radius,
                    borderType=cv2.BORDER_REFLECT
                )
            
            # Optimized bilateral filter for large downscaling
            if scale_factor > 2.0:
                img_np = cv2.bilateralFilter(
                    img_np, 
                    d=9,
                    sigmaColor=75,
                    sigmaSpace=75,
                    borderType=cv2.BORDER_REFLECT
                )
            
            # Pre-calculate sizes for progressive downscaling
                       # Pre-calculate sizes for progressive downscaling
            sizes = [
                (   
                    int(target_width + (image.width - target_width) * (1 - (step + 1) / num_steps)),
                    int(target_height + (image.height - target_height) * (1 - (step + 1) / num_steps))
                ) for step in range(num_steps)
            ]
            
            # Progressive downscaling with optimized memory usage
            for i, (w, h) in enumerate(sizes):
                img_np = cv2.resize(
                    img_np, 
                    (w, h), 
                    interpolation=cv2.INTER_AREA,
                    dst=img_np  # Reuse memory when possible
                )
                
                # Apply cached sharpening kernel on final step
                if i == num_steps - 1:
                    kernel = get_sharpening_kernel()
                    img_np = cv2.filter2D(
                        img_np, 
                        -1, 
                        kernel,
                        borderType=cv2.BORDER_REFLECT
                    )
            
            return Image.fromarray(img_np)
            
        except Exception:
            # Fast fallback without logging
            return image.resize(
                (target_width, target_height), 
                Image.LANCZOS
            )

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

        # Create image to bucket mapping
        self.image_to_bucket = {}
        for bucket, paths in self.bucket_data.items():
            for path in paths:
                self.image_to_bucket[path] = bucket
                
        logger.info(f"Created bucket mappings for {len(self.image_to_bucket)} images")

    def _assign_to_bucket(self, img_path):
        """Fast bucket assignment with caching and optimized calculations"""
        from functools import lru_cache
        import numpy as np
        
        # Cache bucket calculations
        @lru_cache(maxsize=1024)
        def find_best_bucket(target_h, target_w, aspect_ratio):
            if not self.buckets:
                return None
                
            buckets_array = np.array(self.buckets)
            areas = buckets_array[:, 0] * buckets_array[:, 1]
            target_area = target_h * target_w
            
            # Calculate area differences for all buckets at once
            area_diffs = areas - target_area
            valid_buckets = area_diffs >= 0
            
            if np.any(valid_buckets):
                # Find bucket with minimum area difference
                best_idx = np.argmin(area_diffs * valid_buckets)
                return tuple(buckets_array[best_idx])
            
            # If no valid bucket, try aspect ratio matching
            bucket_ars = buckets_array[:, 0] / buckets_array[:, 1]
            ar_diffs = np.abs(bucket_ars - aspect_ratio)
            ar_valid = ar_diffs < 0.1
            
            if np.any(ar_valid):
                best_idx = np.argmin(np.abs(areas - target_area) * ar_valid)
                return tuple(buckets_array[best_idx])
            
            # Return largest bucket as fallback
            return tuple(buckets_array[np.argmax(areas)])

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                if self.all_ar:
                    # Fast all_ar handling
                    bucket = (height, width)
                    if bucket not in self.buckets:
                        self.buckets.append(bucket)
                        self.bucket_data[bucket] = []
                    return img_path, bucket
                
                # Get target dimensions
                target_h, target_w = self._get_target_size(width, height)
                aspect_ratio = height / width
                
                # Find best bucket using cached function
                best_bucket = find_best_bucket(target_h, target_w, aspect_ratio)
                return img_path, best_bucket
                
        except Exception:
            return img_path, None

    def _apply_text_transforms(self, caption):
        """Optimized text augmentation with faster token processing"""
        from functools import lru_cache
        import numpy as np
        
        @lru_cache(maxsize=128)
        def split_and_clean(text):
            """Cache token splitting and cleaning for repeated captions"""
            return [t.strip() for t in text.split(",") if t.strip()]
        
        # Fast caption dropout
        if random.random() < self.caption_dropout_rate:
            return ""
        
        if self.token_dropout_rate > 0:
            # Get clean tokens using cached function
            tokens = split_and_clean(caption)
            
            if not tokens:
                return caption
                
            # Use numpy for faster random operations
            mask = np.random.random(len(tokens)) > self.token_dropout_rate
            
            # Keep at least one token
            if not mask.any():
                mask[np.random.randint(len(tokens))] = True
            
            # Fast token filtering
            kept_tokens = [t for i, t in enumerate(tokens) if mask[i]]
            
            # Efficient string joining
            return ", ".join(kept_tokens)
        
        return caption

    def _process_uncached_item(self, img_path, caption_path):
        """Optimized uncached item processing with better performance"""
        from functools import lru_cache
        
        # Cache transform composition
        @lru_cache(maxsize=1)
        def get_transform():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        try:
            # Load and process image efficiently
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                
                # Fast bucket dimension calculation
                if self.all_ar:
                    bucket_h, bucket_w = image.size[::-1]  # Faster than separate assignment
                else:
                    bucket_h, bucket_w = self._get_bucket_size(image.size)
                
                # Process image with optimized resizing
                processed_image = self._advanced_resize(image, bucket_w, bucket_h)
                original_size = image.size
            
            # Generate latents efficiently
            transform = get_transform()
            with torch.no_grad():
                image_tensor = transform(processed_image).unsqueeze(0).to(
                    self.vae.device, dtype=self.vae.dtype
                )
                
                if self.no_caching_latents:
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents = latents.squeeze(0)
                else:
                    latents = self._get_cached_latents(img_path, processed_image)
            
            # Process caption and generate embeddings efficiently
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            caption = self._apply_text_transforms(caption)
            
            # Batch process text inputs
            text_inputs = {
                'text1': self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.text_encoder.device),
                
                'text2': self.tokenizer_2(
                    caption,
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.text_encoder_2.device)
            }
            
            # Generate embeddings in parallel
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs['text1'].input_ids)[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs['text2'].input_ids,
                    output_hidden_states=True
                )
                
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
            
            # Generate time embeddings efficiently
            add_time_ids = torch.tensor([
                original_size[0], original_size[1],  # original size
                bucket_w, bucket_h,                  # target size
                0, 0,                               # crop top-left
                original_size[0], original_size[1]   # crop bottom-right
            ], dtype=torch.float32, device=self.vae.device).unsqueeze(0)
            
            # Return optimized cache data
            return {
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
            
        except Exception as e:
            logger.error(f"Error processing uncached item {img_path}: {str(e)}")
            return None
    
    def _get_bucket_size(self, image_size):
        """Ultra-optimized bucket size calculation with advanced caching"""
        from functools import lru_cache
        import numpy as np
        
        @lru_cache(maxsize=1)
        def get_buckets_array():
            """Cache numpy array conversion of buckets"""
            return np.array(self.buckets) if self.buckets else None
            
        @lru_cache(maxsize=1)
        def get_bucket_areas():
            """Cache bucket area calculations"""
            buckets = get_buckets_array()
            return buckets[:, 0] * buckets[:, 1] if buckets is not None else None
        
        @lru_cache(maxsize=1024)
        def find_best_bucket(width, height):
            """Find best bucket with optimized calculations"""
            if not self.buckets:
                return None
                
            if self.all_ar:
                return (height, width)
            
                       # Get cached arrays
            buckets_array = get_buckets_array()
            bucket_areas = get_bucket_areas()
            target_area = width * height
            
            # Fast path for exact matches
            exact_match = np.where(
                (buckets_array[:, 0] == height) & 
                (buckets_array[:, 1] == width)
            )
            if exact_match[0].size:
                return tuple(buckets_array[exact_match[0]])
            
            # Vectorized validity check
            valid_mask = (buckets_array[:, 0] >= height) & (buckets_array[:, 1] >= width)
            
            if np.any(valid_mask):
                # Optimized area difference calculation
                area_diffs = np.abs(bucket_areas - target_area)
                area_diffs[~valid_mask] = np.inf
                
                # Consider aspect ratio for better matching
                aspect_ratios = buckets_array[:, 0] / buckets_array[:, 1]
                target_ar = height / width
                ar_diffs = np.abs(aspect_ratios - target_ar)
                
                # Combined score using both area and aspect ratio
                scores = area_diffs * (1 + ar_diffs * 0.1)
                best_idx = np.argmin(scores)
                
                return tuple(buckets_array[best_idx])
            
            # Optimized fallback to largest bucket
            if bucket_areas is not None:
                return tuple(buckets_array[np.argmax(bucket_areas)])
            return None
        
        # Handle invalid inputs
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError(f"Invalid image size: {image_size}")
            
        width, height = map(int, image_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
            
        return find_best_bucket(width, height)
    
    def get_bucket_sizes_batch(self, image_sizes, num_workers=None):
        """Parallel bucket size calculation for multiple images"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        if not image_sizes:
            return []
            
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        def process_batch(sizes_batch):
            return [self._get_bucket_size(size) for size in sizes_batch]
            
        # Process in parallel for large batches
        if len(image_sizes) > 100:
            batch_size = max(50, len(image_sizes) // (num_workers * 4))
            batches = np.array_split(image_sizes, max(1, len(image_sizes) // batch_size))
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                results = [r for f in futures for r in f.result()]
            return results
            
        # Process sequentially for small batches
        return process_batch(image_sizes)
    
    def _advanced_resize(self, image, target_width, target_height):
        """High-quality image resizing with optimized performance"""
        from functools import lru_cache
        import cv2
        import numpy as np
        
        @lru_cache(maxsize=1024)
        def calculate_dimensions(w, h, tw, th):
            aspect_ratio = w / h
            target_ratio = tw / th
            
            if aspect_ratio > target_ratio:
                new_w = tw
                new_h = int(tw / aspect_ratio)
            else:
                new_h = th
                new_w = int(th * aspect_ratio)
                
            left = (tw - new_w) // 2
            top = (th - new_h) // 2
            return new_w, new_h, left, top
        
        try:
            width, height = image.size
            new_width, new_height, left, top = calculate_dimensions(
                width, height, target_width, target_height
            )
            
            # Convert to numpy array once
            img_np = np.array(image)
            
            # Determine if downscaling or upscaling
            if width > new_width or height > new_height:
                # Downscaling with quality preservation
                scale_factor = max(width / new_width, height / new_height)
                
                # Apply initial gaussian blur to prevent aliasing
                if scale_factor > 2.0:
                    sigma = 0.3 * scale_factor
                    kernel_size = int(2 * round(sigma) + 1)
                    img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
                
                # High-quality downscaling
                img_np = cv2.resize(img_np, (new_width, new_height), 
                                  interpolation=cv2.INTER_AREA)
                
            else:
                # Upscaling with enhanced quality
                img_np = cv2.resize(img_np, (new_width, new_height), 
                                  interpolation=cv2.INTER_CUBIC)
            
            # Create padded result efficiently
            result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            result[top:top + new_height, left:left + new_width] = img_np
            
            return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"Advanced resize failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def custom_collate(self, batch):
        """Collate function with better error handling"""
        # Filter out None values
        valid_items = [item for item in batch if item is not None]
        
        if not valid_items:
            logger.warning("Batch contained all None values, retrying with new batch")
            return None

        # Define helper functions at class level or outside
        def fast_stack_tensors(tensors, dim=0):
            """Optimized tensor stacking with pre-allocation"""
            if not tensors:
                return None
            
            # Fast path for single tensor
            if len(tensors) == 1:
                return tensors[0].unsqueeze(0)
                
            # Pre-allocate output tensor
            shape = list(tensors[0].shape)
            shape.insert(dim, len(tensors))
            dtype = tensors[0].dtype
            device = tensors[0].device
            
            result = torch.empty(shape, dtype=dtype, device=device)
            for i, tensor in enumerate(tensors):
                result.index_copy_(dim, torch.tensor([i], device=device), tensor.unsqueeze(dim))
            
            return result
        
        def process_tensor_batch(key, items, batch_size):
            """Process a batch of tensors efficiently"""
            tensors = [None] * batch_size
            for i, item in enumerate(items):
                tensors[i] = item[key]
            return key, fast_stack_tensors(tensors)
        
        try:
            # Optimized batch filtering
            valid_batch = valid_items  # We already filtered above
            batch_size = len(valid_batch)
            
            if batch_size == 0:
                raise RuntimeError("Empty batch after filtering")
                
            if batch_size == 1:
                # Fast path for single item batches
                return {
                    'latents': valid_batch[0]['latents'].unsqueeze(0),
                    'text_embeddings': valid_batch[0]['text_embeddings'].unsqueeze(0),
                    'text_embeddings_2': valid_batch[0]['text_embeddings_2'].unsqueeze(0),
                    'bucket_size': valid_batch[0]['bucket_size'],
                    'added_cond_kwargs': {
                        k: v.unsqueeze(0) if torch.is_tensor(v) else [v]
                        for k, v in valid_batch[0]['added_cond_kwargs'].items()
                    } if 'added_cond_kwargs' in valid_batch[0] else None
                }
            
            # Quick bucket size validation using numpy
            bucket_size = valid_batch[0]['bucket_size']
            if not self.all_ar:
                sizes = np.array([item['bucket_size'] for item in valid_batch])
                if not np.all(sizes == sizes[0]):
                    raise ValueError(f"Inconsistent bucket sizes: {sizes.tolist()}")
            
            # Process main tensors in parallel
            main_keys = ['latents', 'text_embeddings', 'text_embeddings_2']
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        process_tensor_batch, 
                        key, 
                        valid_batch, 
                        batch_size
                    ) for key in main_keys
                ]
                
                # Build result dict from futures
                result = dict(future.result() for future in futures)
                result['bucket_size'] = bucket_size
            
            # Handle added_cond_kwargs if present
            if 'added_cond_kwargs' in valid_batch[0]:
                first_item = valid_batch[0]['added_cond_kwargs']
                added_cond_kwargs = {}
                
                # Process tensor and non-tensor data separately
                tensor_keys = [k for k, v in first_item.items() if torch.is_tensor(v)]
                non_tensor_keys = [k for k, v in first_item.items() if not torch.is_tensor(v)]
                
                # Process tensor keys in parallel
                if tensor_keys:
                    with ThreadPoolExecutor(max_workers=len(tensor_keys)) as executor:
                        futures = [
                            executor.submit(
                                process_tensor_batch,
                                key,
                                [item['added_cond_kwargs'] for item in valid_batch],
                                batch_size
                            ) for key in tensor_keys
                        ]
                        added_cond_kwargs.update(dict(future.result() for future in futures))
                
                # Process non-tensor keys
                for key in non_tensor_keys:
                    added_cond_kwargs[key] = [
                        item['added_cond_kwargs'][key] for item in valid_batch
                    ]
                
                result['added_cond_kwargs'] = added_cond_kwargs
            
            return result
            
        except Exception as e:
            logger.error(f"Collation error: {str(e)}")
            return None

    def __getitem__(self, idx):
        """Optimized item retrieval with caching and efficient processing"""
        from functools import lru_cache
        import numpy as np
        
        @lru_cache(maxsize=1)
        def get_transform():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        @lru_cache(maxsize=1024)
        def get_bucket_size(img_path):
            """Cache bucket size calculations"""
            if not self.enable_bucket_sampler:
                return None, None
                
            bucket = self.image_to_bucket.get(img_path)
            if bucket is None and self.all_ar:
                with Image.open(img_path) as img:
                    w, h = img.size
                    bucket = (h, w)
            return bucket if bucket else (None, None)
        
        try:
            # Fast path for image and caption paths
            img_path = self.image_paths[idx]
            caption_path = Path(img_path).with_suffix('.txt')
            
            # Get cached bucket dimensions
            bucket_h, bucket_w = get_bucket_size(str(img_path))
            
            # Efficient image loading and processing
            with Image.open(img_path) as image:
                # Fast RGB conversion if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Optimized image resizing
                if self.enable_bucket_sampler:
                    image = self.process_image_size([image])[0]
                else:
                    # Get target size and resize
                    w, h = image.size
                    target_w, target_h = self._get_target_size(w, h)
                    image = self._advanced_resize(image, target_w, target_h)
                
                original_size = image.size
            
            # Fast caption loading and processing
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            caption = self._apply_text_transforms(caption)
            
            # Batch process text inputs
            text_inputs = {
                'text1': self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.text_encoder.device),
                
                'text2': self.tokenizer_2(
                    caption,
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.text_encoder_2.device)
            }
            
            # Generate embeddings in parallel
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs['text1'].input_ids)[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs['text2'].input_ids,
                    output_hidden_states=True
                )
                
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
            
            # Generate time embeddings efficiently
            target_size = (bucket_w, bucket_h) if self.enable_bucket_sampler else (target_w, target_h)
            
            add_time_ids = torch.tensor([
                original_size[0], original_size[1],  # original size
                target_size[0], target_size[1],      # target size
                0, 0,                               # crop top-left
                original_size[0], original_size[1]   # crop bottom-right
            ], dtype=torch.float32, device=self.vae.device).unsqueeze(0)
            
            # Optimized latent handling
            if not self.no_caching_latents and self.latent_paths[idx].exists():
                latents = torch.load(
                    self.latent_paths[idx], 
                    map_location='cpu'
                )['latents'].float()
            else:
                # Generate latents with cached transform
                transform = get_transform()
                with torch.no_grad():
                    image_tensor = transform(image).unsqueeze(0).to(
                        self.vae.device, dtype=self.vae.dtype
                    )
                    
                    # Use mixed precision if available
                    with torch.cuda.amp.autocast(enabled=True):
                        latents = self.vae.encode(image_tensor).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                        latents = latents.squeeze(0).cpu()
                    
                    if not self.no_caching_latents:
                        torch.save({'latents': latents}, self.latent_paths[idx])
            
            # Return optimized data structure
            return {
                'latents': latents,
                'text_embeddings': text_embeddings,
                'text_embeddings_2': hidden_states,
                'added_cond_kwargs': {
                    'text_embeds': pooled_output.unsqueeze(1).unsqueeze(2),
                    'time_ids': add_time_ids
                },
                'bucket_size': (bucket_h, bucket_w) if self.enable_bucket_sampler else target_size
            }
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            return None

    def _get_cached_latents(self, img_path, processed_image):
        """Optimized latent generation and caching"""
        from functools import lru_cache
        
        # Cache transform composition
        @lru_cache(maxsize=1)
        def get_transform():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
        
        try:
            if not self.no_caching_latents and cache_path.exists():
                return torch.load(cache_path, map_location='cpu')['latents']
            
            # Generate latents efficiently
            with torch.no_grad():
                transform = get_transform()
                image_tensor = transform(processed_image).unsqueeze(0).to(
                    self.vae.device, dtype=self.vae.dtype
                )
                
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.squeeze(0)
                
                if not self.no_caching_latents:
                    torch.save({'latents': latents}, cache_path)
                
                return latents
                
        except Exception as e:
            logger.error(f"Error generating latents: {str(e)}")
            raise

    def _format_captions(self):
        """Parallel caption formatting with batched processing"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from tqdm import tqdm
        
        def process_caption_batch(paths_batch):
            results = []
            for img_path in paths_batch:
                caption_path = img_path.with_suffix('.txt')
                if caption_path.exists():
                    try:
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        if caption:
                            formatted = self._format_caption(caption)
                            if formatted != caption:
                                results.append(img_path)
                    except Exception:
                        continue
            return results
        
        logger.info("Formatting captions in parallel...")
        
        # Optimize batch size and parallel processing
        optimal_batch_size = 1000
        batches = np.array_split(self.image_paths, 
                               max(1, len(self.image_paths) // optimal_batch_size))
        
        formatted_paths = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_caption_batch, batch) for batch in batches]
            
            # Process results with progress bar
            with tqdm(total=len(batches), desc="Processing caption batches") as pbar:
                for future in futures:
                    batch_results = future.result()
                    formatted_paths.extend(batch_results)
                    pbar.update(1)
        
        logger.info(f"Formatted {len(formatted_paths)} captions")

    def get_image_size(self, idx):
        """Efficiently get image dimensions without loading full image"""
        try:
            img_path = self.image_paths[idx]
            
            # Try to get cached size first
            if hasattr(self, '_size_cache'):
                cached_size = self._size_cache.get(img_path)
                if cached_size is not None:
                    return cached_size
            
            # Use PIL's fast size reading
            with Image.open(img_path) as img:
                size = img.size[::-1]  # Convert (w,h) to (h,w)
                
                # Cache the result if cache exists
                if hasattr(self, '_size_cache'):
                    self._size_cache[img_path] = size
                    
                return size
                
        except Exception as e:
            logger.warning(f"Failed to get size for image {idx}: {str(e)}")
            return None


class BucketSampler(CustomSamplerBase):
    """Memory-efficient bucket sampler with resolution handling"""
    
    def __init__(
        self,
        dataset: CustomDatasetBase,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        resolution_binning: bool = True
    ):
        # Initialize parent with dataset
        super().__init__(data_source=dataset)
        
        try:
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.seed = seed
            self.resolution_binning = resolution_binning
            
            # Initialize epoch counter properly
            self._epoch = 0
            
            # Initialize buckets efficiently
            self._initialize_buckets()
            
            logger.info(
                f"Initialized BucketSampler with {len(self.buckets)} buckets, "
                f"{self.total_samples} total samples"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize BucketSampler: {str(e)}")
            raise

    @property
    def epoch(self) -> int:
        """Get current epoch number"""
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set epoch number and clear caches"""
        self._epoch = value
        # Clear cached weights when epoch changes
        self._calculate_bucket_weights.cache_clear()

    @torch.no_grad()
    def _initialize_buckets(self):
        """Initialize buckets with memory optimization"""
        try:
            # Use defaultdict for automatic bucket creation
            from collections import defaultdict
            self.buckets = defaultdict(list)
            self.total_samples = 0
            
            # Process images in batches for memory efficiency
            batch_size = 1000
            for i in range(0, len(self.dataset), batch_size):
                batch_indices = range(i, min(i + batch_size, len(self.dataset)))
                
                for idx in batch_indices:
                    # Get image size efficiently
                    size = self.dataset.get_image_size(idx)
                    if size is None:
                        continue
                        
                    h, w = size
                    if self.resolution_binning:
                        # Round to nearest multiple of 64 for efficiency
                        h = ((h + 31) // 64) * 64
                        w = ((w + 31) // 64) * 64
                    
                    self.buckets[(h, w)].append(idx)
                    self.total_samples += 1
            
            # Convert defaultdict to regular dict
            self.buckets = dict(self.buckets)
            
            # Pre-calculate bucket weights
            self._calculate_bucket_weights()
            
        except Exception as e:
            logger.error(f"Failed to initialize buckets: {str(e)}")
            raise

    @lru_cache(maxsize=1)
    def _calculate_bucket_weights(self):
        """Cache bucket weights calculation"""
        try:
            self.bucket_weights = {
                res: len(indices) / self.total_samples 
                for res, indices in self.buckets.items()
            }
        except Exception as e:
            logger.error(f"Failed to calculate bucket weights: {str(e)}")
            raise

    def __iter__(self):
        """Memory-efficient iterator implementation"""
        try:
            # Use generator for memory efficiency
            generator = torch.Generator()
            if self.seed is not None:
                generator.manual_seed(self.seed + self._epoch)
            
            # Pre-allocate indices array
            indices = np.zeros(self.total_samples, dtype=np.int32)
            current_idx = 0
            
            # Process buckets efficiently
            for bucket_indices in self.buckets.values():
                bucket_size = len(bucket_indices)
                if self.shuffle:
                    # Use numpy for efficient shuffling
                    bucket_indices = np.random.permutation(bucket_indices)
                
                indices[current_idx:current_idx + bucket_size] = bucket_indices
                current_idx += bucket_size
            
            # Final shuffle if needed
            if self.shuffle:
                np.random.shuffle(indices)
            
            # Handle drop_last efficiently
            if self.drop_last:
                indices = indices[:(len(indices) // self.batch_size) * self.batch_size]
            
            return iter(indices)
            
        except Exception as e:
            logger.error(f"Failed to create iterator: {str(e)}")
            raise

    def __len__(self) -> int:
        """Efficient length calculation"""
        if self.drop_last:
            return self.total_samples // self.batch_size
        return (self.total_samples + self.batch_size - 1) // self.batch_size
    
class CustomDataLoader(CustomDataLoaderBase):
    """Optimized data loader with advanced batching and parallel processing"""
    
    def __init__(self, 
                 dataset: CustomDataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 sampler: Optional[CustomSamplerBase] = None,
                 batch_sampler: Optional[Sampler] = None,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: Optional[callable] = None,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        
        # Initialize base class first
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
        
        # Store attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Handle sampler configuration
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with other parameters')
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            
            self.sampler = sampler
            self.batch_sampler = torch.utils.data.BatchSampler(
                self.sampler,
                batch_size,
                drop_last
            )
        
        # Initialize worker components
        self.worker_pool = None
        self.prefetch_queue = None
        self.batch_queue = None
        self._stop_event = threading.Event()
        self._prefetch_thread = None
        self._iterator = None
        
        # Initialize workers if needed
        if num_workers > 0:
            self._initialize_workers()

    def _initialize_workers(self):
        """Initialize worker processes for parallel data loading"""
        if self.num_workers > 0:
            self.worker_pool = ProcessPoolExecutor(max_workers=self.num_workers)
            self.prefetch_queue = Queue(maxsize=self.prefetch_factor * self.num_workers)
            self.batch_queue = Queue(maxsize=self.prefetch_factor)
            
            # Start prefetch thread
            self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
            self._prefetch_thread.start()

    def _prefetch_worker(self):
        """Background thread for prefetching data"""
        try:
            batch_sampler_iter = iter(self.batch_sampler)
            while not self._stop_event.is_set():
                try:
                    # Get next batch of indices
                    indices = next(batch_sampler_iter)
                    
                    # Submit batch processing to worker pool
                    future = self.worker_pool.submit(
                        self.dataset.get_batch, indices
                    )
                    self.prefetch_queue.put(future)
                    
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Prefetch worker error: {str(e)}")
                    break
        finally:
            # Signal end of iteration
            self.prefetch_queue.put(None)

    def __iter__(self):
        """Return iterator over the dataset"""
        if not self._initialized:
            self.initialize()
            
        # Reset stop event
        self._stop_event.clear()
        
        # Initialize iteration state
        self._iterator = iter(self.batch_sampler)
        
        return self

    def __next__(self):
        """Get next batch of data with retry logic"""
        max_retries = 3
        for _ in range(max_retries):
            try:
                if self.num_workers > 0:
                    # Get prefetched batch
                    future = self.prefetch_queue.get()
                    if future is None:
                        raise StopIteration
                        
                    batch = future.result()
                    collated = self.dataset.collate_fn(batch)
                    if collated is not None:
                        return collated
                else:
                    # Single-threaded processing
                    indices = next(self._iterator)
                    batch = self.dataset.get_batch(indices)
                    collated = self.dataset.collate_fn(batch)
                    if collated is not None:
                        return collated
                    
            except StopIteration:
                self._stop_event.set()
                raise
                
        # If we get here, we've failed all retries
        logger.error("Failed to get valid batch after maximum retries")
        raise RuntimeError("Failed to get valid batch after maximum retries")

    def __len__(self):
        """Return the number of batches in the dataloader"""
        return len(self.batch_sampler)

    def __del__(self):
        """Cleanup resources"""
        self._stop_event.set()
        
        if hasattr(self, '_prefetch_thread') and self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
            
        if hasattr(self, 'worker_pool') and self.worker_pool is not None:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
            
        if hasattr(self, 'prefetch_queue'):
            while not self.prefetch_queue.empty():
                try:
                    self.prefetch_queue.get_nowait()
                except:
                    pass
            self.prefetch_queue = None
            
        if hasattr(self, 'batch_queue'):
            while not self.batch_queue.empty():
                try:
                    self.batch_queue.get_nowait()
                except:
                    pass
            self.batch_queue = None

    def cleanup(self):
        """Explicit cleanup method"""
        self.__del__()

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
    use_tag_weighting=False,
    **kwargs
):
    """
    Create a dataloader with the specified parameters
    """
    # Log initialization parameters
    logger.info("Creating dataloader with settings:")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  no_caching_latents: {no_caching_latents}")
    logger.info(f"  all_ar: {all_ar}")
    logger.info(f"  use_tag_weighting: {use_tag_weighting}")

    # Validate settings
    if all_ar and not enable_bucket_sampler:
        logger.warning("all_ar requires bucket_sampler - enabling bucket_sampler")
        enable_bucket_sampler = True
        
    if no_caching_latents:
        logger.info("Latent caching disabled - processing will be done on-the-fly")

    # Initialize dataset with progress reporting
    dataset = CustomDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        vae=vae,
        num_workers=num_workers,  # Pass num_workers to dataset
        enable_bucket_sampler=enable_bucket_sampler,
        no_caching_latents=no_caching_latents,
        all_ar=all_ar,
        use_tag_weighting=use_tag_weighting,
        **kwargs
    )
    
    logger.info("Setting up bucket sampler...")
    if enable_bucket_sampler:
        sampler = BucketSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True
        )
        logger.info(f"Bucket sampler initialized with {len(sampler)} samples")
    else:
        sampler = None
        logger.info("No bucket sampler used")

    logger.info("Initializing dataloader...")
    dataloader = CustomDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers if num_workers is not None else 0,
        pin_memory=True,
        drop_last=True,
        timeout=0,
        prefetch_factor=2
    )
    logger.info(f"Dataloader initialized with {len(dataloader)} batches")
    
    return dataloader