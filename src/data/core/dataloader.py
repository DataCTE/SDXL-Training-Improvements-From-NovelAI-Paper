"""Custom data loader implementation for efficient batch processing.

This module provides an optimized data loader with advanced features like
parallel processing, prefetching, and efficient memory management. It is
designed specifically for handling large image datasets with varying
resolutions.
"""

import logging
import threading
from queue import Empty, Queue, Full
from typing import (
    Any, Callable, Dict, Iterator, List, Optional,
    Tuple, Union, TypeVar, Generic
)
import torch
import torch.multiprocessing as mp
from torch.utils.data import Sampler
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from .base import (
    CustomDatasetBase, CustomSamplerBase, CustomDataLoaderBase,
    DataLoaderConfig, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
BatchType = TypeVar('BatchType')

# Constants
QUEUE_JOIN_TIMEOUT = 1.0
MAX_QUEUE_SIZE = 50
WORKER_TIMEOUT = 5.0

class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass

class WorkerError(DataLoaderError):
    """Exception raised when worker processing fails."""
    pass

class CustomDataLoader(CustomDataLoaderBase[T, BatchType]):
    """Optimized data loader with parallel processing and prefetching."""
    
    __slots__ = (
        'collate_fn', 'worker_pool', 'prefetch_queue',
        'batch_queue', '_stop_event', '_prefetch_thread',
        '_worker_queues', '_cache'
    )
    
    def __init__(
        self,
        dataset: CustomDatasetBase,
        config: Optional[DataLoaderConfig] = None,
        collate_fn: Optional[Callable[[List[T]], BatchType]] = None
    ) -> None:
        """Initialize the optimized data loader."""
        super().__init__(dataset=dataset, config=config)
        
        # Store collate function
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate
        
        # Initialize worker components
        self.worker_pool: Optional[List[mp.Process]] = None
        self.prefetch_queue: Optional[Queue[List[int]]] = None
        self.batch_queue: Optional[Queue[BatchType]] = None
        self._stop_event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._worker_queues: List[Queue[List[int]]] = []
        self._cache: Dict[int, BatchType] = {}
        
        # Initialize workers if needed
        if self.num_workers > 0:
            self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize worker processes with load balancing."""
        try:
            if self.num_workers > 0:
                # Create queues
                self.prefetch_queue = Queue(
                    maxsize=self.prefetch_factor * self.num_workers
                )
                self.batch_queue = Queue(maxsize=MAX_QUEUE_SIZE)
                
                # Create worker queues for load balancing
                self._worker_queues = [
                    Queue(maxsize=self.prefetch_factor)
                    for _ in range(self.num_workers)
                ]
                
                # Start prefetch thread
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker,
                    daemon=True
                )
                self._prefetch_thread.start()
                
                # Initialize worker processes
                self.worker_pool = []
                for worker_id in range(self.num_workers):
                    worker = mp.Process(
                        target=self._worker_loop,
                        args=(worker_id, self._worker_queues[worker_id]),
                        daemon=True
                    )
                    worker.start()
                    self.worker_pool.append(worker)
                
        except Exception as error:
            logger.error("Failed to initialize workers: %s", str(error))
            self.cleanup()
            raise RuntimeError("Worker initialization failed") from error
    
    def _prefetch_worker(self) -> None:
        """Prefetch batches with load balancing."""
        try:
            worker_idx = 0
            while not self._stop_event.is_set():
                try:
                    # Get next batch indices
                    indices = next(self._iterator)
                    
                    # Prefetch data
                    self.dataset.prefetch(indices)
                    
                    # Distribute work using round-robin
                    if not self._stop_event.is_set():
                        self._worker_queues[worker_idx].put(
                            indices,
                            timeout=WORKER_TIMEOUT
                        )
                        worker_idx = (worker_idx + 1) % self.num_workers
                        
                except StopIteration:
                    break
                except Full:
                    continue
                
        except Exception as error:
            logger.error("Error in prefetch worker: %s", str(error))
            raise WorkerError("Prefetch worker failed") from error
    
    def _worker_loop(self, worker_id: int, queue: Queue[List[int]]) -> None:
        """Process batches with error handling and memory optimization."""
        try:
            # Initialize worker
            if self.worker_init_fn is not None:
                self.worker_init_fn(worker_id)
            
            # Set up CUDA for this process if needed
            if torch.cuda.is_available():
                torch.cuda.set_device(worker_id % torch.cuda.device_count())
            
            while not self._stop_event.is_set():
                try:
                    # Get indices from worker queue
                    indices = queue.get(timeout=self.timeout)
                    
                    # Process batch with memory optimization
                    batch = []
                    for idx in indices:
                        # Check cache first
                        if idx in self._cache:
                            item = self._cache[idx]
                        else:
                            item = self.dataset[idx]
                            # Cache only if memory usage is reasonable
                            if len(self._cache) < MAX_QUEUE_SIZE:
                                self._cache[idx] = item
                        batch.append(item)
                    
                    # Apply collate if needed
                    if self.collate_fn is not None:
                        batch = self.collate_fn(batch)
                    
                    # Put processed batch in queue
                    if not self._stop_event.is_set():
                        self.batch_queue.put(batch, timeout=WORKER_TIMEOUT)
                    
                except Empty:
                    continue
                except Full:
                    continue
                
        except Exception as error:
            logger.error("Worker %d failed: %s", worker_id, str(error))
            raise WorkerError(f"Worker {worker_id} failed") from error
    
    def __iter__(self) -> Iterator[BatchType]:
        """Return optimized iterator over the dataset."""
        # Initialize if needed
        self.initialize()
        
        # Shuffle dataset if enabled
        should_shuffle = (
            hasattr(self.sampler, 'shuffle') and 
            getattr(self.sampler, 'shuffle', False)
        )
        if should_shuffle and hasattr(self.dataset, 'shuffle_dataset'):
            # Use epoch from sampler if available
            seed = getattr(self.sampler, '_epoch', None)
            self.dataset.shuffle_dataset(seed=seed)
        
        # Get iterator from sampler
        self._iterator = iter(self.batch_sampler)
        
        return self
    
    def __next__(self) -> BatchType:
        """Get next batch with optimized memory handling."""
        try:
            # Get batch
            if self.num_workers == 0:
                indices = next(self._iterator)
                batch = self.dataset.get_batch(indices)
            else:
                batch = self.batch_queue.get(timeout=self.timeout)
            
            # Apply pin memory if needed
            if self.pin_memory:
                batch = self._pin_memory(batch)
            
            return batch
            
        except StopIteration:
            self.cleanup()
            raise
        except Exception as error:
            logger.error("Error getting next batch: %s", str(error))
            self.cleanup()
            raise DataLoaderError("Failed to get next batch") from error
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _pin_memory(data: T) -> T:
        """Pin memory with caching for faster GPU transfer."""
        if torch.is_tensor(data):
            return data.pin_memory()
            
        elif isinstance(data, dict):
            return {
                k: CustomDataLoader._pin_memory(v)
                for k, v in data.items()
            }
            
        elif isinstance(data, (tuple, list)):
            return type(data)(
                CustomDataLoader._pin_memory(x) for x in data
            )
            
        return data
    
    @staticmethod
    def _default_collate(
        batch: List[T]
    ) -> Union[torch.Tensor, List[T], Dict[str, Any]]:
        """Optimized collate function with type checking."""
        if not batch:
            return {}
            
        elem = batch[0]
        if torch.is_tensor(elem):
            try:
                return torch.stack(batch, 0)
            except Exception:
                return batch
            
        elif isinstance(elem, (str, bytes)):
            return batch
            
        elif isinstance(elem, dict):
            return {
                key: CustomDataLoader._default_collate(
                    [d[key] for d in batch]
                )
                for key in elem
            }
            
        elif isinstance(elem, (tuple, list)):
            transposed = zip(*batch)
            return [
                CustomDataLoader._default_collate(samples)
                for samples in transposed
            ]
            
        return batch
    
    def cleanup(self) -> None:
        """Cleanup resources with proper synchronization."""
        # Stop prefetch thread
        if self._prefetch_thread is not None:
            self._stop_event.set()
            self._prefetch_thread.join(timeout=QUEUE_JOIN_TIMEOUT)
            self._prefetch_thread = None
        
        # Clear queues
        for queue in [self.prefetch_queue, self.batch_queue, *self._worker_queues]:
            if queue is not None:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
        
        # Cleanup worker pool
        if self.worker_pool is not None:
            for worker in self.worker_pool:
                worker.terminate()
                worker.join(timeout=QUEUE_JOIN_TIMEOUT)
            self.worker_pool = None
        
        # Clear caches
        self._cache.clear()
        
        super().cleanup()
    
    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.cleanup()

def create_dataloader(
    data_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: Optional[int] = None,
    no_caching_latents: bool = False,
    all_ar: bool = False,
    cache_dir: str = "latents_cache",
    vae: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_2: Optional[Any] = None,
    text_encoder: Optional[Any] = None,
    text_encoder_2: Optional[Any] = None,
    min_size: int = 512,
    max_size: int = 2048,
    bucket_step_size: int = 64,
    max_bucket_area: int = 1024*1024,
    token_dropout_rate: float = 0.1,
    caption_dropout_rate: float = 0.1,
    min_tag_weight: float = 0.1,
    max_tag_weight: float = 3.0,
    use_tag_weighting: bool = True,
    **kwargs: Any
) -> CustomDataLoader:
    """Create optimized data loader with smart defaults."""
    from src.data.core.dataset import CustomDataset
    
    # Create dataset
    dataset = CustomDataset(
        data_dir=data_dir,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        cache_dir=cache_dir,
        no_caching_latents=no_caching_latents,
        all_ar=all_ar,
        min_size=min_size,
        max_size=max_size,
        bucket_step_size=bucket_step_size,
        max_bucket_area=max_bucket_area,
        token_dropout_rate=token_dropout_rate,
        caption_dropout_rate=caption_dropout_rate,
        min_tag_weight=min_tag_weight,
        max_tag_weight=max_tag_weight,
        use_tag_weighting=use_tag_weighting,
        **kwargs
    )
    
    # Create optimized config
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers if num_workers is not None else DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Create dataloader with optimized settings
    dataloader = CustomDataLoader(
        dataset=dataset,
        config=config
    )
    
    return dataloader
