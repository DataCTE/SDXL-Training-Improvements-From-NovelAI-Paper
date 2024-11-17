"""Custom data loader implementation for efficient batch processing.

This module provides an optimized data loader with advanced features like
parallel processing, prefetching, and efficient memory management. It is
designed specifically for handling large image datasets with varying
resolutions.

Classes:
    DataLoaderError: Base exception for data loader errors
    CustomDataLoader: Main data loader implementation
    
Functions:
    create_dataloader: Factory function to create configured data loader
"""

import logging
import threading
from queue import Empty, Queue
from typing import (
    Any, Callable, Dict, Iterator, List, Optional,
    Tuple, Union, TypeVar, Generic
)

import torch
from torch.utils.data import Sampler

from .base import CustomDatasetBase, CustomSamplerBase, CustomDataLoaderBase

logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')
BatchType = TypeVar('BatchType')


class DataLoaderError(Exception):
    """Base exception for data loader related errors."""


class WorkerError(DataLoaderError):
    """Exception raised when a worker process fails."""


class CustomDataLoader(CustomDataLoaderBase, Generic[BatchType]):
    """Optimized data loader with advanced batching and parallel processing.
    
    This data loader provides efficient data loading with features like:
    - Parallel processing with multiple workers
    - Data prefetching
    - Memory pinning for faster GPU transfer
    - Custom batch sampling
    - Automatic resource cleanup
    
    Attributes:
        dataset: Source dataset to load data from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        sampler: Custom sampler for batch indices
        batch_sampler: Custom sampler for full batches
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        timeout: Timeout for queue operations
        worker_init_fn: Function to initialize workers
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs
        collate_fn: Function to collate samples into batches
    """
    
    def __init__(
        self,
        dataset: CustomDatasetBase,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[CustomSamplerBase] = None,
        batch_sampler: Optional[Sampler[List[int]]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable[[List[T]], BatchType]] = None
    ) -> None:
        """Initialize the data loader.
        
        Args:
            dataset: Source dataset to load data from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            sampler: Custom sampler for batch indices
            batch_sampler: Custom sampler for full batches
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            timeout: Timeout for queue operations
            worker_init_fn: Function to initialize workers
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Whether to keep workers alive between epochs
            collate_fn: Function to collate samples into batches
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
            
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
        
        # Store collate function
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate
        
        # Initialize worker components
        self.worker_pool: Optional[List[torch.multiprocessing.Process]] = None
        self.prefetch_queue: Optional[Queue[List[int]]] = None
        self.batch_queue: Optional[Queue[BatchType]] = None
        self._stop_event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._iterator: Optional[Iterator[List[int]]] = None
        
        # Initialize workers if needed
        if num_workers > 0:
            self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize worker processes for parallel data loading.
        
        This method sets up the worker processes and queues for parallel
        data loading and prefetching.
        
        Raises:
            RuntimeError: If worker initialization fails
        """
        try:
            if self.num_workers > 0:
                self.prefetch_queue = Queue(
                    maxsize=self.prefetch_factor * self.num_workers
                )
                self.batch_queue = Queue(maxsize=2)
                
                # Start prefetch thread
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker,
                    daemon=True
                )
                self._prefetch_thread.start()
                
                # Initialize worker processes
                import torch.multiprocessing as mp
                self.worker_pool = []
                
                for worker_id in range(self.num_workers):
                    worker = mp.Process(
                        target=self._worker_loop,
                        args=(worker_id,),
                        daemon=True
                    )
                    worker.start()
                    self.worker_pool.append(worker)
                    
        except Exception as error:
            logger.error("Failed to initialize workers: %s", str(error))
            self.cleanup()
            raise RuntimeError("Worker initialization failed") from error
    
    def _prefetch_worker(self) -> None:
        """Background thread for prefetching data.
        
        This method runs in a background thread and prefetches batch indices
        for the worker processes.
        
        Raises:
            WorkerError: If prefetching fails
        """
        try:
            while not self._stop_event.is_set():
                try:
                    # Get next batch indices
                    indices = next(self._iterator)
                    
                    # Prefetch data
                    self.dataset.prefetch(indices)
                    
                    # Put in queue
                    if not self._stop_event.is_set():
                        self.prefetch_queue.put(indices)
                        
                except StopIteration:
                    break
                    
        except Exception as error:
            logger.error("Error in prefetch worker: %s", str(error))
            raise WorkerError("Prefetch worker failed") from error
    
    def _worker_loop(self, worker_id: int) -> None:
        """Main worker process loop.
        
        This method runs in each worker process and processes batches of data.
        
        Args:
            worker_id: ID of the worker process
            
        Raises:
            WorkerError: If worker processing fails
        """
        try:
            # Initialize worker
            if self.worker_init_fn is not None:
                self.worker_init_fn(worker_id)
            
            while not self._stop_event.is_set():
                try:
                    # Get indices from prefetch queue
                    indices = self.prefetch_queue.get(timeout=self.timeout)
                    
                    # Process batch
                    batch = [self.dataset[idx] for idx in indices]
                    
                    # Apply collate if needed
                    if self.collate_fn is not None:
                        batch = self.collate_fn(batch)
                    
                    # Put processed batch in queue
                    if not self._stop_event.is_set():
                        self.batch_queue.put(batch)
                    
                except Empty:
                    continue
                    
        except Exception as error:
            logger.error("Worker %d failed: %s", worker_id, str(error))
            raise WorkerError(f"Worker {worker_id} failed") from error
    
    def __iter__(self) -> Iterator[BatchType]:
        """Return iterator over the dataset.
        
        Returns:
            Iterator yielding batches of data
            
        Note:
            If shuffle is enabled and the dataset supports it, the data
            will be shuffled before iteration.
        """
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
    
    @staticmethod
    def _pin_memory(data: T) -> T:
        """Pin memory for faster GPU transfer.
        
        Args:
            data: Data to pin (tensor, list, tuple, or dict)
            
        Returns:
            Data with memory pinned
            
        Note:
            This operation is recursive for nested data structures.
        """
        if torch.is_tensor(data):
            return data.pin_memory()
            
        elif isinstance(data, dict):
            return {
                k: CustomDataLoader._pin_memory(v)
                for k, v in data.items()
            }
            
        elif isinstance(data, (tuple, list)):
            return type(data)(CustomDataLoader._pin_memory(x) for x in data)
            
        return data
    
    def __next__(self) -> BatchType:
        """Get next batch of data with retry logic.
        
        Returns:
            Next batch of data
            
        Raises:
            StopIteration: When iteration is complete
            DataLoaderError: If batch retrieval fails
        """
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
    
    def __len__(self) -> int:
        """Return the number of batches in the dataloader.
        
        Returns:
            Number of batches that will be yielded
        """
        return len(self.batch_sampler)
    
    def __del__(self) -> None:
        """Cleanup resources when object is deleted."""
        self.cleanup()
    
    def cleanup(self) -> None:
        """Explicit cleanup method for resources.
        
        This method ensures proper cleanup of all resources including
        worker processes, queues, and threads.
        """
        # Stop prefetch thread
        if self._prefetch_thread is not None:
            self._stop_event.set()
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None
        
        # Clear queues
        for queue in [self.prefetch_queue, self.batch_queue]:
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
                worker.join(timeout=1.0)
            self.worker_pool = None
        
        super().cleanup()
    
    @staticmethod
    def _default_collate(batch: List[T]) -> Union[torch.Tensor, List[T], Dict[str, Any]]:
        """Default collate function for batching data.
        
        This method handles basic batching of tensors and other data types,
        supporting nested structures like dictionaries and lists.
        
        Args:
            batch: List of data items from dataset
            
        Returns:
            Batched data in appropriate format
            
        Note:
            Supports tensors, strings, dictionaries, and nested structures.
        """
        if not batch:
            return {}
            
        elem = batch[0]
        if torch.is_tensor(elem):
            return torch.stack(batch, 0)
            
        elif isinstance(elem, (str, bytes)):
            return batch
            
        elif isinstance(elem, dict):
            return {
                key: CustomDataLoader._default_collate([d[key] for d in batch])
                for key in elem
            }
            
        elif isinstance(elem, (tuple, list)):
            transposed = zip(*batch)
            return [CustomDataLoader._default_collate(samples) for samples in transposed]
            
        return batch


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
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
    max_size: int = 4096,
    bucket_step_size: int = 64,
    max_bucket_area: int = 1024*1024,
    token_dropout_rate: float = 0.1,
    caption_dropout_rate: float = 0.1,
    min_tag_weight: float = 0.1,
    max_tag_weight: float = 3.0,
    use_tag_weighting: bool = True,
    **kwargs: Any
) -> CustomDataLoader:
    """Create a configured data loader instance.
    
    This factory function creates and configures a CustomDataLoader with
    the specified parameters and appropriate dataset setup.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes (None for auto)
        no_caching_latents: Whether to disable latent caching
        all_ar: Whether to use aspect ratio based bucketing
        cache_dir: Directory for caching latents
        vae: VAE model for encoding images
        tokenizer: Primary tokenizer for text
        tokenizer_2: Secondary tokenizer for text
        text_encoder: Primary text encoder
        text_encoder_2: Secondary text encoder
        min_size: Minimum image size
        max_size: Maximum image size
        bucket_step_size: Resolution step size for buckets
        max_bucket_area: Maximum area for resolution buckets
        token_dropout_rate: Rate for token dropout
        caption_dropout_rate: Rate for caption dropout
        min_tag_weight: Minimum weight for tags
        max_tag_weight: Maximum weight for tags
        use_tag_weighting: Whether to use tag weighting
        **kwargs: Additional arguments for dataset
        
    Returns:
        Configured CustomDataLoader instance
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If loader creation fails
    """
    from .dataset import CustomDataset
    
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
    
    # Create dataloader
    dataloader = CustomDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers is not None else 0,
        pin_memory=True,
        **kwargs
    )
    
    return dataloader
