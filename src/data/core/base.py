"""Base classes for the custom dataset implementation.

This module provides abstract base classes for implementing custom datasets,
samplers, and data loaders with enhanced functionality for machine learning
tasks, particularly focused on image generation with SDXL.

The module includes three main abstract base classes:
- CustomDatasetBase: Base class for datasets with advanced features
- CustomSamplerBase: Base class for samplers with state management
- CustomDataLoaderBase: Base class for data loaders with worker management

These classes provide a foundation for building efficient and feature-rich
data loading pipelines with proper type checking and error handling.
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Sequence, Tuple, TypeVar, Union, Final
)
from dataclasses import dataclass, field
import threading
from queue import Queue
import torch
from torch.utils.data import Sampler
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')
BatchType = TypeVar('BatchType')
SamplerState = Dict[str, Any]

# Constants
DEFAULT_CACHE_SIZE: Final[int] = 1024
DEFAULT_PREFETCH_SIZE: Final[int] = 2
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_WORKERS: Final[int] = 4

@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset configuration."""
    cache_size: int = DEFAULT_CACHE_SIZE
    prefetch_size: int = DEFAULT_PREFETCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = True
    use_cuda: bool = True
    mixed_precision: bool = True

class DatasetError(Exception):
    """Base exception for dataset related errors."""
    pass

class InitializationError(DatasetError):
    """Exception raised when initialization fails."""
    pass

class CustomDatasetBase(Generic[T], ABC):
    """Abstract base class for datasets with advanced features."""
    
    __slots__ = ('_length', '_initialized', '_config', '_cache', '_prefetch_queue', '_executor')
    
    def __init__(
        self,
        config: Optional[DatasetConfig] = None
    ) -> None:
        """Initialize the dataset base class."""
        self._config = config or DatasetConfig()
        self._length: Optional[int] = None
        self._initialized: bool = False
        self._cache: Dict[int, T] = {}
        self._prefetch_queue: Queue[int] = Queue(maxsize=self._config.prefetch_size)
        self._executor = ThreadPoolExecutor(max_workers=self._config.num_workers)
    
    @property
    def is_initialized(self) -> bool:
        """Check if dataset has been initialized."""
        return self._initialized
    
    def initialize(self) -> None:
        """Optional initialization method for lazy loading."""
        try:
            self._initialized = True
        except Exception as error:
            raise InitializationError("Dataset initialization failed") from error
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx: Union[int, slice]) -> T:
        """Get a single item or slice of items from the dataset."""
        raise NotImplementedError
    
    def get_batch(self, indices: Sequence[int]) -> List[T]:
        """Efficiently get multiple items at once with caching."""
        try:
            # Check cache first
            batch = []
            missing_indices = []
            
            for idx in indices:
                if idx in self._cache:
                    batch.append(self._cache[idx])
                else:
                    missing_indices.append(idx)
            
            # Load missing items in parallel
            if missing_indices:
                futures = [
                    self._executor.submit(self.__getitem__, idx)
                    for idx in missing_indices
                ]
                items = [future.result() for future in futures]
                
                # Update cache and batch
                for idx, item in zip(missing_indices, items):
                    self._cache[idx] = item
                    batch.append(item)
                
                # Maintain cache size
                if len(self._cache) > self._config.cache_size:
                    # Remove oldest items
                    remove_count = len(self._cache) - self._config.cache_size
                    for _ in range(remove_count):
                        self._cache.pop(next(iter(self._cache)))
            
            return batch
            
        except Exception as error:
            logger.error("Failed to get batch: %s", str(error))
            raise DatasetError(f"Failed to get batch: {error}") from error
    
    def prefetch(self, indices: Sequence[int]) -> None:
        """Prefetch items into cache for future use."""
        try:
            for idx in indices:
                if idx not in self._cache and not self._prefetch_queue.full():
                    self._prefetch_queue.put_nowait(idx)
            
            # Start prefetch in background
            def _prefetch_worker():
                while not self._prefetch_queue.empty():
                    idx = self._prefetch_queue.get_nowait()
                    if idx not in self._cache:
                        try:
                            self._cache[idx] = self.__getitem__(idx)
                        except Exception as e:
                            logger.warning("Failed to prefetch item %d: %s", idx, str(e))
            
            self._executor.submit(_prefetch_worker)
            
        except Exception as error:
            logger.warning("Prefetch failed: %s", str(error))
    
    def cleanup(self) -> None:
        """Release resources."""
        self._cache.clear()
        self._executor.shutdown(wait=False)

class CustomSamplerBase(Generic[T], ABC):
    """Abstract base class for samplers with state management."""
    
    __slots__ = ('data_source', '_iterator', '_epoch', '_state')
    
    def __init__(self, data_source: CustomDatasetBase[T]) -> None:
        """Initialize the sampler."""
        self.data_source = data_source
        self._iterator: Optional[Iterator[int]] = None
        self._epoch: int = 0
        self._state: SamplerState = {}
    
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Return iterator over indices."""
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
    
    def state_dict(self) -> SamplerState:
        """Get sampler state for checkpointing."""
        return {
            'epoch': self._epoch,
            'state': self._state
        }
    
    def load_state_dict(self, state_dict: SamplerState) -> None:
        """Load sampler state from checkpoint."""
        self._epoch = state_dict.get('epoch', 0)
        self._state = state_dict.get('state', {})

@dataclass
class DataLoaderConfig:
    """Configuration for data loader."""
    batch_size: int = DEFAULT_BATCH_SIZE
    shuffle: bool = True
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = True
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    worker_init_fn: Optional[Callable[[int], None]] = None

class CustomDataLoaderBase(Generic[T, BatchType], ABC):
    """Abstract base class for data loaders with worker management."""
    
    __slots__ = (
        'config', 'dataset', 'batch_size', 'num_workers',
        'pin_memory', 'drop_last', 'timeout', 'worker_init_fn',
        'prefetch_factor', 'persistent_workers', 'batch_sampler',
        'sampler', '_initialized', '_iterator'
    )
    
    def __init__(
        self,
        dataset: CustomDatasetBase[T],
        config: Optional[DataLoaderConfig] = None
    ) -> None:
        """Initialize the data loader."""
        self.config = config or DataLoaderConfig()
        
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.config.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.config.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if self.config.timeout < 0:
            raise ValueError("timeout cannot be negative")
        
        self.dataset = dataset
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.drop_last = self.config.drop_last
        self.timeout = self.config.timeout
        self.worker_init_fn = self.config.worker_init_fn
        self.prefetch_factor = self.config.prefetch_factor
        self.persistent_workers = self.config.persistent_workers
        
        # Initialize samplers
        self.batch_sampler: Optional[Sampler[List[int]]] = None
        self.sampler: Optional[Union[Sampler[int], CustomSamplerBase[T]]] = None
        
        if self.config.shuffle:
            self.sampler = torch.utils.data.RandomSampler(dataset)
        else:
            self.sampler = torch.utils.data.SequentialSampler(dataset)
        
        self.batch_sampler = torch.utils.data.BatchSampler(
            self.sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last
        )
        
        self._initialized = False
        self._iterator: Optional[Iterator[BatchType]] = None
    
    def initialize(self) -> None:
        """Initialize the dataloader and dataset."""
        if not self._initialized:
            try:
                if not self.dataset.is_initialized:
                    self.dataset.initialize()
                self._initialized = True
            except Exception as error:
                raise InitializationError("DataLoader initialization failed") from error
    
    @abstractmethod
    def __iter__(self) -> Iterator[BatchType]:
        """Return iterator over the dataset."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    def state_dict(self) -> Dict[str, Any]:
        """Get dataloader state for checkpointing."""
        return {
            'initialized': self._initialized,
            'iterator_state': getattr(self._iterator, 'state', None) if self._iterator else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""
        self._initialized = state_dict.get('initialized', False)
        if self._iterator and hasattr(self._iterator, 'load_state'):
            self._iterator.load_state(state_dict.get('iterator_state'))
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._iterator is not None:
            del self._iterator
            self._iterator = None
        self.dataset.cleanup()
