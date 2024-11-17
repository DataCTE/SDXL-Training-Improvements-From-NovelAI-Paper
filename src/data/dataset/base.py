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

Classes:
    DatasetError: Base exception for dataset related errors
    CustomDatasetBase: Abstract base class for datasets
    CustomSamplerBase: Abstract base class for samplers
    CustomDataLoaderBase: Abstract base class for data loaders
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Sequence, Tuple, TypeVar, Union
)

import torch
from torch.utils.data import Sampler

# Type variables for generic types
T = TypeVar('T')
BatchType = TypeVar('BatchType')
SamplerState = Dict[str, Any]


class DatasetError(Exception):
    """Base exception for dataset related errors."""


class InitializationError(DatasetError):
    """Exception raised when initialization fails."""


class CustomDatasetBase(ABC, Generic[T]):
    """Abstract base class for custom datasets with enhanced functionality.
    
    This class provides a foundation for implementing custom datasets with
    features like lazy initialization, batch fetching, and prefetching.
    
    Attributes:
        _length: Optional cached length of the dataset
        _initialized: Whether the dataset has been initialized
        
    Note:
        Subclasses must implement __len__ and __getitem__ methods.
    """
    
    def __init__(self) -> None:
        """Initialize the dataset base class."""
        self._length: Optional[int] = None
        self._initialized: bool = False
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset.
        
        Returns:
            Number of items in the dataset
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx: Union[int, slice]) -> T:
        """Get a single item or slice of items from the dataset.
        
        Args:
            idx: Index or slice to retrieve
            
        Returns:
            Dataset item(s) at the specified index/slice
            
        Raises:
            NotImplementedError: Must be implemented by subclass
            IndexError: If index is out of bounds
        """
        raise NotImplementedError
    
    def initialize(self) -> None:
        """Optional initialization method for lazy loading.
        
        This method should be called before using the dataset if it requires
        any setup or resource allocation.
        
        Raises:
            InitializationError: If initialization fails
        """
        try:
            self._initialized = True
        except Exception as error:
            raise InitializationError("Dataset initialization failed") from error
    
    @property
    def is_initialized(self) -> bool:
        """Check if dataset has been initialized.
        
        Returns:
            True if dataset is initialized, False otherwise
        """
        return self._initialized
    
    def get_batch(self, indices: Sequence[int]) -> List[T]:
        """Efficiently get multiple items at once.
        
        Args:
            indices: Sequence of indices to retrieve
            
        Returns:
            List of dataset items at the specified indices
            
        Raises:
            IndexError: If any index is out of bounds
        """
        try:
            return [self[idx] for idx in indices]
        except IndexError as error:
            raise IndexError(f"Invalid indices in batch: {error}") from error
    
    def prefetch(self, indices: Sequence[int]) -> None:
        """Optional prefetch method for optimization.
        
        This method can be implemented to preload data that will be needed
        soon, improving data loading performance.
        
        Args:
            indices: Sequence of indices to prefetch
            
        Note:
            Default implementation does nothing
        """
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method for releasing resources.
        
        This method should be called when the dataset is no longer needed
        to free any allocated resources.
        """
        pass


class CustomSamplerBase(ABC, Generic[T]):
    """Abstract base class for custom samplers with enhanced functionality.
    
    This class provides a foundation for implementing custom samplers with
    features like state management and epoch tracking.
    
    Attributes:
        data_source: Dataset to sample from
        _iterator: Current iterator instance
        _epoch: Current epoch number
        
    Note:
        Subclasses must implement __iter__ and __len__ methods.
    """
    
    def __init__(self, data_source: CustomDatasetBase[T]) -> None:
        """Initialize the sampler.
        
        Args:
            data_source: Dataset to sample from
        """
        self.data_source = data_source
        self._iterator: Optional[Iterator[int]] = None
        self._epoch: int = 0
    
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Return iterator over dataset indices.
        
        Returns:
            Iterator yielding dataset indices
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the sampler.
        
        Returns:
            Number of samples
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducibility.
        
        Args:
            epoch: Epoch number to set
            
        Raises:
            ValueError: If epoch is negative
        """
        if epoch < 0:
            raise ValueError("Epoch number cannot be negative")
        self._epoch = epoch
    
    @property
    def epoch(self) -> int:
        """Get current epoch.
        
        Returns:
            Current epoch number
        """
        return self._epoch
    
    def state_dict(self) -> SamplerState:
        """Get sampler state for checkpointing.
        
        Returns:
            Dictionary containing sampler state
        """
        return {
            'epoch': self._epoch,
            'iterator_state': getattr(self._iterator, 'state', None)
        }
    
    def load_state_dict(self, state_dict: SamplerState) -> None:
        """Load sampler state from checkpoint.
        
        Args:
            state_dict: Dictionary containing sampler state
        """
        self._epoch = state_dict.get('epoch', 0)
        if hasattr(self._iterator, 'load_state'):
            self._iterator.load_state(state_dict.get('iterator_state'))


class CustomDataLoaderBase(ABC, Generic[T, BatchType]):
    """Abstract base class for custom data loaders with enhanced functionality.
    
    This class provides a foundation for implementing custom data loaders with
    features like worker management, batch sampling, and state management.
    
    Attributes:
        dataset: Source dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        timeout: Timeout for queue operations
        worker_init_fn: Function to initialize workers
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs
        batch_sampler: Sampler for batches
        sampler: Sampler for individual samples
        
    Note:
        Subclasses must implement __iter__ method.
    """
    
    def __init__(
        self, 
        dataset: CustomDatasetBase[T],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[CustomSamplerBase[T]] = None,
        batch_sampler: Optional[Sampler[List[int]]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
    ) -> None:
        """Initialize the data loader.
        
        Args:
            dataset: Source dataset
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
            
        Raises:
            ValueError: If parameters are invalid
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        
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
        self.batch_sampler: Optional[Sampler[List[int]]] = None
        self.sampler: Optional[Union[Sampler[int], CustomSamplerBase[T]]] = None
        
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
        self._iterator: Optional[Iterator[BatchType]] = None
    
    @abstractmethod
    def __iter__(self) -> Iterator[BatchType]:
        """Return iterator over the dataset.
        
        Returns:
            Iterator yielding batches of data
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of batches in the dataloader.
        
        Returns:
            Number of batches
        """
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    def initialize(self) -> None:
        """Initialize the dataloader and its dataset.
        
        This method ensures the dataset is initialized before use.
        
        Raises:
            InitializationError: If initialization fails
        """
        if not self._initialized:
            try:
                if not self.dataset.is_initialized:
                    self.dataset.initialize()
                self._initialized = True
            except Exception as error:
                raise InitializationError("DataLoader initialization failed") from error
    
    def cleanup(self) -> None:
        """Cleanup resources.
        
        This method ensures proper cleanup of all resources including
        the iterator and dataset.
        """
        if self._iterator is not None:
            del self._iterator
            self._iterator = None
        self.dataset.cleanup()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get dataloader state for checkpointing.
        
        Returns:
            Dictionary containing dataloader state
        """
        return {
            'initialized': self._initialized,
            'iterator_state': getattr(self._iterator, 'state', None) if self._iterator else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load dataloader state from checkpoint.
        
        Args:
            state_dict: Dictionary containing dataloader state
        """
        self._initialized = state_dict.get('initialized', False)
        if self._iterator and hasattr(self._iterator, 'load_state'):
            self._iterator.load_state(state_dict.get('iterator_state'))
