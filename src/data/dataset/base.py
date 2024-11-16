from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Sequence, Union, Dict
from torch.utils.data import Sampler
import torch

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
        """Optional prefetch method for optimization
        
        Args:
            indices: Sequence of indices to prefetch
        """
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method for releasing resources"""
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
            'iterator_state': getattr(self._iterator, 'state', None) if self._iterator else None
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load dataloader state from checkpoint"""
        self._initialized = state_dict.get('initialized', False)
        if self._iterator and hasattr(self._iterator, 'load_state'):
            self._iterator.load_state(state_dict.get('iterator_state'))
