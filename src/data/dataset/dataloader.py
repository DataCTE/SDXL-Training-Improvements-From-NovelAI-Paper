import threading
import torch
from typing import Optional, Iterator
from queue import Queue, Empty
from .base import CustomDatasetBase, CustomSamplerBase, CustomDataLoaderBase
from torch.utils.data import Sampler
import logging

logger = logging.getLogger(__name__)

class CustomDataLoader(CustomDataLoaderBase):
    """Optimized data loader with advanced batching and parallel processing"""
    
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
                 persistent_workers: bool = False,
                 collate_fn: Optional[callable] = None):
        
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
        
        # Store collate function
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate
        
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
            self.prefetch_queue = Queue(maxsize=self.prefetch_factor * self.num_workers)
            self.batch_queue = Queue(maxsize=2)
            
            # Start prefetch thread
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self._prefetch_thread.start()
            
            # Initialize worker processes using torch.multiprocessing
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
    
    def _prefetch_worker(self):
        """Background thread for prefetching data"""
        try:
            while not self._stop_event.is_set():
                # Get next batch indices
                if self.batch_sampler is not None:
                    indices = next(self._iterator)
                else:
                    indices = [next(self._iterator)]
                
                # Prefetch data
                self.dataset.prefetch(indices)
                
                # Put in queue
                self.prefetch_queue.put(indices)
                
        except StopIteration:
            pass
        except Exception as e:
            logger.error("Error in prefetch worker: %s", str(e))
            raise
    
    def _worker_loop(self, worker_id: int):
        """Main worker process loop"""
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
                    self.batch_queue.put(batch)
                    
                except Empty:
                    continue
                    
        except Exception as e:
            logger.error("Worker %d failed: %s", worker_id, str(e))
            raise
    
    def __iter__(self) -> Iterator:
        """Return iterator over the dataset"""
        # Initialize if needed
        self.initialize()
        
        # Get iterator from sampler
        self._iterator = iter(self.batch_sampler)
        
        return self
    
    @staticmethod
    def _pin_memory(data):
        """Pin memory for faster GPU transfer.
        
        Args:
            data: Data to pin (tensor, list, tuple, or dict)
            
        Returns:
            Data with memory pinned
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

    def __next__(self):
        """Get next batch of data with retry logic"""
        try:
            # Get indices
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
        except Exception as e:
            logger.error("Error getting next batch: %s", str(e))
            self.cleanup()
            raise
    
    def __len__(self) -> int:
        """Return the number of batches in the dataloader"""
        return len(self.batch_sampler)
    
    def __del__(self):
        """Cleanup resources"""
        self.cleanup()
    
    def cleanup(self):
        """Explicit cleanup method"""
        # Stop prefetch thread
        if self._prefetch_thread is not None:
            self._stop_event.set()
            self._prefetch_thread.join()
            self._prefetch_thread = None
        
        # Clear queues
        if self.prefetch_queue is not None:
            while not self.prefetch_queue.empty():
                try:
                    self.prefetch_queue.get_nowait()
                except Empty:
                    # Queue is empty or get_nowait failed
                    break
        
        if self.batch_queue is not None:
            while not self.batch_queue.empty():
                try:
                    self.batch_queue.get_nowait()
                except Empty:
                    # Queue is empty or get_nowait failed
                    break
        
        # Cleanup worker pool
        if self.worker_pool is not None:
            for worker in self.worker_pool:
                worker.terminate()
            self.worker_pool = None
        
        super().cleanup()

    @staticmethod
    def _default_collate(batch):
        """Default collate function that handles basic batching of tensors and other data types.
        
        Args:
            batch: List of data items from dataset
            
        Returns:
            Batched data in the appropriate format
        """
        if len(batch) == 0:
            return {}
            
        if torch.is_tensor(batch[0]):
            return torch.stack(batch, 0)
            
        elif isinstance(batch[0], (str, bytes)):
            return batch
            
        elif isinstance(batch[0], dict):
            return {
                key: CustomDataLoader._default_collate([d[key] for d in batch])
                for key in batch[0].keys()
            }
            
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [CustomDataLoader._default_collate(samples) for samples in transposed]
            
        return batch

def create_dataloader(
    data_dir,
    batch_size=1,
    num_workers=None,
    no_caching_latents=False,
    all_ar=False,
    cache_dir="latents_cache",
    vae=None,
    tokenizer=None,
    tokenizer_2=None,
    text_encoder=None,
    text_encoder_2=None,
    **kwargs
):
    """Create a dataloader with proper model initialization."""
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
        num_workers=num_workers,
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
