from typing import List, Optional, Iterator, Dict, DefaultDict, Set, Tuple, Union
import torch
from torch.utils.data import Sampler
import random
import numpy as np
from collections import defaultdict
import logging
import math
from dataclasses import dataclass, field
import heapq
from src.data.utils import (
    get_optimal_workers,
    create_thread_pool,
    calculate_chunk_size,
    get_memory_usage_gb
)
from src.data.bucket import BucketManager

logger = logging.getLogger(__name__)

@dataclass
class BucketInfo:
    """Information about a bucket and its contents."""
    indices: List[int]
    width: int
    height: int
    aspect_ratio: float
    total_samples: int = 0
    used_samples: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize derived properties."""
        self.total_samples = len(self.indices)
        self.used_samples = 0
        
    def __len__(self) -> int:
        return self.total_samples
    
    def remaining_samples(self) -> int:
        """Get number of remaining samples."""
        return self.total_samples - self.used_samples
    
    def get_next_batch(self, batch_size: int) -> List[int]:
        """Get next batch of indices efficiently."""
        if self.used_samples >= self.total_samples:
            return []
            
        end_idx = min(self.used_samples + batch_size, self.total_samples)
        batch = self.indices[self.used_samples:end_idx]
        self.used_samples = end_idx
        return batch

class AspectBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups images with similar aspect ratios."""
    
    def __init__(
        self,
        dataset,  # NovelAIDataset type hint removed to avoid circular import
        batch_size: int,
        max_image_size: Union[Tuple[int, int], int] = (768, 1024),
        min_image_size: Union[Tuple[int, int], int] = (256, 256),
        max_dim: Optional[int] = None,
        bucket_step: int = 64,
        min_bucket_resolution: int = 65536,  # 256x256
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_tolerance: float = 0.2,
        max_aspect_ratio: float = 4.0,
        min_bucket_length: Optional[int] = None,
        max_consecutive_batch_samples: int = 2,
        num_workers: Optional[int] = None
    ):
        """Initialize the sampler with optimized settings."""
        super().__init__(dataset)
        
        # Convert single integers to tuples
        if isinstance(max_image_size, int):
            max_image_size = (max_image_size, max_image_size)
        if isinstance(min_image_size, int):
            min_image_size = (min_image_size, min_image_size)
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_tolerance = bucket_tolerance
        self.max_aspect_ratio = max_aspect_ratio
        self.min_bucket_length = min_bucket_length or batch_size
        self.max_consecutive_batch_samples = max_consecutive_batch_samples
        
        # Initialize bucket manager
        self.bucket_manager = BucketManager(
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            bucket_step=bucket_step,
            min_bucket_resolution=min_bucket_resolution,
            max_aspect_ratio=max_aspect_ratio,
            bucket_tolerance=bucket_tolerance
        )
        
        # Initialize thread pool for parallel processing
        self.num_workers = num_workers or get_optimal_workers(memory_per_worker_gb=0.5)
        self.executor = create_thread_pool(self.num_workers)
        
        # Initialize state
        self.buckets: Dict[float, BucketInfo] = {}
        self.total_samples = 0
        self.epoch = 0
        self.rng = np.random.RandomState()
        
        # Pre-allocate reusable buffers
        self.bucket_heap = []
        self.used_buckets: Set[float] = set()
        
        # Create initial buckets and assign samples
        self._create_buckets()
        self._assign_samples_to_buckets()
        
        # Calculate number of batches
        self.total_batches = self._calculate_total_batches()
        
        logger.info(
            f"Initialized sampler:\n"
            f"- Buckets: {len(self.buckets)}\n"
            f"- Total batches: {self.total_batches}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Memory usage: {get_memory_usage_gb():.1f}GB"
        )

    def _create_buckets(self) -> None:
        """Create buckets from bucket manager."""
        # Convert bucket manager buckets to sampler buckets
        for aspect, bucket in self.bucket_manager.buckets.items():
            self.buckets[aspect] = BucketInfo(
                indices=[],
                width=bucket.width,
                height=bucket.height,
                aspect_ratio=aspect
            )
        
        logger.info(f"Created {len(self.buckets)} buckets")

    def _assign_samples_to_buckets(self) -> None:
        """Assign dataset samples to appropriate buckets efficiently."""
        # Pre-calculate log aspect ratios for faster lookup
        bucket_aspects = np.array(list(self.buckets.keys()))
        log_bucket_aspects = np.log(bucket_aspects)
        
        # Process samples in parallel chunks
        chunk_size = calculate_chunk_size(
            total_items=len(self.dataset),
            optimal_workers=self.num_workers
        )
        
        futures = []
        for start_idx in range(0, len(self.dataset), chunk_size):
            end_idx = min(start_idx + chunk_size, len(self.dataset))
            future = self.executor.submit(
                self._process_sample_chunk,
                start_idx, end_idx,
                bucket_aspects,
                log_bucket_aspects
            )
            futures.append(future)
        
        # Collect results and assign to buckets
        for future in futures:
            chunk_assignments = future.result()
            for aspect, indices in chunk_assignments.items():
                if aspect in self.buckets:
                    self.buckets[aspect].indices.extend(indices)
                    
        # Remove empty buckets
        self.buckets = {
            aspect: info for aspect, info in self.buckets.items()
            if len(info.indices) >= self.min_bucket_length
        }
        
        self.total_samples = sum(len(bucket.indices) for bucket in self.buckets.values())
        logger.info(f"Assigned {self.total_samples} samples to {len(self.buckets)} buckets")

    def _process_sample_chunk(
        self,
        start_idx: int,
        end_idx: int,
        bucket_aspects: np.ndarray,
        log_bucket_aspects: np.ndarray
    ) -> Dict[float, List[int]]:
        """Process a chunk of samples for bucket assignment."""
        chunk_assignments: DefaultDict[float, List[int]] = defaultdict(list)
        
        for idx in range(start_idx, end_idx):
            try:
                # Get image dimensions from dataset
                width = self.dataset.items[idx][1].width
                height = self.dataset.items[idx][1].height
                
                if width <= 0 or height <= 0:
                    continue
                    
                # Find closest bucket in log space
                image_aspect = width / height
                if image_aspect > self.max_aspect_ratio or image_aspect < (1/self.max_aspect_ratio):
                    continue
                    
                log_image_aspect = np.log(image_aspect)
                bucket_idx = np.argmin(np.abs(log_bucket_aspects - log_image_aspect))
                bucket_aspect = bucket_aspects[bucket_idx]
                
                chunk_assignments[bucket_aspect].append(idx)
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
                
        return chunk_assignments

    def _calculate_total_batches(self) -> int:
        """Calculate total number of batches efficiently."""
        if self.drop_last:
            return sum(len(bucket) // self.batch_size for bucket in self.buckets.values())
        else:
            return sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.buckets.values())

    def _shuffle_buckets(self, epoch: int) -> None:
        """Shuffle bucket contents with optimized memory usage."""
        if not self.shuffle:
            return
            
        # Set seed for reproducibility
        self.rng.seed(epoch)
        
        # Shuffle within each bucket using numpy for efficiency
        for bucket in self.buckets.values():
            # Convert to numpy array for faster shuffling
            indices = np.array(bucket.indices, dtype=np.int32)
            self.rng.shuffle(indices)
            bucket.indices = indices.tolist()
            bucket.used_samples = 0

    def _create_batches(self) -> List[List[int]]:
        """Create batches with optimized scheduling."""
        all_batches = []
        
        # Initialize priority queue with bucket priorities
        self.bucket_heap = [
            (-len(bucket), aspect)  # Negative length for max-heap
            for aspect, bucket in self.buckets.items()
        ]
        heapq.heapify(self.bucket_heap)
        
        self.used_buckets.clear()
        consecutive_samples = defaultdict(int)
        
        while self.bucket_heap:
            # Get bucket with most remaining samples
            _, aspect = heapq.heappop(self.bucket_heap)
            bucket = self.buckets[aspect]
            
            # Skip if bucket is depleted
            if bucket.remaining_samples() == 0:
                continue
                
            # Check consecutive sample limit
            if consecutive_samples[aspect] >= self.max_consecutive_batch_samples:
                # Put back in heap with adjusted priority
                heapq.heappush(self.bucket_heap, (-bucket.remaining_samples(), aspect))
                continue
            
            # Get next batch
            batch = bucket.get_next_batch(self.batch_size)
            if batch:
                all_batches.append(batch)
                consecutive_samples[aspect] += 1
                
                # Reset other buckets' consecutive counts
                for other_aspect in consecutive_samples:
                    if other_aspect != aspect:
                        consecutive_samples[other_aspect] = 0
                
                # Put back in heap if not depleted
                if bucket.remaining_samples() > 0:
                    heapq.heappush(self.bucket_heap, (-bucket.remaining_samples(), aspect))
            
            # Handle final partial batch
            elif not self.drop_last and bucket.remaining_samples() > 0:
                final_batch = bucket.get_next_batch(bucket.remaining_samples())
                if final_batch:
                    all_batches.append(final_batch)
        
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        """Create and yield batches for current epoch efficiently."""
        # Shuffle buckets if needed
        self._shuffle_buckets(self.epoch)
        
        # Create all batches
        all_batches = self._create_batches()
        
        # Shuffle batch order if needed
        if self.shuffle:
            self.rng.shuffle(all_batches)
        
        # Increment epoch
        self.epoch += 1
        
        # Yield batches
        yield from all_batches

    def __len__(self) -> int:
        return self.total_batches
    
    def get_stats(self) -> Dict:
        """Get detailed statistics about current bucketing."""
        stats = {
            "total_buckets": len(self.buckets),
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "aspect_ratios": sorted(self.buckets.keys()),
            "samples_per_bucket": {
                aspect: len(bucket) for aspect, bucket in self.buckets.items()
            },
            "avg_batch_size": self.total_samples / self.total_batches,
            "memory_usage_gb": get_memory_usage_gb(),
            "max_bucket_size": max(len(b) for b in self.buckets.values()),
            "min_bucket_size": min(len(b) for b in self.buckets.values()),
            "avg_bucket_size": self.total_samples / len(self.buckets) if self.buckets else 0
        }
        return stats