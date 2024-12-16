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
from src.data.utils import get_memory_usage_gb
from src.data.bucket import BucketManager, ImageBucket

logger = logging.getLogger(__name__)

@dataclass
class BucketInfo:
    """Information about a bucket and its contents."""
    indices: List[int] = field(default_factory=list)
    bucket: Optional[ImageBucket] = None
    used_samples: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize derived properties."""
        self.used_samples = 0
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def remaining_samples(self) -> int:
        """Get number of remaining samples."""
        return len(self.indices) - self.used_samples
    
    def get_next_batch(self, batch_size: int) -> List[int]:
        """Get next batch of indices efficiently."""
        if self.used_samples >= len(self.indices):
            return []
            
        end_idx = min(self.used_samples + batch_size, len(self.indices))
        batch = self.indices[self.used_samples:end_idx]
        self.used_samples = end_idx
        return batch

class AspectBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups images with similar aspect ratios."""
    
    def __init__(
        self,
        dataset,  # NovelAIDataset
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        max_consecutive_batch_samples: int = 2,
        min_bucket_length: int = 1
    ):
        """Initialize using dataset's bucket information."""
        super().__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_consecutive_batch_samples = max_consecutive_batch_samples
        self.min_bucket_length = min_bucket_length
        
        # Use dataset's bucket manager
        self.bucket_manager = dataset.bucket_manager
        
        # Initialize state
        self.buckets: Dict[str, BucketInfo] = {}
        self.total_samples = 0
        self.epoch = 0
        self.rng = np.random.RandomState()
        
        # Pre-allocate reusable buffers
        self.bucket_heap = []
        self.used_buckets: Set[str] = set()
        
        # Create initial buckets and assign samples
        self._create_buckets()
        self._assign_samples_to_buckets()
        
        # Calculate number of batches
        self.total_batches = self._calculate_total_batches()
        
        logger.info(
            f"Initialized sampler:\n"
            f"- Buckets: {len(self.buckets)}\n"
            f"- Total batches: {self.total_batches}\n"
            f"- Total samples: {self.total_samples}\n"
            f"- Memory usage: {get_memory_usage_gb():.1f}GB"
        )

    def _get_bucket_key(self, bucket: ImageBucket) -> str:
        """Get unique key for bucket based on resolution."""
        key = f"{bucket.width}x{bucket.height}"
        if not self.bucket_manager.validate_bucket_key(key):
            raise ValueError(f"Invalid bucket dimensions: {key}")
        return key

    def _create_buckets(self) -> None:
        """Create buckets using dataset's bucket information."""
        bucket_mapping = self.bucket_manager.get_bucket_info()
        
        # Create BucketInfo objects for each unique bucket
        for item in self.dataset.items:
            bucket = item.get('bucket')
            if bucket is not None and isinstance(bucket, ImageBucket):
                bucket_key = self._get_bucket_key(bucket)
                if bucket_key not in self.buckets and bucket_key in bucket_mapping:
                    self.buckets[bucket_key] = BucketInfo(bucket=bucket_mapping[bucket_key])
        
        if not self.buckets:
            raise ValueError("No valid buckets could be created from dataset")
            
        logger.info(f"Created {len(self.buckets)} buckets from dataset")

    def _assign_samples_to_buckets(self) -> None:
        """Assign dataset samples to appropriate buckets using existing bucket information."""
        try:
            total_assigned = 0
            
            for idx, item in enumerate(self.dataset.items):
                try:
                    bucket = item.get('bucket')
                    if bucket is None or not isinstance(bucket, ImageBucket):
                        logger.error(f"Invalid bucket information for sample {idx}")
                        continue
                        
                    bucket_key = self._get_bucket_key(bucket)
                    if bucket_key in self.buckets:
                        self.buckets[bucket_key].indices.append(idx)
                        total_assigned += 1
                    else:
                        logger.debug(f"No matching bucket for sample {idx} (resolution {bucket.width}x{bucket.height})")
                        
                except Exception as e:
                    logger.error(f"Error assigning sample {idx}: {str(e)}")
                    continue
                    
            # Remove empty buckets
            self.buckets = {
                key: info for key, info in self.buckets.items()
                if len(info.indices) >= self.min_bucket_length
            }
            
            if total_assigned == 0:
                raise ValueError("No samples were assigned to any buckets")
                
            self.total_samples = sum(len(bucket.indices) for bucket in self.buckets.values())
            logger.info(f"Assigned {self.total_samples} samples to {len(self.buckets)} buckets")
            
        except Exception as e:
            logger.error(f"Failed to assign samples to buckets: {str(e)}")
            raise

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
            (-len(bucket), key)  # Negative length for max-heap
            for key, bucket in self.buckets.items()
        ]
        heapq.heapify(self.bucket_heap)
        
        self.used_buckets.clear()
        consecutive_samples = defaultdict(int)
        
        while self.bucket_heap:
            # Get bucket with most remaining samples
            _, bucket_key = heapq.heappop(self.bucket_heap)
            bucket = self.buckets[bucket_key]
            
            # Skip if bucket is depleted
            if bucket.remaining_samples() == 0:
                continue
                
            # Check consecutive sample limit
            if consecutive_samples[bucket_key] >= self.max_consecutive_batch_samples:
                # Put back in heap with adjusted priority
                heapq.heappush(self.bucket_heap, (-bucket.remaining_samples(), bucket_key))
                continue
            
            # Get next batch
            batch = bucket.get_next_batch(self.batch_size)
            if batch:
                all_batches.append(batch)
                consecutive_samples[bucket_key] += 1
                
                # Reset other buckets' consecutive counts
                for other_key in consecutive_samples:
                    if other_key != bucket_key:
                        consecutive_samples[other_key] = 0
                
                # Put back in heap if not depleted
                if bucket.remaining_samples() > 0:
                    heapq.heappush(self.bucket_heap, (-bucket.remaining_samples(), bucket_key))
            
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