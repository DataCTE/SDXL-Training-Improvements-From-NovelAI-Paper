from typing import List, Optional, Iterator, Dict, DefaultDict
import torch
from torch.utils.data import Sampler
import random
import numpy as np
from collections import defaultdict
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BucketInfo:
    """Information about a bucket and its contents."""
    indices: List[int]
    width: int
    height: int
    aspect_ratio: float
    total_samples: int = 0
    
    def __post_init__(self):
        self.total_samples = len(self.indices)
        
    def __len__(self) -> int:
        return self.total_samples

class AspectBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups images with similar aspect ratios."""
    
    def __init__(
        self,
        dataset,  # NovelAIDataset type hint removed to avoid circular import
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_tolerance: float = 0.2,
        max_aspect_ratio: float = 4.0,
        min_bucket_length: Optional[int] = None,
        max_consecutive_batch_samples: int = 2
    ):
        """Initialize the sampler.
        
        Args:
            dataset: Dataset containing image information
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle batches and samples
            drop_last: Whether to drop incomplete batches
            bucket_tolerance: Tolerance for aspect ratio bucketing
            max_aspect_ratio: Maximum allowed aspect ratio (filters outliers)
            min_bucket_length: Minimum samples required for a bucket
            max_consecutive_batch_samples: Maximum consecutive samples from same bucket
        """
        super().__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_tolerance = bucket_tolerance
        self.max_aspect_ratio = max_aspect_ratio
        self.min_bucket_length = min_bucket_length or batch_size
        self.max_consecutive_batch_samples = max_consecutive_batch_samples
        
        # Initialize state
        self.buckets: Dict[float, BucketInfo] = {}
        self.total_samples = 0
        self.epoch = 0
        
        # Create initial buckets
        self._create_buckets()
        
        # Calculate number of batches
        self.total_batches = self._calculate_total_batches()
        
        logger.info(f"Created sampler with {len(self.buckets)} buckets and {self.total_batches} batches")

    def _create_buckets(self) -> None:
        """Create aspect ratio buckets efficiently."""
        temp_buckets: DefaultDict[float, List[int]] = defaultdict(list)
        skipped = 0
        
        # First pass: group by rounded aspect ratios
        for idx, (_, bucket, _, _) in enumerate(self.dataset.items):
            aspect = bucket.width / bucket.height
            
            # Skip extreme aspect ratios
            if aspect > self.max_aspect_ratio or aspect < (1/self.max_aspect_ratio):
                skipped += 1
                continue
                
            # Round aspect ratio based on tolerance
            rounded_aspect = round(aspect / self.bucket_tolerance) * self.bucket_tolerance
            temp_buckets[rounded_aspect].append(idx)
        
        # Second pass: create final buckets with sufficient samples
        for aspect, indices in temp_buckets.items():
            if len(indices) >= self.min_bucket_length:
                self.buckets[aspect] = BucketInfo(
                    indices=indices,
                    width=self.dataset.items[indices[0]][1].width,
                    height=self.dataset.items[indices[0]][1].height,
                    aspect_ratio=aspect
                )
                self.total_samples += len(indices)
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} samples with extreme aspect ratios")

    def _calculate_total_batches(self) -> int:
        """Calculate total number of batches."""
        total = 0
        for bucket in self.buckets.values():
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += math.ceil(len(bucket) / self.batch_size)
        return total

    def _shuffle_buckets(self, epoch: int) -> None:
        """Shuffle bucket contents with epoch-based seed."""
        if not self.shuffle:
            return
            
        # Set seed for reproducibility
        rng = random.Random(epoch)
        
        # Shuffle within each bucket
        for bucket in self.buckets.values():
            rng.shuffle(bucket.indices)

    def _create_batches(self) -> List[List[int]]:
        """Create batches with optimization constraints."""
        all_batches = []
        bucket_counts = {aspect: 0 for aspect in self.buckets.keys()}
        available_buckets = list(self.buckets.keys())
        
        while available_buckets:
            # Select bucket with fewest used samples
            aspect = min(available_buckets, key=lambda x: bucket_counts[x])
            bucket = self.buckets[aspect]
            
            # Get next batch from this bucket
            start_idx = bucket_counts[aspect]
            end_idx = start_idx + self.batch_size
            
            if end_idx <= len(bucket):
                batch = bucket.indices[start_idx:end_idx]
                all_batches.append(batch)
                bucket_counts[aspect] = end_idx
            
            # Remove bucket if depleted
            if bucket_counts[aspect] + self.batch_size > len(bucket):
                if not self.drop_last and bucket_counts[aspect] < len(bucket):
                    # Add final partial batch if not dropping
                    final_batch = bucket.indices[bucket_counts[aspect]:]
                    all_batches.append(final_batch)
                available_buckets.remove(aspect)
        
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        """Create and yield batches for current epoch."""
        # Shuffle buckets if needed
        self._shuffle_buckets(self.epoch)
        
        # Create all batches
        all_batches = self._create_batches()
        
        # Shuffle batch order if needed
        if self.shuffle:
            random.Random(self.epoch).shuffle(all_batches)
        
        # Increment epoch
        self.epoch += 1
        
        # Yield batches
        yield from all_batches

    def __len__(self) -> int:
        return self.total_batches
    
    def get_stats(self) -> Dict:
        """Get statistics about current bucketing."""
        stats = {
            "total_buckets": len(self.buckets),
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "aspect_ratios": sorted(self.buckets.keys()),
            "samples_per_bucket": {
                aspect: len(bucket) for aspect, bucket in self.buckets.items()
            },
            "avg_batch_size": self.total_samples / self.total_batches,
        }
        return stats