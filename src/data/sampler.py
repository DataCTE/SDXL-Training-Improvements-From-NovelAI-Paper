from typing import List, Optional, Iterator
import torch
from torch.utils.data import Sampler
from .dataset import NovelAIDataset
import random
import numpy as np
from collections import defaultdict

class AspectBatchSampler(Sampler[List[int]]):
    """
    Custom batch sampler that groups images with similar aspect ratios together.
    This reduces padding waste and improves memory efficiency.
    """
    
    def __init__(
        self,
        dataset: NovelAIDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_tolerance: float = 0.2
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_tolerance = bucket_tolerance
        
        # Group indices by similar aspect ratios
        self.buckets = self._create_buckets()
        
        # Calculate number of batches
        total_batches = 0
        for bucket in self.buckets.values():
            if self.drop_last:
                total_batches += len(bucket) // self.batch_size
            else:
                total_batches += (len(bucket) + self.batch_size - 1) // self.batch_size
        self.total_batches = total_batches

    def _create_buckets(self) -> dict:
        """Group dataset indices by similar aspect ratios"""
        buckets = defaultdict(list)
        
        # Calculate aspect ratios for all items
        for idx, (_, bucket, _, _) in enumerate(self.dataset.items):
            # Get aspect ratio and round to reduce bucket count
            aspect = round(bucket.width / bucket.height, 1)
            buckets[aspect].append(idx)
            
        return buckets

    def _shuffle_buckets(self):
        """Shuffle indices within each bucket"""
        for bucket in self.buckets.values():
            random.shuffle(bucket)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle if needed
        if self.shuffle:
            self._shuffle_buckets()
        
        # Create batches from each bucket
        all_batches = []
        for bucket_indices in self.buckets.values():
            # Skip empty buckets
            if not bucket_indices:
                continue
                
            # Create batches for this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        
        # Shuffle batches if needed
        if self.shuffle:
            random.shuffle(all_batches)
        
        # Yield batches
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return self.total_batches 