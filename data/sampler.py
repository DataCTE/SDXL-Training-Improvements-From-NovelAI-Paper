import random
import torch
from torch.utils.data import Sampler
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data.distributed import DistributedSampler

class AspectBatchSampler(Sampler):
    def __init__(
        self, 
        dataset,
        batch_size: int,
        shuffle: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Distributed training setup
        self.world_size = world_size or 1
        self.rank = rank or 0
        self.distributed = world_size is not None
        
        # Group indices by exact bucket dimensions
        self.groups = self._group_indices()
        self.batches = self._create_batches()
        
    def _group_indices(self):
        # Group indices by exact bucket dimensions
        groups = {}
        for idx, (_, bucket, img_cache_path, _) in enumerate(self.dataset.items):
            key = (bucket.width, bucket.height)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return groups
    
    def _create_batches(self):
        batches = []
        # Create batches of exactly matching dimensions
        for indices in self.groups.values():
            # Sort indices by actual image dimensions
            sorted_indices = []
            for idx in indices:
                img_path, _, _, _ = self.dataset.items[idx]
                with Image.open(img_path) as img:
                    width, height = img.size
                    sorted_indices.append((idx, (width, height)))
            
            # Group by exact dimensions
            exact_groups = {}
            for idx, dims in sorted_indices:
                if dims not in exact_groups:
                    exact_groups[dims] = []
                exact_groups[dims].append(idx)
            
            # Create batches from each exact group
            for exact_indices in exact_groups.values():
                for i in range(0, len(exact_indices), self.batch_size):
                    batch = exact_indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size:  # Only use full batches
                        batches.append(batch)
        
        if self.distributed:
            # Ensure number of batches is divisible by world size
            num_batches = len(batches)
            padding = (self.world_size - (num_batches % self.world_size)) % self.world_size
            batches.extend(batches[:padding])
            
        return batches
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches) 