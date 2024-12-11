import random
import torch
from torch.utils.data import Sampler
from PIL import Image
from typing import List, Tuple

class AspectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by exact bucket dimensions
        self.groups = {}
        for idx, (_, bucket, img_cache_path, _) in enumerate(dataset.items):
            key = (bucket.width, bucket.height)
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)
        
        # Create batches of exactly matching dimensions
        self.batches = []
        for indices in self.groups.values():
            # Sort indices by actual image dimensions
            sorted_indices = []
            for idx in indices:
                img_path, _, _, _ = dataset.items[idx]
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
                        self.batches.append(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches) 