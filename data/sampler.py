import random
import torch
from torch.utils.data import Sampler
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data.distributed import DistributedSampler
from utils.error_handling import error_handler
from data.dataset import NovelAIDataset

class AspectBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: NovelAIDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by latent dimensions
        self.groups = {}
        for idx in range(len(dataset)):
            img_cache_path = dataset.cache_dir / f"{dataset.image_files[idx].stem}.pt"
            if img_cache_path.exists():
                # Load cached latents to get dimensions
                try:
                    latents = torch.load(img_cache_path, map_location='cpu')
                    key = (latents.shape[2], latents.shape[3])  # Group by latent height, width
                    if key not in self.groups:
                        self.groups[key] = []
                    self.groups[key].append(idx)
                except Exception as e:
                    print(f"Error loading latents from {img_cache_path}: {e}")
                    continue
            else:
                # If latents not cached, get dimensions from original image
                try:
                    with Image.open(dataset.image_files[idx]) as img:
                        width, height = img.size
                        # Convert to latent dimensions (divide by 8)
                        latent_height = height // 8
                        latent_width = width // 8
                        key = (latent_height, latent_width)
                        if key not in self.groups:
                            self.groups[key] = []
                        self.groups[key].append(idx)
                except Exception as e:
                    print(f"Error processing {dataset.image_files[idx]}: {e}")
                    continue
        
        # Create batches from groups
        self.batches = []
        for indices in self.groups.values():
            # Create batches of exactly matching dimensions
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only use full batches
                    self.batches.append(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)
