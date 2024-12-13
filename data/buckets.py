from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import math
import torch

@dataclass
class ImageBucket:
    width: int
    height: int
    items: List = None
    
    def __post_init__(self):
        self.aspect_ratio = self.width / self.height
        if self.items is None:
            self.items = []

class AspectRatioBucket:
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (768, 1024),
        max_dim: int = 1024,
        bucket_step: int = 64
    ):
        self.max_width, self.max_height = max_image_size
        self.max_dim = max_dim
        self.bucket_step = bucket_step
        self.buckets: List[ImageBucket] = []
        self._generate_buckets()
        
    def _generate_buckets(self):
        """Generate bucket resolutions following section 4.1.2"""
        # Generate width-first buckets
        width = 256
        while width <= self.max_dim:
            # Find largest height that satisfies constraints
            height = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / width)
            )
            self.buckets.append(ImageBucket(
                width=width,
                height=height
            ))
            width += self.bucket_step
            
        # Generate height-first buckets
        height = 256
        while height <= self.max_dim:
            width = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / height)
            )
            # Skip if bucket already exists
            if not any(b.width == width and b.height == height for b in self.buckets):
                self.buckets.append(ImageBucket(
                    width=width,
                    height=height
                ))
            height += self.bucket_step
            
        # Add standard square bucket
        if not any(b.width == 1024 and b.height == 1024 for b in self.buckets):
            self.buckets.append(ImageBucket(
                width=1024,
                height=1024
            ))

    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best fitting bucket for given image dimensions"""
        image_aspect = width / height
        log_aspects = np.log([b.aspect_ratio for b in self.buckets])
        log_image_aspect = np.log(image_aspect)
        
        # Find closest bucket in log-space
        idx = np.argmin(np.abs(log_aspects - log_image_aspect))
        return self.buckets[idx]