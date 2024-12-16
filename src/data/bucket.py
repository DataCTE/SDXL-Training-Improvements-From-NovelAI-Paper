from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
import math
import numpy as np
import logging
from collections import Counter
from src.data.thread_config import get_optimal_cpu_threads

logger = logging.getLogger(__name__)

@dataclass
class ImageBucket:
    """Bucket for images of similar aspect ratios."""
    width: int
    height: int
    items: List = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError(f"Width and height must be integers, got {type(self.width)} and {type(self.height)}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Width and height must be positive, got {self.width} and {self.height}")
            
        self.aspect_ratio = self.width / self.height
        self.resolution = self.width * self.height
        self.area = self.resolution  # Alias for resolution
        
    def add_item(self, item) -> None:
        """Add item to bucket."""
        self.items.append(item)
        
    def clear(self) -> None:
        """Clear all items from bucket."""
        self.items.clear()
        
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return f"ImageBucket({self.width}x{self.height}, {len(self)} items, ratio={self.aspect_ratio:.3f})"

class AspectRatioBucket:
    """SDXL aspect ratio bucketing system."""
    
    def __init__(
        self,
        max_image_size: Union[Tuple[int, int], int] = (768, 1024),
        min_image_size: Union[Tuple[int, int], int] = (256, 256),
        max_dim: int = 1024,
        bucket_step: int = 64,
        min_bucket_resolution: int = 65536,  # 256x256
        max_bucket_resolution: int = 1048576,  # 1024x1024
        force_square_bucket: bool = True
    ):
        """Initialize bucketing system.
        
        Args:
            max_image_size: Maximum (width, height) for images or single max dimension
            min_image_size: Minimum (width, height) for images or single min dimension
            max_dim: Maximum single dimension
            bucket_step: Step size for bucket dimensions
            min_bucket_resolution: Minimum total pixels in a bucket
            max_bucket_resolution: Maximum total pixels in a bucket
            force_square_bucket: Whether to ensure a square bucket exists
        """
        # Convert single integers to tuples
        if isinstance(max_image_size, int):
            max_image_size = (max_image_size, max_image_size)
        if isinstance(min_image_size, int):
            min_image_size = (min_image_size, min_image_size)
            
        # Validate inputs
        if not all(isinstance(x, int) and x > 0 for x in max_image_size):
            raise ValueError(f"Invalid max_image_size: {max_image_size}")
        if not all(isinstance(x, int) and x > 0 for x in min_image_size):
            raise ValueError(f"Invalid min_image_size: {min_image_size}")
        if max_dim <= 0 or not isinstance(max_dim, int):
            raise ValueError(f"Invalid max_dim: {max_dim}")
        if bucket_step <= 0 or not isinstance(bucket_step, int):
            raise ValueError(f"Invalid bucket_step: {bucket_step}")
            
        self.max_width, self.max_height = max_image_size
        self.min_width, self.min_height = min_image_size
        self.max_dim = max_dim
        self.bucket_step = bucket_step
        self.min_bucket_resolution = min_bucket_resolution
        self.max_bucket_resolution = max_bucket_resolution
        self.force_square_bucket = force_square_bucket
        
        # Initialize buckets
        self.buckets: List[ImageBucket] = []
        self._generate_buckets()
        
        # Cache for aspect ratio lookup
        self._aspect_ratios = np.array([b.aspect_ratio for b in self.buckets])
        self._log_aspects = np.log(self._aspect_ratios)
        
        logger.info(f"Created {len(self.buckets)} buckets")
        
    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Validate and adjust dimensions to constraints."""
        # Round to bucket step
        width = round(width / self.bucket_step) * self.bucket_step
        height = round(height / self.bucket_step) * self.bucket_step
        
        # Clamp to min/max
        width = max(min(width, self.max_width), self.min_width)
        height = max(min(height, self.max_height), self.min_height)
        
        # Ensure max dimension constraint
        if width > self.max_dim:
            height = int(height * (self.max_dim / width))
            width = self.max_dim
        if height > self.max_dim:
            width = int(width * (self.max_dim / height))
            height = self.max_dim
            
        return width, height

    def _generate_buckets(self) -> None:
        """Generate bucket resolutions following SDXL paper."""
        seen_resolutions = set()
        
        def add_bucket(width: int, height: int) -> None:
            """Helper to add bucket if valid."""
            width, height = self._validate_dimensions(width, height)
            resolution = width * height
            
            # Skip if we've seen this resolution or it's outside bounds
            if (width, height) in seen_resolutions:
                return
            if resolution < self.min_bucket_resolution:
                return
            if resolution > self.max_bucket_resolution:
                return
                
            self.buckets.append(ImageBucket(width=width, height=height))
            seen_resolutions.add((width, height))
        
        # Generate width-first buckets
        width = self.min_width
        while width <= self.max_dim:
            height = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / width)
            )
            add_bucket(width, height)
            width += self.bucket_step
            
        # Generate height-first buckets
        height = self.min_height
        while height <= self.max_dim:
            width = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / height)
            )
            add_bucket(width, height)
            height += self.bucket_step
            
        # Add square bucket if requested
        if self.force_square_bucket:
            add_bucket(1024, 1024)
            
        # Sort buckets by aspect ratio for efficient lookup
        self.buckets.sort(key=lambda x: x.aspect_ratio)

    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best fitting bucket for given image dimensions.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Best matching ImageBucket or None if no suitable bucket found
        """
        try:
            # Validate input
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")
                
            image_aspect = width / height
            log_image_aspect = np.log(image_aspect)
            
            # Find closest bucket in log space
            idx = np.argmin(np.abs(self._log_aspects - log_image_aspect))
            return self.buckets[idx]
            
        except Exception as e:
            logger.error(f"Error finding bucket for {width}x{height}: {e}")
            return None
            
    def get_stats(self) -> Dict:
        """Get statistics about current bucket usage."""
        stats = {
            "total_buckets": len(self.buckets),
            "total_images": sum(len(b) for b in self.buckets),
            "bucket_sizes": [(b.width, b.height) for b in self.buckets],
            "images_per_bucket": [len(b) for b in self.buckets],
            "empty_buckets": sum(1 for b in self.buckets if len(b) == 0),
            "aspect_ratios": [b.aspect_ratio for b in self.buckets]
        }
        return stats
        
    def clear_buckets(self) -> None:
        """Clear all items from all buckets."""
        for bucket in self.buckets:
            bucket.clear()

    def _create_buckets(self):
        # Use optimal chunk size for parallel operations
        chunk_size = get_optimal_cpu_threads().chunk_size