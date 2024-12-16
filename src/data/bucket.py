from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ImageBucket:
    """Bucket for images of similar aspect ratios."""
    width: int
    height: int
    items: List[Any] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError(f"Width and height must be integers, got {type(self.width)} and {type(self.height)}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Width and height must be positive, got {self.width} and {self.height}")
            
        self.aspect_ratio = self.width / self.height
        self.resolution = self.width * self.height
        self.used_items = 0
        
    def add_item(self, item: Any) -> None:
        """Add item to bucket."""
        self.items.append(item)
        
    def clear(self) -> None:
        """Clear all items from bucket."""
        self.items.clear()
        self.used_items = 0
        
    def get_next_batch(self, batch_size: int) -> List[Any]:
        """Get next batch of items up to batch_size."""
        if self.used_items >= len(self.items):
            return []
        end_idx = min(self.used_items + batch_size, len(self.items))
        batch = self.items[self.used_items:end_idx]
        self.used_items = end_idx
        return batch
        
    def remaining_items(self) -> int:
        """Get number of remaining items."""
        return len(self.items) - self.used_items
        
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return f"ImageBucket({self.width}x{self.height}, {len(self)} items, ratio={self.aspect_ratio:.3f})"

class BucketManager:
    """Manager for creating and assigning image buckets."""
    
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (8192, 8192),
        min_image_size: Tuple[int, int] = (256, 256),
        bucket_step: int = 64,
        min_bucket_resolution: int = 65536,  # 256x256
        max_aspect_ratio: float = 4.0,
        bucket_tolerance: float = 0.2
    ):
        self.max_width, self.max_height = max_image_size
        self.min_width, self.min_height = min_image_size
        self.bucket_step = bucket_step
        self.min_bucket_resolution = min_bucket_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.bucket_tolerance = bucket_tolerance
        
        # Initialize buckets
        self.buckets: Dict[float, ImageBucket] = {}
        self._create_buckets()
        
        logger.info(
            f"Initialized BucketManager:\n"
            f"- Size range: {min_image_size} to {max_image_size}\n"
            f"- Bucket step: {bucket_step}\n"
            f"- Total buckets: {len(self.buckets)}"
        )
    
    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Validate and adjust dimensions to constraints."""
        # Round to bucket step
        width = round(width / self.bucket_step) * self.bucket_step
        height = round(height / self.bucket_step) * self.bucket_step
        
        # Enforce minimum dimensions
        width = max(width, self.min_width)
        height = max(height, self.min_height)
        
        # Enforce maximum dimensions
        width = min(width, self.max_width)
        height = min(height, self.max_height)
            
        return width, height
    
    def _create_buckets(self) -> None:
        """Generate bucket resolutions efficiently."""
        seen_resolutions = set()
        
        def add_bucket(width: int, height: int) -> None:
            """Helper to add bucket if valid."""
            width, height = self._validate_dimensions(width, height)
            resolution = width * height
            
            # Skip if we've seen this resolution or it's below minimum
            if (width, height) in seen_resolutions:
                return
            if resolution < self.min_bucket_resolution:
                return
                
            aspect = width / height
            if aspect > self.max_aspect_ratio or aspect < (1/self.max_aspect_ratio):
                return
                
            self.buckets[aspect] = ImageBucket(width=width, height=height)
            seen_resolutions.add((width, height))
            logger.debug(f"Added bucket: {width}x{height} (ratio: {aspect:.2f})")
        
        # Generate buckets more efficiently for large ranges
        current_width = self.min_width
        while current_width <= self.max_width:
            current_height = self.min_height
            while current_height <= self.max_height:
                # Skip if resolution would be too large
                if current_width * current_height > (8192 * 8192):
                    break
                    
                add_bucket(current_width, current_height)
                
                # Use larger steps for higher resolutions to avoid too many buckets
                if current_height >= 2048:
                    current_height += self.bucket_step * 4
                elif current_height >= 1024:
                    current_height += self.bucket_step * 2
                else:
                    current_height += self.bucket_step
                    
            if current_width >= 2048:
                current_width += self.bucket_step * 4
            elif current_width >= 1024:
                current_width += self.bucket_step * 2
            else:
                current_width += self.bucket_step
            
        logger.info(f"Created {len(self.buckets)} buckets")
    
    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best bucket for given dimensions."""
        if width <= 0 or height <= 0:
            return None
            
        # Validate and adjust dimensions
        width, height = self._validate_dimensions(width, height)
            
        # Calculate aspect ratio
        aspect = width / height
        if aspect > self.max_aspect_ratio or aspect < (1/self.max_aspect_ratio):
            return None
            
        # Find closest bucket in log space
        log_aspect = np.log(aspect)
        bucket_aspects = np.array(list(self.buckets.keys()))
        log_bucket_aspects = np.log(bucket_aspects)
        
        # Find closest bucket within tolerance
        diffs = np.abs(log_bucket_aspects - log_aspect)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= self.bucket_tolerance:
            return self.buckets[bucket_aspects[min_idx]]
            
        return None
    
    def get_stats(self) -> Dict:
        """Get bucket statistics."""
        return {
            "total_buckets": len(self.buckets),
            "aspect_ratios": sorted(self.buckets.keys()),
            "items_per_bucket": {
                aspect: len(bucket) for aspect, bucket in self.buckets.items()
            },
            "total_items": sum(len(b) for b in self.buckets.values()),
            "max_bucket_size": max((len(b) for b in self.buckets.values()), default=0),
            "min_bucket_size": min((len(b) for b in self.buckets.values()), default=0)
        }
    
    def get_bucket_by_key(self, key: str) -> Optional[ImageBucket]:
        """Get bucket by resolution key (e.g. '512x512')."""
        width, height = map(int, key.split('x'))
        aspect = width / height
        return self.buckets.get(aspect)
    
    def get_bucket_info(self) -> Dict[str, ImageBucket]:
        """Get mapping of resolution keys to buckets."""
        return {
            f"{bucket.width}x{bucket.height}": bucket 
            for bucket in self.buckets.values()
        }
    
    def validate_bucket_key(self, key: str) -> bool:
        """Validate if a bucket key matches expected format and constraints."""
        try:
            width, height = map(int, key.split('x'))
            if width <= 0 or height <= 0:
                return False
            aspect = width / height
            return (aspect <= self.max_aspect_ratio and 
                    aspect >= 1/self.max_aspect_ratio and
                    width * height >= self.min_bucket_resolution)
        except (ValueError, TypeError):
            return False