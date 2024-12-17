from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple, Iterator
import logging
import numpy as np
import heapq
from collections import defaultdict
import gc
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

@dataclass
class ImageBucket:
    """Bucket for images of similar aspect ratios."""
    width: int
    height: int
    indices: List[int] = field(default_factory=list)
    used_samples: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError(f"Width and height must be integers, got {type(self.width)} and {type(self.height)}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Width and height must be positive, got {self.width} and {self.height}")
            
        self.aspect_ratio = self.width / self.height
        self.resolution = self.width * self.height
        self.used_samples = 0
        
    def add_index(self, idx: int) -> None:
        """Add sample index to bucket."""
        self.indices.append(idx)
        
    def clear(self) -> None:
        """Clear all indices from bucket."""
        self.indices.clear()
        self.used_samples = 0
        gc.collect()  # Help clean up any references
        
    def get_next_batch(self, batch_size: int) -> List[int]:
        """Get next batch of indices up to batch_size."""
        if self.used_samples >= len(self.indices):
            return []
        end_idx = min(self.used_samples + batch_size, len(self.indices))
        batch = self.indices[self.used_samples:end_idx]
        self.used_samples = end_idx
        return batch
        
    def remaining_samples(self) -> int:
        """Get number of remaining samples."""
        return len(self.indices) - self.used_samples
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __repr__(self) -> str:
        return f"ImageBucket({self.width}x{self.height}, {len(self)} samples, ratio={self.aspect_ratio:.3f})"

class BucketManager:
    """Manager for creating and assigning image buckets."""
    
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (2048, 2048),
        min_image_size: Tuple[int, int] = (256, 256),
        bucket_step: int = 64,
        min_bucket_resolution: int = 2048 * 2048,  # 2048x2048=
        max_aspect_ratio: float = 4.0,
        bucket_tolerance: float = 0.2
    ):
        self.max_width, self.max_height = max_image_size
        self.min_width, self.min_height = min_image_size
        self.bucket_step = bucket_step
        self.min_bucket_resolution = min_bucket_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.bucket_tolerance = bucket_tolerance
        
        # Initialize buckets with WeakValueDictionary to help prevent memory leaks
        self.buckets: Dict[str, ImageBucket] = WeakValueDictionary()
        self._create_buckets()
        
        # State tracking
        self.total_samples = 0
        self.epoch = 0
        self.rng = np.random.RandomState()
        
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
            bucket_key = f"{width}x{height}"
            
            # Skip if we've seen this resolution or it's below minimum
            if bucket_key in seen_resolutions:
                return
            if resolution < self.min_bucket_resolution:
                return
                
            aspect = width / height
            if aspect > self.max_aspect_ratio or aspect < (1/self.max_aspect_ratio):
                return
                
            self.buckets[bucket_key] = ImageBucket(width=width, height=height)
            seen_resolutions.add(bucket_key)
            logger.debug(f"Added bucket: {width}x{height} (ratio: {aspect:.2f})")
        
        # Generate buckets more efficiently for large ranges
        current_width = self.min_width
        while current_width <= self.max_width:
            current_height = self.min_height
            while current_height <= self.max_height:
                # Skip if resolution would be too large
                if current_width * current_height > (2048 * 2048):
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
        
        # Clear temporary set
        seen_resolutions.clear()
        gc.collect()
            
        logger.info(f"Created {len(self.buckets)} buckets")
    
    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best bucket for given dimensions."""
        if width <= 0 or height <= 0:
            return None
            
        # Validate and adjust dimensions
        width, height = self._validate_dimensions(width, height)
        bucket_key = f"{width}x{height}"
            
        # Return exact match if exists
        if bucket_key in self.buckets:
            return self.buckets[bucket_key]
            
        # Calculate aspect ratio
        aspect = width / height
        if aspect > self.max_aspect_ratio or aspect < (1/self.max_aspect_ratio):
            return None
            
        # Find closest bucket by resolution and aspect ratio
        min_diff = float('inf')
        best_bucket = None
        target_res = width * height
        
        for key, bucket in self.buckets.items():
            # Check aspect ratio first
            ratio_diff = abs(np.log(aspect) - np.log(bucket.aspect_ratio))
            if ratio_diff > self.bucket_tolerance:
                continue
                
            # Then check resolution
            res_diff = abs(target_res - bucket.resolution)
            if res_diff < min_diff:
                min_diff = res_diff
                best_bucket = bucket
                
        return best_bucket
    
    def assign_to_buckets(self, items: List[Dict], shuffle: bool = True) -> None:
        """Assign items to buckets and optionally shuffle."""
        # Reset state
        self.total_samples = 0
        for bucket in self.buckets.values():
            bucket.clear()
        
        # Assign items to buckets
        for idx, item in enumerate(items):
            width = item.get('width')
            height = item.get('height')
            if not width or not height:
                continue
                
            bucket = self.find_bucket(width, height)
            if bucket is not None:
                bucket.add_index(idx)
                self.total_samples += 1
        
        # Shuffle if requested
        if shuffle:
            self.shuffle_buckets()
        
        # Clear memory after assignment
        gc.collect()
    
    def shuffle_buckets(self, epoch: Optional[int] = None) -> None:
        """Shuffle indices within each bucket."""
        if epoch is not None:
            self.rng.seed(epoch)
            
        for bucket in self.buckets.values():
            if len(bucket.indices) > 0:
                indices = np.array(bucket.indices)
                self.rng.shuffle(indices)
                bucket.indices = indices.tolist()
                bucket.used_samples = 0
                
                # Clear numpy array
                del indices
        
        # Clear memory after shuffling
        gc.collect()
    
    def get_bucket_by_key(self, key: str) -> Optional[ImageBucket]:
        """Get bucket by resolution key (e.g. '512x512')."""
        return self.buckets.get(key)
    
    def get_bucket_info(self) -> Dict[str, ImageBucket]:
        """Get mapping of resolution keys to buckets."""
        return dict(self.buckets)  # Create new dict to avoid WeakValueDictionary issues
    
    def get_stats(self) -> Dict:
        """Get comprehensive bucket statistics."""
        stats = {
            "total_buckets": len(self.buckets),
            "total_samples": self.total_samples,
            "samples_per_bucket": {
                key: len(bucket) for key, bucket in self.buckets.items()
            },
            "max_bucket_size": max((len(b) for b in self.buckets.values()), default=0),
            "min_bucket_size": min((len(b) for b in self.buckets.values()), default=0),
            "avg_bucket_size": self.total_samples / len(self.buckets) if self.buckets else 0
        }
        
        # Add aspect ratio stats
        if self.buckets:
            ratios = [b.aspect_ratio for b in self.buckets.values()]
            stats["aspect_ratios"] = {
                "min": min(ratios),
                "max": max(ratios),
                "mean": sum(ratios) / len(ratios)
            }
            
        return stats
    
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

    def cleanup(self):
        """Clean up bucket manager resources."""
        try:
            # Clear all buckets
            for bucket in self.buckets.values():
                bucket.clear()
            
            # Clear bucket dictionary
            self.buckets.clear()
            
            # Clear random number generator
            del self.rng
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Successfully cleaned up bucket manager resources")
            
        except Exception as e:
            logger.error(f"Error during bucket manager cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when bucket manager is deleted."""
        self.cleanup()