from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple, Iterator
import logging
import numpy as np
import heapq
from collections import defaultdict
import gc
from weakref import WeakValueDictionary
from src.config.config import BucketConfig

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
    
    def __init__(self, config: BucketConfig):
        """Initialize with consolidated config."""
        self.config = config
        self.max_width, self.max_height = config.max_image_size
        self.min_width, self.min_height = config.min_image_size
        
        self.buckets: Dict[str, ImageBucket] = WeakValueDictionary()
        self.total_samples = 0
        self.epoch = 0
        self.rng = np.random.RandomState()

        logger.info("Initialized empty BucketManager (no step-based creation).")
    
    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Round dimensions to multiples of config.bucket_step; clamp to min/max."""
        step = self.config.bucket_step
        width = round(width / step) * step
        height = round(height / step) * step

        width = max(min(width, self.max_width), self.min_width)
        height = max(min(height, self.max_height), self.min_height)
        return width, height
    
    def create_buckets_from_dataset(
        self,
        items: List[Dict],
        min_count_for_bucket: int = 1
    ) -> None:
        """
        Dynamically create buckets based on the actual images in 'items'.
        Each item must have 'width' and 'height' fields.
        Only create a bucket if that resolution has at least 'min_count_for_bucket' images.
        """
        resolution_counts = {}
        skip_details = []

        for i, item in enumerate(items):
            orig_w = item.get("width")
            orig_h = item.get("height")
            if not orig_w or not orig_h:
                skip_details.append(f"Item {i} missing width/height: {item}")
                continue

            w, h = self._validate_dimensions(orig_w, orig_h)
            aspect = w / h if h else 0
            resolution = w * h

            if resolution < self.config.min_bucket_resolution:
                skip_details.append(
                    f"Skip item {i}: under min resolution {resolution} < {self.config.min_bucket_resolution} "
                    f"[orig={orig_w}x{orig_h}, final={w}x{h}, aspect={aspect:.3f}]"
                )
                continue

            if aspect > self.config.max_aspect_ratio or aspect < (1 / self.config.max_aspect_ratio):
                skip_details.append(
                    f"Skip item {i}: aspect {aspect:.3f} not in "
                    f"[{1/self.config.max_aspect_ratio:.3f}, {self.config.max_aspect_ratio:.3f}] "
                    f"[orig={orig_w}x{orig_h}, final={w}x{h}, res={resolution}]"
                )
                continue

            bucket_key = f"{w}x{h}"
            resolution_counts[bucket_key] = resolution_counts.get(bucket_key, 0) + 1

        for key, count in resolution_counts.items():
            if count >= min_count_for_bucket:
                w_str, h_str = key.split("x")
                w, h = int(w_str), int(h_str)
                self.buckets[key] = ImageBucket(width=w, height=h)

        if skip_details:
            logger.debug("Skipped items:\n" + "\n".join(skip_details))

        if len(self.buckets) == 0:
            # Create a fallback bucket to avoid an empty bucket list
            fallback_w, fallback_h = 1024, 1024
            logger.warning(
                f"No dynamic buckets created; adding fallback bucket "
                f"[{fallback_w}x{fallback_h}] to prevent empty bucket list."
            )
            fallback_key = f"{fallback_w}x{fallback_h}"
            self.buckets[fallback_key] = ImageBucket(width=fallback_w, height=fallback_h)

        logger.info(
            f"After fallback check, total {len(self.buckets)} bucket(s): {list(self.buckets.keys())}"
        )

        logger.info(f"Created {len(self.buckets)} dynamic buckets from dataset.")

        # If zero buckets, raise an error with skip info
        if len(self.buckets) == 0:
            err_msg = "No dynamic buckets created. Detailed skip reasons:\n" + "\n".join(skip_details)
            logger.error(err_msg)
            raise ValueError(err_msg)
    
    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Return the matching bucket for given dimensions, if any."""
        w, h = self._validate_dimensions(width, height)
        bucket_key = f"{w}x{h}"
        return self.buckets.get(bucket_key, None)
    
    def assign_to_buckets(self, items: List[Dict], shuffle: bool = True) -> None:
        """Assign items to existing buckets (created via create_buckets_from_dataset)."""
        self.total_samples = 0
        for bucket in self.buckets.values():
            bucket.clear()

        for idx, item in enumerate(items):
            w = item.get("width")
            h = item.get("height")
            if not w or not h:
                continue

            bucket = self.find_bucket(w, h)
            if bucket:
                bucket.add_index(idx)
                self.total_samples += 1

        if shuffle:
            self.shuffle_buckets()

        gc.collect()
    
    def shuffle_buckets(self):
        """Shuffle indices in each bucket."""
        for bucket in self.buckets.values():
            self.rng.shuffle(bucket.indices)
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
            return (aspect <= self.config.max_aspect_ratio and 
                    aspect >= 1/self.config.max_aspect_ratio and
                    width * height >= self.config.min_bucket_resolution)
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