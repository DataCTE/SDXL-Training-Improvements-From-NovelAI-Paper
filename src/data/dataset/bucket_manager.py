"""
Bucket management module for SDXL training pipeline.

This module implements the NovelAI bucketing algorithm with improvements for
efficient batch processing of images with varying aspect ratios. It handles
dynamic bucket generation and image assignment based on resolution statistics.

Classes:
    BucketManager: Manages image resolution buckets and assignments
"""

import logging
import math
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class BucketManager:
    """Manages image resolution buckets according to NovelAI paper recommendations.
    
    This class implements an improved version of the NovelAI bucketing algorithm
    with support for adaptive buckets and dataset statistics. It ensures efficient
    batch processing while preserving aspect ratios.
    
    Attributes:
        min_resolution: Minimum allowed dimension size
        max_resolution: Maximum allowed dimension size
        resolution_step: Size increment between buckets
        tolerance: Aspect ratio tolerance (default: 0.033 or 3.3%)
        buckets: List of available bucket resolutions
        image_buckets: Mapping of images to their assigned buckets
    """
    
    def __init__(
        self,
        min_resolution: int = 512,
        max_resolution: int = 4096,
        resolution_step: int = 64,
        tolerance: float = 0.033
    ) -> None:
        """Initialize the bucket manager.
        
        Args:
            min_resolution: Minimum allowed dimension
            max_resolution: Maximum allowed dimension
            resolution_step: Resolution increment between buckets
            tolerance: Aspect ratio tolerance (default: 0.033)
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.resolution_step = resolution_step
        self.tolerance = tolerance
        
        # Initialize storage
        self.buckets: List[Tuple[int, int]] = []
        self.image_buckets: Dict[str, Tuple[int, int]] = {}
        self.bucket_images: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self.image_stats: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Generate initial buckets
        self._generate_buckets()
        logger.info("Generated %d initial buckets", len(self.buckets))
        
    def _generate_buckets(self) -> None:
        """Generate initial set of resolution buckets."""
        buckets = set()
        max_area = self.max_resolution * self.max_resolution
        
        # Generate landscape buckets
        for width in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
            for height in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
                if width * height <= max_area:
                    buckets.add((height, width))
        
        # Add square buckets
        for size in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
            if size * size <= max_area:
                buckets.add((size, size))
        
        self.buckets = sorted(list(buckets))
        
    def add_image(
        self,
        image_path: str,
        width: int,
        height: int,
        no_upscale: bool = True
    ) -> None:
        """Add an image to the appropriate bucket.
        
        Args:
            image_path: Path to the image
            width: Image width
            height: Image height
            no_upscale: Whether to prevent upscaling
        """
        # Update statistics
        self.image_stats[(height, width)] += 1
        
        # Find best bucket
        bucket = self._find_bucket(height, width, no_upscale)
        if bucket is not None:
            self.image_buckets[image_path] = bucket
            self.bucket_images[bucket].append(image_path)
            
    def _find_bucket(
        self,
        height: int,
        width: int,
        no_upscale: bool = True
    ) -> Optional[Tuple[int, int]]:
        """Find the most appropriate bucket for given dimensions.
        
        Args:
            height: Image height
            width: Image width
            no_upscale: Whether to prevent upscaling
            
        Returns:
            Tuple of (bucket_height, bucket_width) or None if no suitable bucket
        """
        original_ar = width / height
        best_bucket = None
        min_area = float('inf')
        min_ar_error = float('inf')
        
        for bucket_height, bucket_width in self.buckets:
            # Skip if would require upscaling
            if no_upscale and (bucket_height > height or bucket_width > width):
                continue
                
            bucket_ar = bucket_width / bucket_height
            ar_error = abs(bucket_ar - original_ar)
            
            # Check if aspect ratio is within tolerance
            if ar_error <= self.tolerance:
                area = bucket_height * bucket_width
                if area < min_area:
                    min_area = area
                    best_bucket = (bucket_height, bucket_width)
                    min_ar_error = ar_error
            # Consider this bucket if no good match found yet
            elif best_bucket is None or ar_error < min_ar_error:
                min_ar_error = ar_error
                best_bucket = (bucket_height, bucket_width)
        
        return best_bucket
        
    def get_bucket_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get the target size for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (target_height, target_width) or None if not found
        """
        return self.image_buckets.get(image_path)
        
    def get_bucket_images(self, bucket: Tuple[int, int]) -> List[str]:
        """Get all images assigned to a bucket.
        
        Args:
            bucket: Tuple of (height, width)
            
        Returns:
            List of image paths in the bucket
        """
        return self.bucket_images[bucket]
        
    def finalize_buckets(self) -> None:
        """Finalize bucket assignments and optimize bucket distribution."""
        # Remove empty buckets
        empty_buckets = [
            bucket for bucket in self.buckets
            if not self.bucket_images[bucket]
        ]
        for bucket in empty_buckets:
            self.buckets.remove(bucket)
            del self.bucket_images[bucket]
            
        # Log statistics
        total_images = sum(len(images) for images in self.bucket_images.values())
        logger.info(
            "Finalized %d buckets containing %d images",
            len(self.buckets), total_images
        )
        
        # Log bucket distribution
        for bucket in sorted(self.buckets):
            image_count = len(self.bucket_images[bucket])
            if image_count > 0:
                logger.debug(
                    "Bucket %dx%d: %d images",
                    bucket[1], bucket[0], image_count
                )
