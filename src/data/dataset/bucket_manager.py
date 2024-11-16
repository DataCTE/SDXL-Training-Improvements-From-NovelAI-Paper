import logging
import math
from typing import List, Tuple, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)

class BucketManager:
    """Manages image resolution buckets according to NovelAI paper recommendations.
    
    Algorithm steps from the paper:
    1. Start width at 256
    2. Find largest height that satisfies width * height ≤ 512 * 768
    3. Increment width by 64 until reaching 1024
    4. Repeat process with width/height swapped
    5. Add 512×512 bucket
    6. Remove duplicate buckets
    """
    
    def __init__(self, 
                 min_size: int = 512,
                 max_size: int = 2048,  # Updated to 2048 per SDXL requirements
                 step_size: int = 64,
                 max_area: int = 1024 * 1024,  # Updated to match SDXL's typical training resolution
                 add_square: bool = True,
                 adaptive_buckets: bool = True):
        """Initialize the bucket manager.
        
        Args:
            min_size: Minimum dimension size (default: 512)
            max_size: Maximum dimension size (default: 2048)
            step_size: Size increment between buckets (default: 64)
            max_area: Maximum area constraint (default: 1024*1024)
            add_square: Whether to add square buckets (default: True)
            adaptive_buckets: Whether to use adaptive bucket sizes (default: True)
        """
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.max_area = max_area
        self.add_square = add_square
        self.adaptive_buckets = adaptive_buckets
        self.image_stats = defaultdict(int)  # Track image resolution statistics
        
        # Generate buckets
        self.buckets = self._generate_buckets()
        logger.info(f"Generated {len(self.buckets)} buckets")
        
    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """Generate resolution buckets following the NovelAI algorithm with improvements.
        
        Returns:
            List of (height, width) tuples representing bucket resolutions
        """
        buckets = set()
        
        if self.adaptive_buckets and self.image_stats:
            # Generate adaptive buckets based on dataset statistics
            buckets.update(self._generate_adaptive_buckets())
        
        # Add standard buckets from NovelAI algorithm
        for width in range(self.min_size, self.max_size + 1, self.step_size):
            # Find largest height that satisfies area constraint
            height = min(
                self.max_size,  # Don't exceed max_size
                math.floor(self.max_area / width)  # Height from area constraint
            )
            # Round down to nearest step size
            height = (height // self.step_size) * self.step_size
            if height >= self.min_size:
                buckets.add((height, width))
        
        # Add portrait buckets (swap height/width)
        for height in range(self.min_size, self.max_size + 1, self.step_size):
            width = min(
                self.max_size,
                math.floor(self.max_area / height)
            )
            width = (width // self.step_size) * self.step_size
            if width >= self.min_size:
                buckets.add((height, width))
        
        # Add square buckets if requested
        if self.add_square:
            for size in range(self.min_size, self.max_size + 1, self.step_size):
                if size * size <= self.max_area:
                    buckets.add((size, size))
        
        # Convert to sorted list for consistent ordering
        buckets = sorted(buckets)
        
        # Log bucket statistics
        total_pixels = sum(h * w for h, w in buckets)
        avg_pixels = total_pixels / len(buckets)
        logger.info(f"Average bucket resolution: {math.sqrt(avg_pixels):.1f} x {math.sqrt(avg_pixels):.1f}")
        logger.info(f"Bucket resolutions: {buckets}")
        
        return buckets

    def _generate_adaptive_buckets(self) -> set[Tuple[int, int]]:
        """Generate adaptive buckets based on dataset statistics"""
        adaptive_buckets = set()
        
        # Find most common aspect ratios
        aspect_ratios = defaultdict(int)
        for (h, w), count in self.image_stats.items():
            ar = w / h
            # Round to nearest 0.1 to group similar ratios
            ar_rounded = round(ar * 10) / 10
            aspect_ratios[ar_rounded] += count
        
        # Sort by frequency
        common_ars = sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)
        
        # Generate buckets for most common aspect ratios
        for ar, _ in common_ars[:5]:  # Top 5 most common ratios
            # Generate multiple sizes maintaining this aspect ratio
            for base_size in range(self.min_size, self.max_size + 1, self.step_size):
                height = base_size
                width = round(height * ar / self.step_size) * self.step_size
                
                if width >= self.min_size and width <= self.max_size and height * width <= self.max_area:
                    adaptive_buckets.add((height, width))
        
        return adaptive_buckets

    def update_stats(self, height: int, width: int):
        """Update dataset statistics with a new image"""
        self.image_stats[(height, width)] += 1
        
    def find_closest_bucket(self, height: int, width: int) -> Tuple[int, int]:
        """Find the closest bucket for a given image size.
        
        The closest bucket is determined by:
        1. Maintaining aspect ratio as closely as possible (error < 0.033)
        2. Minimizing total pixel count while satisfying aspect ratio
        3. Considering dataset statistics for adaptive bucketing
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Tuple of (target_height, target_width)
        """
        # Update statistics
        self.update_stats(height, width)
        
        original_ar = width / height
        min_ar_error = float('inf')
        best_bucket = None
        min_area = float('inf')
        
        for bucket_height, bucket_width in self.buckets:
            bucket_ar = bucket_width / bucket_height
            ar_error = abs(bucket_ar - original_ar)
            
            # Check if aspect ratio is within tolerance
            if ar_error < 0.033:  # 3.3% aspect ratio tolerance
                area = bucket_height * bucket_width
                if area < min_area:
                    min_area = area
                    best_bucket = (bucket_height, bucket_width)
                    min_ar_error = ar_error
            # If no good match found yet, consider this bucket
            elif best_bucket is None or ar_error < min_ar_error:
                min_ar_error = ar_error
                best_bucket = (bucket_height, bucket_width)
        
        return best_bucket
    
    def get_buckets(self) -> List[Tuple[int, int]]:
        """Get the list of all buckets.
        
        Returns:
            List of (height, width) tuples for all buckets
        """
        return self.buckets.copy()
    
    def assign_to_buckets(self, images: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Assign a list of images to buckets.
        
        Args:
            images: List of (height, width) tuples representing image dimensions
            
        Returns:
            Dictionary mapping bucket dimensions to list of image indices
        """
        bucket_assignments = defaultdict(list)
        
        for idx, (height, width) in enumerate(images):
            bucket = self.find_closest_bucket(height, width)
            bucket_assignments[bucket].append(idx)
            
        return dict(bucket_assignments)
