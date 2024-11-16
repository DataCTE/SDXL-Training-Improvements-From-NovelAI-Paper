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
                 min_size: int = 256,
                 max_size: int = 2048,  # Updated to 2048 per SDXL requirements
                 step_size: int = 64,
                 max_area: int = 512 * 768,
                 add_square: bool = True):
        """Initialize the bucket manager.
        
        Args:
            min_size: Minimum dimension size (default: 256)
            max_size: Maximum dimension size (default: 2048)
            step_size: Size increment between buckets (default: 64)
            max_area: Maximum area constraint (default: 512*768)
            add_square: Whether to add square buckets (default: True)
        """
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.max_area = max_area
        self.add_square = add_square
        
        # Generate buckets
        self.buckets = self._generate_buckets()
        logger.info(f"Generated {len(self.buckets)} buckets")
        
    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """Generate resolution buckets following the NovelAI algorithm.
        
        Returns:
            List of (height, width) tuples representing bucket resolutions
        """
        buckets = set()
        
        # Step 1-3: Start with width, find compatible heights
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
        
        # Step 4: Repeat with width/height swapped
        for height in range(self.min_size, self.max_size + 1, self.step_size):
            width = min(
                self.max_size,
                math.floor(self.max_area / height)
            )
            width = (width // self.step_size) * self.step_size
            if width >= self.min_size:
                buckets.add((height, width))
        
        # Step 5: Add square bucket if requested
        if self.add_square:
            buckets.add((512, 512))
        
        # Convert to sorted list for consistent ordering
        buckets = sorted(buckets)
        
        # Log bucket statistics
        total_pixels = sum(h * w for h, w in buckets)
        avg_pixels = total_pixels / len(buckets)
        logger.info(f"Average bucket resolution: {math.sqrt(avg_pixels):.1f} x {math.sqrt(avg_pixels):.1f}")
        logger.info(f"Bucket resolutions: {buckets}")
        
        return buckets
    
    def find_closest_bucket(self, height: int, width: int) -> Tuple[int, int]:
        """Find the closest bucket for a given image size.
        
        The closest bucket is determined by:
        1. Maintaining aspect ratio as closely as possible (error < 0.033)
        2. Minimizing total pixel count while satisfying aspect ratio
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Tuple of (bucket_height, bucket_width)
        """
        target_ar = width / height
        min_ar_error = float('inf')
        best_bucket = None
        
        for bucket_height, bucket_width in self.buckets:
            bucket_ar = bucket_width / bucket_height
            ar_error = abs(bucket_ar - target_ar)
            
            # NovelAI paper specifies aspect ratio error should be < 0.033
            if ar_error < min_ar_error:
                min_ar_error = ar_error
                best_bucket = (bucket_height, bucket_width)
                
                # If we find a very good match, stop searching
                if ar_error < 0.033:
                    break
        
        if best_bucket is None:
            # Fallback to closest area bucket if no good AR match
            target_area = height * width
            best_bucket = min(self.buckets, 
                            key=lambda b: abs(b[0] * b[1] - target_area))
            
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
