import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class ImageGrouper:
    """Handles grouping of images into resolution buckets."""
    
    def __init__(self, bucket_manager):
        self.bucket_manager = bucket_manager
        
    def get_ar_bucket(self, height: int, width: int) -> Tuple[int, int]:
        """Get bucket dimensions based on aspect ratio while maintaining area constraints."""
        ar = width / height
        
        # Calculate dimensions that maintain AR and fit within max_area
        if ar >= 1:  # Wider than tall
            w = min(self.bucket_manager.max_size, 
                   int((self.bucket_manager.max_area * ar)**0.5))
            w = (w // self.bucket_manager.step_size) * self.bucket_manager.step_size
            h = int(w / ar)
            h = (h // self.bucket_manager.step_size) * self.bucket_manager.step_size
        else:  # Taller than wide
            h = min(self.bucket_manager.max_size, 
                   int((self.bucket_manager.max_area / ar)**0.5))
            h = (h // self.bucket_manager.step_size) * self.bucket_manager.step_size
            w = int(h * ar)
            w = (w // self.bucket_manager.step_size) * self.bucket_manager.step_size
        
        # Ensure minimum size
        h = max(h, self.bucket_manager.min_size)
        w = max(w, self.bucket_manager.min_size)
        
        # Ensure one dimension is below 1024
        if h > 1024 and w > 1024:
            if ar >= 1:
                h = 1024
            else:
                w = 1024
        
        return (h, w)
    
    def process_ar_chunk(self, chunk: List[str]) -> Dict[Tuple[int, int], List[str]]:
        """Process a chunk of images for AR-based bucket assignment."""
        local_bucket_groups = defaultdict(list)
        
        for image_path in chunk:
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    orig_height, orig_width = img.size[1], img.size[0]
                    
                    # Get bucket based on aspect ratio
                    target_height, target_width = self.get_ar_bucket(orig_height, orig_width)
                    local_bucket_groups[(target_height, target_width)].append(image_path)
            except Exception as e:
                logger.debug(f"Failed to process image {image_path} for AR grouping: {str(e)}")
                continue
        
        return dict(local_bucket_groups)
    
    def process_standard_chunk(self, chunk: List[str]) -> Dict[Tuple[int, int], List[str]]:
        """Process a chunk of images for standard bucket assignment."""
        local_bucket_groups = defaultdict(list)
        
        for image_path in chunk:
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    orig_height, orig_width = img.size[1], img.size[0]
                    
                    # Find closest bucket resolution
                    target_height, target_width = self.bucket_manager.find_bucket(orig_height, orig_width)
                    local_bucket_groups[(target_height, target_width)].append(image_path)
            except Exception as e:
                logger.debug(f"Failed to process image {image_path} for grouping: {str(e)}")
                continue
        
        return dict(local_bucket_groups)
    
    def merge_bucket_results(self, results: List[Dict[Tuple[int, int], List[str]]]) -> Dict[Tuple[int, int], List[str]]:
        """Merge bucket assignment results from multiple workers."""
        merged = defaultdict(list)
        for result in results:
            for bucket, paths in result.items():
                merged[bucket].extend(paths)
        return dict(merged)
    
    def group_images(self, image_paths: List[str], use_ar: bool = False) -> Dict[Tuple[int, int], List[str]]:
        """Group images by their closest bucket resolution using parallel processing."""
        num_workers = 24
        chunk_size = max(1, len(image_paths) // num_workers)
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
        
        process_func = self.process_ar_chunk if use_ar else self.process_standard_chunk
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Merge results
        bucket_groups = self.merge_bucket_results(results)
        
        # Log statistics
        total_images = sum(len(paths) for paths in bucket_groups.values())
        logger.info(f"Grouped {total_images} images into {len(bucket_groups)} buckets using {num_workers} workers")
        
        return bucket_groups
