"""Image grouping module for resolution-based bucketing.

This module provides functionality for grouping images into resolution buckets
based on either aspect ratio or standard dimensions. It supports parallel
processing for efficient handling of large image datasets.

Classes:
    ImageGroupingError: Base exception for image grouping errors
    ImageGrouper: Main class for image grouping operations
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict

from PIL import Image
from PIL.Image import DecompressionBombError

logger = logging.getLogger(__name__)


class ImageGroupingError(Exception):
    """Base exception for image grouping related errors."""


class ImageLoadError(ImageGroupingError):
    """Exception raised when an image cannot be loaded or processed."""


class ImageGrouper:
    """Handles grouping of images into resolution buckets.
    
    This class provides methods for organizing images into buckets based on
    their resolutions, supporting both aspect ratio-based and standard
    bucketing approaches. It uses parallel processing for efficiency.
    
    Attributes:
        bucket_manager: Manager object that defines bucket constraints and methods
        MAX_WORKERS: Maximum number of parallel workers for image processing
    """
    
    MAX_WORKERS: int = 24  # Maximum number of parallel workers
    
    def __init__(self, bucket_manager) -> None:
        """Initialize the ImageGrouper.
        
        Args:
            bucket_manager: Object providing bucket management functionality
        """
        self.bucket_manager = bucket_manager
        
    def get_ar_bucket(self, height: int, width: int) -> Tuple[int, int]:
        """Get bucket dimensions based on aspect ratio while maintaining area constraints.
        
        Args:
            height: Original image height
            width: Original image width
            
        Returns:
            Tuple of (target_height, target_width) that maintains aspect ratio
            while fitting within bucket constraints
        """
        ar = width / height
        
        # Calculate dimensions that maintain AR and fit within max_area
        if ar >= 1:  # Wider than tall
            w = min(
                self.bucket_manager.max_size,
                int((self.bucket_manager.max_area * ar)**0.5)
            )
            w = (w // self.bucket_manager.step_size) * self.bucket_manager.step_size
            h = int(w / ar)
            h = (h // self.bucket_manager.step_size) * self.bucket_manager.step_size
        else:  # Taller than wide
            h = min(
                self.bucket_manager.max_size,
                int((self.bucket_manager.max_area / ar)**0.5)
            )
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
        """Process a chunk of images for AR-based bucket assignment.
        
        Args:
            chunk: List of image paths to process
            
        Returns:
            Dictionary mapping bucket dimensions to lists of image paths
            
        Note:
            Images that fail to process are logged and skipped
        """
        local_bucket_groups: DefaultDict[Tuple[int, int], List[str]] = defaultdict(list)
        
        for image_path in chunk:
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    orig_height, orig_width = img.size[1], img.size[0]
                    
                    # Get bucket based on aspect ratio
                    target_height, target_width = self.get_ar_bucket(orig_height, orig_width)
                    local_bucket_groups[(target_height, target_width)].append(image_path)
                    
            except (OSError, DecompressionBombError) as error:
                logger.debug("Failed to process image %s for AR grouping: %s",
                           image_path, str(error))
                continue
            
        return dict(local_bucket_groups)
    
    def process_standard_chunk(self, chunk: List[str]) -> Dict[Tuple[int, int], List[str]]:
        """Process a chunk of images for standard bucket assignment.
        
        Args:
            chunk: List of image paths to process
            
        Returns:
            Dictionary mapping bucket dimensions to lists of image paths
            
        Note:
            Images that fail to process are logged and skipped
        """
        local_bucket_groups: DefaultDict[Tuple[int, int], List[str]] = defaultdict(list)
        
        for image_path in chunk:
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    orig_height, orig_width = img.size[1], img.size[0]
                    
                    # Find closest bucket resolution
                    target_height, target_width = self.bucket_manager.find_bucket(
                        orig_height, orig_width
                    )
                    local_bucket_groups[(target_height, target_width)].append(image_path)
                    
            except (OSError, DecompressionBombError) as error:
                logger.debug("Failed to process image %s for grouping: %s",
                           image_path, str(error))
                continue
            
        return dict(local_bucket_groups)
    
    def merge_bucket_results(
        self,
        results: List[Dict[Tuple[int, int], List[str]]]
    ) -> Dict[Tuple[int, int], List[str]]:
        """Merge bucket assignment results from multiple workers.
        
        Args:
            results: List of bucket assignment dictionaries from workers
            
        Returns:
            Merged dictionary mapping bucket dimensions to lists of image paths
        """
        merged: DefaultDict[Tuple[int, int], List[str]] = defaultdict(list)
        for result in results:
            for bucket, paths in result.items():
                merged[bucket].extend(paths)
        return dict(merged)
    
    def group_images(
        self,
        image_paths: List[str],
        use_ar: bool = False
    ) -> Dict[Tuple[int, int], List[str]]:
        """Group images by their closest bucket resolution using parallel processing.
        
        Args:
            image_paths: List of paths to images to process
            use_ar: If True, use aspect ratio based bucketing, otherwise use
                   standard bucketing
                   
        Returns:
            Dictionary mapping bucket dimensions to lists of image paths
            
        Note:
            Uses parallel processing with ThreadPoolExecutor for efficiency
        """
        chunk_size = max(1, len(image_paths) // self.MAX_WORKERS)
        chunks = [
            image_paths[i:i + chunk_size]
            for i in range(0, len(image_paths), chunk_size)
        ]
        
        process_func = self.process_ar_chunk if use_ar else self.process_standard_chunk
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Merge results
        bucket_groups = self.merge_bucket_results(results)
        
        # Log statistics
        total_images = sum(len(paths) for paths in bucket_groups.values())
        logger.info(
            "Grouped %d images into %d buckets using %d workers",
            total_images, len(bucket_groups), self.MAX_WORKERS
        )
        
        return bucket_groups
