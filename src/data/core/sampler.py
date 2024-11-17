"""Bucket-based dataset sampler for efficient batch processing.

This module provides a memory-efficient sampler that groups data into resolution
buckets for optimized batch processing. It supports resolution-based binning
and maintains efficient caching mechanisms.

Classes:
    SamplerError: Base exception for sampler-related errors
    BucketSampler: Main sampler implementation with resolution handling
"""

import logging
from typing import Dict, List, Optional, Iterator, Union, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

from .base import CustomDatasetBase, CustomSamplerBase

logger = logging.getLogger(__name__)


class SamplerError(Exception):
    """Base exception for sampler-related errors."""


class BucketInitializationError(SamplerError):
    """Exception raised when bucket initialization fails."""


class BucketSampler(CustomSamplerBase):
    """Memory-efficient bucket sampler with resolution handling.
    
    This sampler organizes data into resolution-based buckets for efficient
    batch processing. It supports both standard and resolution-based binning,
    with configurable batch sizes and shuffling.
    
    Attributes:
        dataset: Source dataset containing the samples
        batch_size: Number of samples per batch
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle samples within buckets
        seed: Random seed for reproducibility
        resolution_binning: Whether to use resolution-based binning
        buckets: List of bucket indices
        _bucket_cache: Cache for bucket metadata
        _indices_cache: Cache for shuffled indices
        _cached_weights: Cached bucket sampling weights
        _cached_length: Cached total length
    """
    
    def __init__(
        self,
        dataset: CustomDatasetBase,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        resolution_binning: bool = True
    ) -> None:
        """Initialize the bucket sampler.
        
        Args:
            dataset: Source dataset containing the samples
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples within buckets
            seed: Random seed for reproducibility
            resolution_binning: Whether to use resolution-based binning
            
        Raises:
            BucketInitializationError: If bucket initialization fails
            ValueError: If dataset is invalid or parameters are incorrect
        """
        if not isinstance(dataset, CustomDatasetBase):
            raise ValueError("Dataset must be an instance of CustomDatasetBase")
            
        super().__init__(data_source=dataset)
        
        try:
            # Validate parameters
            if batch_size < 1:
                raise ValueError("batch_size must be >= 1")
            
            # Core parameters
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.seed = seed
            self.resolution_binning = resolution_binning
            
            # Initialize internal state
            self._epoch: int = 0
            self._current_bucket: int = 0
            self._current_index: int = 0
            self._shuffled: bool = False
            
            # Initialize caches
            self._bucket_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
            self._indices_cache: Dict[int, List[int]] = {}
            self._length_cache: Optional[int] = None
            self._cached_weights: Optional[List[float]] = None
            self._cached_length: Optional[int] = None
            
            # Initialize buckets
            self.buckets: List[List[int]] = []
            self._initialize_buckets()
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Log initialization
            logger.info(
                "Initialized BucketSampler with %d buckets, %d total samples",
                len(self.buckets),
                sum(len(b) for b in self.buckets)
            )
            
        except Exception as error:
            logger.error("Failed to initialize BucketSampler: %s", str(error))
            raise BucketInitializationError(str(error)) from error
    
    @property
    def epoch(self) -> int:
        """Get current epoch number.
        
        Returns:
            Current epoch number
        """
        return self._epoch
    
    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set epoch number and clear caches.
        
        Args:
            value: New epoch number
            
        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Epoch number cannot be negative")
        self._epoch = value
        self._clear_caches()
    
    def _initialize_buckets(self) -> None:
        """Initialize bucket data structures.
        
        This method organizes the dataset into resolution-based buckets
        and calculates necessary metadata for efficient sampling.
        
        Raises:
            BucketInitializationError: If bucket initialization fails
            ValueError: If dataset lacks required attributes
        """
        try:
            if not hasattr(self.dataset, 'bucket_data'):
                raise ValueError("Dataset must have bucket_data attribute")
            
            for bucket_dims, image_paths in self.dataset.bucket_data.items():
                bucket_indices = list(range(len(image_paths)))
                self.buckets.append(bucket_indices)
                
                if self.resolution_binning:
                    # Calculate area for resolution-based binning
                    area = bucket_dims[0] * bucket_dims[1]
                    self._bucket_cache[bucket_dims] = {
                        'indices': bucket_indices,
                        'area': area,
                        'aspect_ratio': bucket_dims[0] / bucket_dims[1]
                    }
                else:
                    # Simple bucket assignment
                    self._bucket_cache[bucket_dims] = {
                        'indices': bucket_indices,
                        'dims': bucket_dims
                    }
            
            # Sort buckets by area if using resolution binning
            if self.resolution_binning:
                self.buckets.sort(
                    key=lambda x: self._bucket_cache[x]['area']
                )
            
            # Initialize bucket weights
            total_samples = sum(len(bucket) for bucket in self.buckets)
            self._cached_weights = [
                len(bucket) / total_samples for bucket in self.buckets
            ]
            
            logger.info(
                "Initialized %d buckets with resolution binning %s",
                len(self.buckets),
                'enabled' if self.resolution_binning else 'disabled'
            )
            
        except Exception as error:
            logger.error("Failed to initialize buckets: %s", str(error))
            raise BucketInitializationError(str(error)) from error
    
    def _calculate_bucket_weights(self) -> None:
        """Cache bucket weights calculation.
        
        This method calculates and caches the sampling weights for each bucket
        based on the number of samples they contain.
        """
        if not hasattr(self, '_cached_weights') or self._cached_weights is None:
            self._cached_weights = [len(bucket) for bucket in self.buckets]
    
    def _clear_caches(self) -> None:
        """Clear any cached data between epochs.
        
        This method resets all internal caches and state variables when
        transitioning between epochs.
        """
        # Clear caches
        self._bucket_cache.clear()
        self._indices_cache.clear()
        self._length_cache = None
        self._cached_length = None
        
        # Reset state
        self._current_bucket = 0
        self._current_index = 0
        self._shuffled = False
    
    def __iter__(self) -> Iterator[List[int]]:
        """Memory-efficient iterator implementation.
        
        Yields:
            Batches of indices for sampling from the dataset
            
        Note:
            If shuffle is True, both bucket order and samples within buckets
            are randomized.
        """
        if self.shuffle:
            # Shuffle buckets
            bucket_indices = torch.randperm(len(self.buckets)).tolist()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Shuffled bucket order: %s", bucket_indices)
            
            for bucket_idx in bucket_indices:
                bucket = self.buckets[bucket_idx]
                indices = (
                    torch.randperm(len(bucket)).tolist()
                    if self.shuffle
                    else range(len(bucket))
                )
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Processing bucket %d with %d samples",
                        bucket_idx, len(indices)
                    )
                
                # Yield batches
                batch: List[int] = []
                for idx in indices:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                
                if batch and not self.drop_last:
                    yield batch
        else:
            # Sequential iteration
            for bucket in self.buckets:
                batch: List[int] = []
                for idx in bucket:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                
                if batch and not self.drop_last:
                    yield batch
    
    def __len__(self) -> int:
        """Efficient length calculation with caching.
        
        Returns:
            Total number of batches that will be yielded
            
        Note:
            The result is cached for efficiency after first calculation.
        """
        if self._cached_length is None:
            total = 0
            for bucket in self.buckets:
                bucket_size = len(bucket)
                if self.drop_last:
                    total += bucket_size // self.batch_size
                else:
                    total += (bucket_size + self.batch_size - 1) // self.batch_size
            self._cached_length = total
        return self._cached_length
