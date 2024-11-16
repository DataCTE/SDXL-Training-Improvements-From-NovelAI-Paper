import logging
from typing import Optional, Iterator

import torch

from .base import CustomDatasetBase, CustomSamplerBase

logger = logging.getLogger(__name__)

class BucketSampler(CustomSamplerBase):
    """Memory-efficient bucket sampler with resolution handling"""

    def __init__(
            self,
            dataset: CustomDatasetBase,
            batch_size: int = 1,
            drop_last: bool = False,
            shuffle: bool = True,
            seed: Optional[int] = None,
            resolution_binning: bool = True
        ):
        # Initialize parent with dataset
        super().__init__(data_source=dataset)

        try:
            # Core parameters
            self.dataset = dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.seed = seed
            self.resolution_binning = resolution_binning

            # Initialize internal state
            self._epoch = 0
            self._current_bucket = 0
            self._current_index = 0
            self._shuffled = False

            # Initialize caches
            self._bucket_cache = {}
            self._indices_cache = {}
            self._length_cache = None
            self._cached_weights = None
            self._cached_length = None

            # Initialize buckets efficiently
            self._initialize_buckets()

            # Log initialization
            logger.info(
                "Initialized BucketSampler with %d buckets, %d total samples",
                len(self.buckets),
                sum(len(b) for b in self.buckets)
            )

        except Exception as e:
            logger.error("Failed to initialize BucketSampler: %s", str(e))
            raise

    @property
    def epoch(self) -> int:
        """Get current epoch number"""
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set epoch number and clear caches"""
        self._epoch = value
        self._clear_caches()

    def _initialize_buckets(self) -> None:
        """Initialize buckets with memory optimization"""
        self.buckets = []
        self.bucket_weights = []
        self.total_samples = 0

        try:
            # Get image sizes from dataset
            image_sizes = [(i, self.dataset.get_image_size(i)) for i in range(len(self.dataset))]

            if self.resolution_binning:
                # Group by resolution
                resolution_groups = {}
                for idx, size in image_sizes:
                    res_key = (size[0] // 64 * 64, size[1] // 64 * 64)  # Group by 64-pixel bins
                    if res_key not in resolution_groups:
                        resolution_groups[res_key] = []
                    resolution_groups[res_key].append(idx)

                # Create buckets from groups
                for indices in resolution_groups.values():
                    if len(indices) >= self.batch_size or not self.drop_last:
                        self.buckets.append(indices)
                        self.bucket_weights.append(len(indices))
                        self.total_samples += len(indices)
            else:
                # Single bucket with all indices
                all_indices = [idx for idx, _ in image_sizes]
                self.buckets.append(all_indices)
                self.bucket_weights.append(len(all_indices))
                self.total_samples = len(all_indices)

        except Exception as e:
            logger.error("Failed to initialize buckets: %s", str(e))
            raise

    def _calculate_bucket_weights(self) -> None:
        """Cache bucket weights calculation"""
        if not hasattr(self, '_cached_weights') or self._cached_weights is None:
            self._cached_weights = [len(bucket) for bucket in self.buckets]

    def _clear_caches(self) -> None:
        """Clear any cached data between epochs"""
        # Clear bucket cache
        if hasattr(self, '_bucket_cache'):
            self._bucket_cache.clear()

        # Clear indices cache
        if hasattr(self, '_indices_cache'):
            self._indices_cache.clear()

        # Clear length caches
        self._length_cache = None
        self._cached_length = None

        # Reset any other epoch-specific state
        self._current_bucket = 0
        self._current_index = 0
        self._shuffled = False

    def __iter__(self) -> Iterator[int]:
        """Memory-efficient iterator implementation"""
        if self.shuffle:
            # Shuffle buckets
            bucket_indices = torch.randperm(len(self.buckets)).tolist()

            # Log shuffling if debug enabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Shuffled bucket order: %s", bucket_indices)

            for bucket_idx in bucket_indices:
                bucket = self.buckets[bucket_idx]
                if self.shuffle:
                    indices = torch.randperm(len(bucket)).tolist()
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Shuffled indices for bucket %d: %s", bucket_idx, indices)
                else:
                    indices = range(len(bucket))

                # Yield indices
                batch = []
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
                batch = []
                for idx in bucket:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

                if batch and not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        """Efficient length calculation with caching"""
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
