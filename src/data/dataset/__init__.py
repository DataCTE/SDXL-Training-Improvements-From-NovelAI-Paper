"""Dataset module for SDXL training with efficient data loading and processing.

This module provides a comprehensive set of classes for handling training data:
- Base classes for customization and extension
- Optimized dataset implementation with caching and parallel processing
- Resolution-aware bucket sampling
- Memory-efficient data loading with prefetching
"""

from .base import (
    CustomDatasetBase,
    CustomSamplerBase,
    CustomDataLoaderBase
)
from .dataset import CustomDataset
from .sampler import BucketSampler
from .dataloader import (
    CustomDataLoader,
    create_dataloader
)

__all__ = [
    # Base classes
    'CustomDatasetBase',
    'CustomSamplerBase', 
    'CustomDataLoaderBase',
    
    # Core implementations
    'CustomDataset',
    'BucketSampler',
    'CustomDataLoader',
    
    # Factory functions
    'create_dataloader',
]
