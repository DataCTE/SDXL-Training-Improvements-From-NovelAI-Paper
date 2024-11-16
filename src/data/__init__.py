"""Data handling module for SDXL training and processing.

This module provides components for:
- Dataset handling and efficient data loading
- Tag-based loss weighting for improved training
- High-quality image upscaling
- Image conversion and processing utilities
"""

from .dataset import (
    # Base classes
    CustomDatasetBase,
    CustomSamplerBase,
    CustomDataLoaderBase,
    # Core implementations
    CustomDataset,
    BucketSampler,
    CustomDataLoader,
    # Factory functions
    create_dataloader
)
from .tag_weighter import (
    TagBasedLossWeighter,
    TagStats,
    TagCache
)
from .ultimate_upscaler import (
    UltimateUpscaler,
    USDUMode,
    USDUSFMode
)
from .utils import (
    ImageConverter,
    tensor_to_pil,
    pil_to_tensor,
    get_image_stats,
    converter
)

__all__ = [
    # Dataset components
    'CustomDatasetBase',
    'CustomSamplerBase',
    'CustomDataLoaderBase',
    'CustomDataset',
    'BucketSampler',
    'CustomDataLoader',
    'create_dataloader',
    
    # Tag weighting system
    'TagBasedLossWeighter',
    'TagStats',
    'TagCache',
    
    # Image upscaling
    'UltimateUpscaler',
    'USDUMode',
    'USDUSFMode',
    
    # Image utilities
    'ImageConverter',
    'tensor_to_pil',
    'pil_to_tensor',
    'get_image_stats',
    'converter'
]