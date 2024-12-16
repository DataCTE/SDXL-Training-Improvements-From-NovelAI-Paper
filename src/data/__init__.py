# src/data/__init__.py
import torch.multiprocessing as mp
import os
import torch

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Set number of threads for CPU operations
os.environ["OMP_NUM_THREADS"] = "24"  # Adjust based on your CPU
os.environ["MKL_NUM_THREADS"] = "24"  # Adjust based on your CPU
torch.set_num_threads(24)  # Adjust based on your CPU

from .dataset import NovelAIDataset, NovelAIDatasetConfig
from .text_embedder import TextEmbedder
from .tag_weighter import TagWeighter, TagWeightingConfig
from .bucket import AspectRatioBucket, ImageBucket
from .image_processor import ImageProcessor, ImageProcessorConfig
from .cache_manager import CacheManager
from .batch_processor import BatchProcessor
from .sampler import AspectBatchSampler

__all__ = [
    'NovelAIDataset',
    'NovelAIDatasetConfig',
    'TextEmbedder',
    'TagWeighter',
    'TagWeightingConfig',
    'AspectRatioBucket',
    'ImageBucket',
    'ImageProcessor',
    'ImageProcessorConfig',
    'CacheManager',
    'BatchProcessor',
    'AspectBatchSampler'
]