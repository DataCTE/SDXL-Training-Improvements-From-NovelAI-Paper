# src/data/__init__.py
import torch.multiprocessing as mp
import os
import multiprocessing
import torch
from typing import Dict
from dataclasses import dataclass

@dataclass
class ThreadConfig:
    """Global thread configuration"""
    num_threads: int
    chunk_size: int
    prefetch_factor: int

def get_optimal_thread_config() -> ThreadConfig:
    """Calculate optimal thread configuration using 90% of CPU resources"""
    cpu_count = multiprocessing.cpu_count()
    num_threads = max(1, int(cpu_count * 0.9))
    
    return ThreadConfig(
        num_threads=num_threads,
        chunk_size=max(1, num_threads // 2),  # Optimal chunk size for parallel operations
        prefetch_factor=2  # Number of batches to prefetch
    )

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Get thread configuration
thread_config = get_optimal_thread_config()

# Set global thread settings
os.environ["OMP_NUM_THREADS"] = str(thread_config.num_threads)
os.environ["MKL_NUM_THREADS"] = str(thread_config.num_threads)
torch.set_num_threads(thread_config.num_threads)

# Import components after setting thread config
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
    'AspectBatchSampler',
    'get_optimal_thread_config',
    'thread_config'
]