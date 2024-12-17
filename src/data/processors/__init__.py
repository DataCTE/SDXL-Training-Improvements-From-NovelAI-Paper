import torch.multiprocessing as mp
import os
import multiprocessing
import torch
from typing import Dict
from dataclasses import dataclass

# Import all processors
from .text_processor import TextProcessor
from .batch_processor import BatchProcessor
from .image_processor import ImageProcessor, ImageProcessorConfig
from .cache_manager import CacheManager
from .bucket import BucketManager, ImageBucket
from .sampler import AspectBatchSampler

# Import utilities
from .utils.thread_config import get_optimal_cpu_threads, get_optimal_thread_config
from .utils.caption.text_embedder import TextEmbedder
from .utils.caption.tag_weighter import TagWeighter, TagWeighterConfig
from .utils import (
    get_system_resources,
    get_optimal_workers,
    get_gpu_memory_usage,
    create_thread_pool,
    adjust_batch_size,
    get_memory_usage_gb,
    log_system_info,
    calculate_chunk_size,
    MemoryCache,
    find_matching_files,
    ensure_dir,
    get_file_size,
    calculate_optimal_batch_size,
    create_progress_stats,
    update_progress_stats,
    format_time,
    log_progress
)

__all__ = [
    # Main processors
    'TextProcessor',
    'BatchProcessor',
    'ImageProcessor',
    'ImageProcessorConfig',
    'CacheManager',
    'BucketManager',
    'ImageBucket',
    'AspectBatchSampler',
    
    # Text processing utilities
    'TextEmbedder',
    'TagWeighter',
    'TagWeighterConfig',
    
    # Thread and system utilities
    'get_optimal_cpu_threads',
    'get_optimal_thread_config',
    'get_system_resources',
    'get_optimal_workers',
    'get_gpu_memory_usage',
    'create_thread_pool',
    'adjust_batch_size',
    'get_memory_usage_gb',
    
    # File and data utilities
    'find_matching_files',
    'ensure_dir',
    'get_file_size',
    'calculate_optimal_batch_size',
    'calculate_chunk_size',
    'MemoryCache',
    
    # Progress and logging utilities
    'log_system_info',
    'create_progress_stats',
    'update_progress_stats',
    'format_time',
    'log_progress'
]