import torch.multiprocessing as mp
import os
import multiprocessing
import torch
from typing import Dict
from dataclasses import dataclass

# Import all processors
from .text_processor import TextProcessor
from .batch_processor import BatchProcessor
from .image_processor import ImageProcessor
from .cache_manager import CacheManager
from .bucket import BucketManager, ImageBucket
from .sampler import AspectBatchSampler

# Import utilities
from .utils import (
    # File utilities
    find_matching_files,
    ensure_dir,
    get_file_size,
    validate_image_text_pair,
    
    # System utilities
    get_system_resources,
    get_optimal_workers,
    get_gpu_memory_usage,
    create_thread_pool,
    adjust_batch_size,
    get_memory_usage_gb,
    log_system_info,
    calculate_chunk_size,
    calculate_optimal_batch_size,
    
    # Progress utilities
    create_progress_tracker,
    update_tracker,
    log_progress,
    format_time,
    
    # Image utilities
    load_and_validate_image,
    resize_image,
    get_image_stats
)

__all__ = [
    # Main processors
    'TextProcessor',
    'BatchProcessor',
    'ImageProcessor',
    'CacheManager',
    'BucketManager',
    'ImageBucket',
    'AspectBatchSampler',
    
    # Text processing utilities
    'TextEmbedder',
    'TagWeighter',
    'TagWeighterConfig',

    #image processing utilities
    'load_and_validate_image',
    'resize_image',
    'get_image_stats',
    
    # Thread and system utilities
    'get_optimal_cpu_threads',
    'get_optimal_thread_config',
    'get_system_resources',
    'get_optimal_workers',
    'get_gpu_memory_usage',
    'create_thread_pool',
    'adjust_batch_size',
    'get_memory_usage_gb',
    'log_system_info',
    
    # File and data utilities
    'find_matching_files',
    'ensure_dir',
    'get_file_size',
    'calculate_optimal_batch_size',
    'calculate_chunk_size',
    'MemoryCache',
    'validate_image_text_pair',
    
    # Progress and logging utilities
    'create_progress_tracker',
    'update_tracker',
    'log_progress',
    'format_time'

]