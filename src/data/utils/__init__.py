from .system_utils import (
    SystemResources,
    get_system_resources,
    get_optimal_workers,
    get_gpu_memory_usage,
    create_thread_pool,
    adjust_batch_size,
    get_memory_usage_gb,
    log_system_info,
    calculate_chunk_size,
    calculate_optimal_batch_size,
    MemoryCache
)

from .image_utils import (
    load_and_validate_image,
    resize_image,
    get_image_stats,
    validate_image_text_pair
)

from .progress_utils import (
    ProgressStats,
    format_time,
    log_progress,
    create_progress_stats,
    update_progress_stats
)

from .file_utils import (
    ensure_dir,
    get_file_size,
    find_matching_files,
    safe_file_write,
    get_cache_paths,
    cleanup_temp_files
)

from .batch_utils import (
    BatchConfig,
    BatchProcessor,
    create_tensor_buffer,
    process_in_chunks,
    process_in_chunks_sync
)

import asyncio
from typing import List, Any, Callable, Tuple, Dict


async def process_in_chunks_sync(
    items: List[Any],
    chunk_size: int,
    process_fn: Callable,
    num_workers: int,
    **kwargs
) -> Tuple[List[Any], Dict[str, Any]]:
    """Synchronous wrapper for process_in_chunks."""
    return await process_in_chunks(
        items=items,
        chunk_size=chunk_size,
        process_fn=process_fn,
        num_workers=num_workers,
        **kwargs
    )

__all__ = [
    # System utilities
    'SystemResources',
    'get_system_resources',
    'get_optimal_workers',
    'get_gpu_memory_usage',
    'create_thread_pool',
    'adjust_batch_size',
    'get_memory_usage_gb',
    'log_system_info',
    'calculate_chunk_size',
    'calculate_optimal_batch_size',
    'MemoryCache',
    
    # Image utilities
    'load_and_validate_image',
    'resize_image',
    'get_image_stats',
    'validate_image_text_pair',
    
    # Progress utilities
    'ProgressStats',
    'format_time',
    'log_progress',
    'create_progress_stats',
    'update_progress_stats',
    
    # File utilities
    'ensure_dir',
    'get_file_size',
    'find_matching_files',
    'safe_file_write',
    'get_cache_paths',
    'cleanup_temp_files',
    
    # Batch utilities
    'BatchConfig',
    'BatchProcessor',
    'create_tensor_buffer',
    'process_in_chunks',
    'process_in_chunks_sync'
] 