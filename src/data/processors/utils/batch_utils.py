import torch
import logging
from typing import TypeVar, List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from .system_utils import get_gpu_memory_usage, get_memory_usage_gb, adjust_batch_size
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
import gc
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for batch items
R = TypeVar('R')  # Generic type for processed results

def calculate_optimal_batch_size(
    device: torch.device,
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    target_memory_usage: float = 0.8
) -> int:
    """Calculate optimal batch size based on available GPU memory."""
    if device.type != "cuda":
        return min(8, max_batch_size)  # Default CPU batch size
        
    try:
        current_memory = get_gpu_memory_usage(device)
        available_memory = 1 - current_memory
        
        if available_memory <= 0:
            logger.warning("No available GPU memory, using minimum batch size")
            return min_batch_size
            
        # Calculate batch size based on available memory
        memory_ratio = available_memory / target_memory_usage
        optimal_size = int(min(
            max_batch_size,
            max(min_batch_size, memory_ratio * max_batch_size)
        ))
        
        return optimal_size
        
    except Exception as e:
        log_error_with_context(e, "Error calculating optimal batch size")
        return min_batch_size

async def process_in_chunks(
    items: List[T],
    process_fn: Callable[[List[T], int], Tuple[List[R], Dict]],
    chunk_size: int,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[R], Dict]:
    """Process items in chunks with progress tracking."""
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    all_results: List[R] = []
    
    final_stats = {
        'total': 0,
        'errors': 0,
        'skipped': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'error_types': {},
        'elapsed_seconds': 0
    }
    
    start_time = time.time()
    
    for chunk_id, chunk in enumerate(chunks):
        try:
            # Process chunk and await result
            chunk_results, chunk_stats = await process_fn(chunk, chunk_id)
            
            # Store results
            all_results.extend(chunk_results)
            
            # Update stats
            for key in ['total', 'errors', 'skipped', 'cache_hits', 'cache_misses']:
                final_stats[key] += chunk_stats.get(key, 0)
                
            # Merge error types
            for error_type, count in chunk_stats.get('error_types', {}).items():
                final_stats['error_types'][error_type] = \
                    final_stats['error_types'].get(error_type, 0) + count
            
            # Call progress callback if provided
            if progress_callback is not None:
                try:
                    progress_callback(len(chunk_results), chunk_stats)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            final_stats['errors'] += len(chunk)
            final_stats['error_types']['chunk_failure'] = \
                final_stats['error_types'].get('chunk_failure', 0) + 1
    
    # Calculate elapsed time
    final_stats['elapsed_seconds'] = time.time() - start_time
    
    return all_results, final_stats