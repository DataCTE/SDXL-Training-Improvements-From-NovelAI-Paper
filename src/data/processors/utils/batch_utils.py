import torch
import logging
from typing import TypeVar, List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from .system_utils import get_gpu_memory_usage, get_memory_usage_gb, adjust_batch_size
from .progress_utils import (
    ProgressStats,
    create_progress_tracker,
    update_tracker,
    log_progress
)
from src.config.config import BatchProcessorConfig
import gc
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for batch items
R = TypeVar('R')  # Generic type for processed results

def calculate_optimal_batch_size(
    config: BatchProcessorConfig
) -> int:
    """Calculate optimal batch size based on available GPU memory."""
    if config.device.type != "cuda":
        return min(8, config.max_batch_size)  # Default CPU batch size
        
    try:
        current_memory = get_gpu_memory_usage(config.device)
        available_memory = 1 - current_memory
        
        if available_memory <= 0:
            logger.warning("No available GPU memory, using minimum batch size")
            return config.min_batch_size
            
        # Calculate batch size based on available memory
        memory_ratio = available_memory / config.max_memory_usage
        optimal_size = int(min(
            config.max_batch_size,
            max(
                config.min_batch_size,
                memory_ratio * config.max_batch_size * config.memory_growth_factor
            )
        ))
        
        logger.info(
            f"Calculated optimal batch size: {optimal_size} "
            f"(memory usage: {current_memory:.1%}, target: {config.max_memory_usage:.1%})"
        )
        return optimal_size
        
    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}, using default")
        return min(8, config.max_batch_size)

class BatchProcessor:
    """Generic batch processor with GPU optimization and progress tracking."""
    
    def __init__(
        self,
        config: BatchProcessorConfig,
        executor: ThreadPoolExecutor,
        name: str = "BatchProcessor"
    ):
        self.config = config
        self.executor = executor
        self.name = name
        self.stats = ProgressStats(total_items=0)
        self.last_memory_check = time.time()
        self._tensor_cache = WeakValueDictionary()
        
    def __del__(self):
        """Cleanup when processor is deleted."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear tensor cache
            if hasattr(self, '_tensor_cache'):
                self._tensor_cache.clear()
            
            # Clear CUDA cache if using GPU
            if hasattr(self, 'config') and hasattr(self.config, 'device'):
                if self.config.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def _get_cached_tensor(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=self.config.device)
            self._tensor_cache[key] = tensor
        return tensor
        
    def _should_adjust_batch_size(self) -> bool:
        """Check if it's time to adjust batch size."""
        current_time = time.time()
        if current_time - self.last_memory_check >= self.config.memory_check_interval:
            self.last_memory_check = current_time
            return True
        return False
        
    def _adjust_batch_size(self) -> None:
        """Dynamically adjust batch size based on GPU memory usage."""
        if not self._should_adjust_batch_size():
            return
            
        current_memory = get_gpu_memory_usage(self.config.device)
        
        # Force cleanup if memory usage is too high
        if current_memory > self.config.high_memory_threshold:
            logger.warning("High memory usage detected, forcing cleanup")
            self.cleanup()
            current_memory = get_gpu_memory_usage(self.config.device)
            
        self.config.batch_size = adjust_batch_size(
            current_batch_size=self.config.batch_size,
            max_batch_size=self.config.max_batch_size,
            min_batch_size=self.config.min_batch_size,
            current_memory_usage=current_memory,
            max_memory_usage=self.config.max_memory_usage
        )
        
    async def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], R],
        post_process_fn: Optional[Callable[[R], Any]] = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
        retry_count: int = 3,
        backoff_factor: float = 1.5
    ) -> List[Any]:
        """Process items in batches with automatic memory management and retry logic."""
        self.stats = create_progress_tracker(len(items))
        results = []
        
        try:
            for attempt in range(retry_count):
                try:
                    for start_idx in range(0, len(items), self.config.batch_size):
                        end_idx = min(start_idx + self.config.batch_size, len(items))
                        batch_items = items[start_idx:end_idx]
                        
                        try:
                            # Process batch
                            batch_result = await process_fn(batch_items)
                            
                            # Post-process if needed
                            if post_process_fn:
                                batch_result = post_process_fn(batch_result)
                                
                            results.extend(batch_result)
                            
                            # Update stats
                            update_tracker(
                                self.stats,
                                processed=len(batch_items),
                                memory_gb=get_memory_usage_gb()
                            )
                            
                            # Log progress
                            if self.stats.should_log(self.config.log_interval):
                                log_progress(
                                    self.stats,
                                    prefix=f"{self.name} - ",
                                    extra_stats={
                                        'batch_size': self.config.batch_size,
                                        'gpu_memory': f"{get_gpu_memory_usage(self.config.device):.1%}"
                                    }
                                )
                                
                            # Adjust batch size and cleanup if needed
                            self._adjust_batch_size()
                            
                            # Periodic cleanup
                            if self.stats.processed_items % 1000 == 0:
                                self.cleanup()
                            
                        except Exception as e:
                            logger.error(f"Error processing batch {start_idx}:{end_idx}: {e}")
                            update_tracker(self.stats, failed=len(batch_items))
                            
                            # Cleanup on error
                            self.cleanup()
                            
                        finally:
                            if cleanup_fn:
                                cleanup_fn()
                            
                    return results
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and attempt < retry_count - 1:
                        # Reduce batch size and retry
                        self.config.batch_size = max(
                            1, 
                            int(self.config.batch_size / backoff_factor)
                        )
                        logger.warning(
                            f"OOM error, reducing batch size to {self.config.batch_size} "
                            f"(attempt {attempt + 1}/{retry_count})"
                        )
                        self.cleanup()  # Force cleanup on OOM
                        continue
                    raise
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
            
        finally:
            # Final cleanup
            self.cleanup()
            
        return results

def create_tensor_buffer(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Create reusable tensor buffer for batch processing."""
    return torch.empty(
        (batch_size, channels, height, width),
        dtype=dtype,
        device=device
    )

async def process_in_chunks(
    items: List[T],
    chunk_size: int,
    process_fn: Callable[[List[T], int], Tuple[List[Any], Dict[str, int]]],
    progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """Process items in chunks with progress tracking."""
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    all_results = []
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