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

class BatchProcessor:
    """Generic batch processor with GPU optimization and progress tracking."""
    
    def __init__(
        self,
        config: Any,
        executor: ThreadPoolExecutor,
        name: str = "BatchProcessor"
    ):
        """Initialize batch processor with configuration."""
        try:
            self.config = config
            self.executor = executor
            self.name = name
            self._tensor_cache = WeakValueDictionary()
            
            # Log initialization
            logger.info(
                f"Initialized {name}:\n"
                f"- Device: {config.device}\n"
                f"- Batch size: {config.batch_size}\n"
                f"- Cache enabled: {getattr(config, 'use_cache', False)}"
            )
            log_system_metrics(prefix=f"{name} initialization: ")
            
        except Exception as e:
            log_error_with_context(e, f"Error initializing {name}")
            raise

    def __del__(self):
        """Cleanup when processor is deleted."""
        self.cleanup()

    async def process_batch(
        self,
        batch_items: List[T],
        stats: Dict[str, Any]
    ) -> Tuple[List[R], Dict[str, Any]]:
        """Process a batch of items with resource tracking."""
        try:
            batch_stats = {
                'start_time': time.time(),
                'start_memory': get_gpu_memory_usage(self.config.device),
                'batch_size': len(batch_items),
                'processed': 0,
                'errors': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }

            # Process items
            results = []
            for item in batch_items:
                try:
                    result = await self._process_item(item)
                    if result is not None:
                        results.append(result)
                        batch_stats['processed'] += 1
                except Exception as e:
                    batch_stats['errors'] += 1
                    logger.error(f"Error processing item: {str(e)[:200]}...")

            # Update final stats
            batch_stats.update({
                'end_memory': get_gpu_memory_usage(self.config.device),
                'duration': time.time() - batch_stats['start_time'],
                'memory_change': (
                    get_gpu_memory_usage(self.config.device) - 
                    batch_stats['start_memory']
                )
            })

            # Log batch metrics
            log_metrics({
                'batch_size': batch_stats['batch_size'],
                'processed': batch_stats['processed'],
                'errors': batch_stats['errors'],
                'duration': f"{batch_stats['duration']:.2f}s",
                'items_per_second': f"{batch_stats['processed']/batch_stats['duration']:.1f}",
                'memory_usage': f"{batch_stats['end_memory']:.1%}",
                'memory_change': f"{batch_stats['memory_change']:.1%}"
            }, step=stats.total_items, step_type="batch")

            return results, batch_stats

        except Exception as e:
            log_error_with_context(e, "Error in batch processing")
            return [], {'errors': len(batch_items)}

    async def cleanup(self):
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
            
            logger.info(f"Successfully cleaned up {self.name} resources")
            
        except Exception as e:
            logger.error(f"Error during {self.name} cleanup: {str(e)}")
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass

    async def _process_item(self, item: T) -> Optional[R]:
        """Process a single item. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _process_item")

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