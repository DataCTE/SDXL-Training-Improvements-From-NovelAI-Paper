import torch
import logging
from typing import TypeVar, List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from .system_utils import get_gpu_memory_usage, get_memory_usage_gb, adjust_batch_size
from .progress_utils import ProgressStats, create_progress_stats, update_progress_stats, log_progress

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for batch items
R = TypeVar('R')  # Generic type for processed results

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int
    device: torch.device
    dtype: torch.dtype = torch.float16
    max_memory_usage: float = 0.9
    prefetch_factor: int = 2
    log_interval: float = 5.0

class BatchProcessor:
    """Generic batch processor with GPU optimization and progress tracking."""
    
    def __init__(
        self,
        config: BatchConfig,
        executor: ThreadPoolExecutor,
        name: str = "BatchProcessor"
    ):
        self.config = config
        self.executor = executor
        self.name = name
        self.stats = ProgressStats(total_items=0)
        self.last_memory_check = time.time()
        
    def _should_adjust_batch_size(self) -> bool:
        """Check if it's time to adjust batch size."""
        current_time = time.time()
        if current_time - self.last_memory_check >= 30:  # Check every 30 seconds
            self.last_memory_check = current_time
            return True
        return False
        
    def _adjust_batch_size(self) -> None:
        """Dynamically adjust batch size based on GPU memory usage."""
        if not self._should_adjust_batch_size():
            return
            
        current_memory = get_gpu_memory_usage(self.config.device)
        self.config.batch_size = adjust_batch_size(
            current_batch_size=self.config.batch_size,
            max_batch_size=64,
            min_batch_size=1,
            current_memory_usage=current_memory,
            max_memory_usage=self.config.max_memory_usage
        )
        
    async def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], R],
        post_process_fn: Optional[Callable[[R], Any]] = None,
        cleanup_fn: Optional[Callable[[], None]] = None
    ) -> List[Any]:
        """Process items in batches with automatic memory management."""
        self.stats = create_progress_stats(len(items))
        results = []
        
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
                    update_progress_stats(
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
                        
                    # Adjust batch size
                    self._adjust_batch_size()
                    
                except Exception as e:
                    logger.error(f"Error processing batch {start_idx}:{end_idx}: {e}")
                    update_progress_stats(self.stats, failed=len(batch_items))
                    
                finally:
                    if cleanup_fn:
                        cleanup_fn()
                        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
            
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

def process_in_chunks(
    items: List[T],
    chunk_size: int,
    process_fn: Callable[[List[T]], Tuple[List[Any], Dict[str, int]]],
    num_workers: int,
    progress_interval: float = 5.0
) -> Tuple[List[Any], Dict[str, Any]]:
    """Process items in chunks with parallel execution."""
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    stats = create_progress_stats(len(items))
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk_id, chunk in enumerate(chunks):
            future = executor.submit(process_fn, chunk, chunk_id)
            futures.append((future, chunk_id))
            
        for future, chunk_id in futures:
            try:
                chunk_results, chunk_stats = future.result()
                results.extend(chunk_results)
                
                # Update error types
                if 'error_types' in chunk_stats:
                    for error_type, count in chunk_stats['error_types'].items():
                        stats.error_types[error_type] = stats.error_types.get(error_type, 0) + count
                
                update_progress_stats(
                    stats,
                    processed=chunk_stats['total'],
                    failed=chunk_stats.get('errors', 0),
                    memory_gb=get_memory_usage_gb()
                )
                
                if stats.should_log(progress_interval):
                    log_progress(
                        stats,
                        prefix=f"Chunk {chunk_id}/{len(chunks)} - ",
                        extra_stats=chunk_stats
                    )
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {e}")
                update_progress_stats(stats, failed=chunk_size)
                
    return results, stats.get_stats() 