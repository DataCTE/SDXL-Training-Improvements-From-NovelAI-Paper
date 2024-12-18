import time
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import torch
from src.utils.logging.metrics import log_error_with_context, log_metrics
from src.data.processors.utils.batch_utils import get_gpu_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class ProgressStats:
    """Progress tracking statistics."""
    total_items: int
    start_time: float = field(default_factory=time.time)
    processed_items: int = 0
    failed_items: int = 0
    last_log_time: float = field(default_factory=time.time)
    last_memory_check: float = field(default_factory=time.time)
    memory_usage_gb: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    batch_size: Optional[int] = None
    device: Optional[torch.device] = None
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def progress(self) -> float:
        """Get progress as fraction."""
        return self.processed_items / max(1, self.total_items)
    
    @property
    def rate(self) -> float:
        """Get processing rate in items/second."""
        return self.processed_items / max(0.1, self.elapsed)
    
    @property
    def eta_seconds(self) -> float:
        """Get estimated time remaining in seconds."""
        remaining = self.total_items - self.processed_items
        return remaining / max(0.1, self.rate)

    def should_log(self, interval: float = 5.0) -> bool:
        """Check if enough time has passed to log progress."""
        current_time = time.time()
        if current_time - self.last_log_time >= interval:
            self.last_log_time = current_time
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'elapsed_seconds': self.elapsed,
            'progress': self.progress,
            'rate': self.rate,
            'eta_seconds': self.eta_seconds,
            'memory_usage_gb': self.memory_usage_gb,
            'error_types': self.error_types,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
        
        if self.batch_size:
            stats['batch_size'] = self.batch_size
            
        if self.device and self.device.type == 'cuda':
            stats['gpu_memory'] = f"{torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated():.1%}"
            
        return stats

    def get(self, key: str, default: Any = None) -> Any:
        """Get statistic by key name."""
        return self.get_stats().get(key, default)

def create_progress_tracker(
    total_items: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None
) -> ProgressStats:
    """Create a new progress tracker with optional batch and device info."""
    return ProgressStats(
        total_items=total_items,
        batch_size=batch_size,
        device=device
    )

def update_tracker(
    stats: ProgressStats,
    processed: int = 0,
    failed: int = 0,
    cache_hits: int = 0,
    cache_misses: int = 0,
    memory_gb: Optional[float] = None,
    error_type: Optional[str] = None
) -> None:
    """Update progress tracker with new statistics."""
    stats.processed_items += processed
    stats.failed_items += failed
    stats.cache_hits += cache_hits
    stats.cache_misses += cache_misses
    
    if memory_gb is not None:
        stats.memory_usage_gb = memory_gb
        
    if error_type:
        stats.error_types[error_type] = stats.error_types.get(error_type, 0) + 1

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def log_progress(
    stats: ProgressStats,
    prefix: str = "",
    extra_stats: Optional[Dict] = None,
    log_interval: int = 10
) -> None:
    """Log progress with improved metrics."""
    if stats.should_log(interval=log_interval):
        try:
            # Prepare base metrics
            metrics = {
                'processed': stats.processed_items,
                'total': stats.total_items,
                'progress': f"{stats.progress*100:.1f}%",
                'rate': f"{stats.rate:.1f} items/s",
                'elapsed': format_time(stats.elapsed),
                'eta': format_time(stats.eta_seconds),
                'failed': stats.failed_items,
                'cache_hits': stats.cache_hits,
                'cache_misses': stats.cache_misses
            }
            
            # Add memory usage if device available
            if stats.device and stats.device.type == 'cuda':
                metrics['gpu_memory'] = f"{get_gpu_memory_usage(stats.device):.1%}"
            
            # Add extra stats
            if extra_stats:
                metrics.update(extra_stats)
            
            # Log using metrics logger
            log_metrics(
                metrics=metrics,
                step=stats.processed_items,
                is_main_process=True,
                step_type="progress"
            )
            
        except Exception as e:
            log_error_with_context(e, "Error logging progress")