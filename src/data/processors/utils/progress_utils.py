import time
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import torch

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
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    """Log progress with optional extra statistics and callback."""
    progress_msg = (
        f"{prefix}Progress: {stats.processed_items}/{stats.total_items} "
        f"({stats.progress*100:.1f}%)\n"
        f"Performance:\n"
        f"- Processing rate: {stats.rate:.1f} items/s\n"
        f"- Memory usage: {stats.memory_usage_gb:.1f}GB\n"
        f"- Failed items: {stats.failed_items}\n"
        f"- Cache hits/misses: {stats.cache_hits}/{stats.cache_misses}\n"
        f"- Elapsed: {format_time(stats.elapsed)}\n"
        f"- ETA: {format_time(stats.eta_seconds)}"
    )
    
    if extra_stats:
        stats_msg = "\nExtra stats:\n" + "\n".join(
            f"- {k}: {v}" for k, v in extra_stats.items()
        )
        progress_msg += stats_msg
        
    logger.info(progress_msg)
    
    if callback:
        callback(stats.get_stats())