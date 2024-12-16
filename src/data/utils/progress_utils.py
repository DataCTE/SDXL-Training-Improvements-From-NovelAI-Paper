import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

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
    
    def should_check_memory(self, interval: float = 30.0) -> bool:
        """Check if enough time has passed to check memory usage."""
        current_time = time.time()
        if current_time - self.last_memory_check >= interval:
            self.last_memory_check = current_time
            return True
        return False

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
    extra_stats: Optional[Dict] = None
) -> None:
    """Log progress with optional extra statistics."""
    progress_msg = (
        f"{prefix}Progress: {stats.processed_items}/{stats.total_items} "
        f"({stats.progress*100:.1f}%)\n"
        f"Performance:\n"
        f"- Processing rate: {stats.rate:.1f} items/s\n"
        f"- Memory usage: {stats.memory_usage_gb:.1f}GB\n"
        f"- Failed items: {stats.failed_items}\n"
        f"- Elapsed: {format_time(stats.elapsed)}\n"
        f"- ETA: {format_time(stats.eta_seconds)}"
    )
    
    if extra_stats:
        stats_msg = "\nExtra stats:\n" + "\n".join(
            f"- {k}: {v}" for k, v in extra_stats.items()
        )
        progress_msg += stats_msg
        
    logger.info(progress_msg)

def create_progress_stats(total_items: int) -> ProgressStats:
    """Create new progress stats tracker."""
    return ProgressStats(total_items=total_items)

def update_progress_stats(
    stats: ProgressStats,
    processed: int = 0,
    failed: int = 0,
    memory_gb: Optional[float] = None
) -> None:
    """Update progress statistics."""
    stats.processed_items += processed
    stats.failed_items += failed
    if memory_gb is not None:
        stats.memory_usage_gb = memory_gb 