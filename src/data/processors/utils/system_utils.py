import torch
import psutil
import logging
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from src.data.processors.thread_config import get_optimal_cpu_threads
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """System resource information."""
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_memory_total: Optional[float] = None
    gpu_memory_used: Optional[float] = None

def get_system_resources() -> SystemResources:
    """Get current system resource information."""
    cpu_count = psutil.cpu_count(logical=True)
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    
    # Get GPU memory info if available
    gpu_total = None
    gpu_used = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_total = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024 * 1024)  # GB
        gpu_used = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # GB
    
    return SystemResources(
        cpu_count=cpu_count,
        total_memory_gb=total_memory,
        available_memory_gb=available_memory,
        gpu_memory_total=gpu_total,
        gpu_memory_used=gpu_used
    )

def get_optimal_workers(memory_per_worker_gb: float = 2.0) -> int:
    """Calculate optimal number of worker threads based on system resources."""
    resources = get_system_resources()
    return min(
        resources.cpu_count,
        max(1, int(resources.available_memory_gb / memory_per_worker_gb)),
        get_optimal_cpu_threads().num_threads
    )

def get_gpu_memory_usage(device: torch.device) -> float:
    """Get current GPU memory usage as a fraction."""
    if device.type == "cuda":
        return torch.cuda.memory_reserved() / torch.cuda.get_device_properties(device).total_memory
    return 0.0

def create_thread_pool(num_workers: Optional[int] = None, memory_per_worker_gb: float = 2.0) -> ThreadPoolExecutor:
    """Create thread pool with optimal number of workers."""
    if num_workers is None:
        num_workers = get_optimal_workers(memory_per_worker_gb)
    return ThreadPoolExecutor(max_workers=num_workers)

def adjust_batch_size(
    current_batch_size: int,
    max_batch_size: int,
    min_batch_size: int,
    current_memory_usage: float,
    max_memory_usage: float,
    growth_factor: float = 0.7,
    reduction_factor: float = 0.8
) -> int:
    """Dynamically adjust batch size based on memory usage."""
    if current_memory_usage > max_memory_usage and current_batch_size > min_batch_size:
        new_size = max(min_batch_size, int(current_batch_size * reduction_factor))
        logger.warning(
            f"High memory usage ({current_memory_usage:.1%}), "
            f"reducing batch size to {new_size}"
        )
        return new_size
    elif current_memory_usage < max_memory_usage * growth_factor and current_batch_size < max_batch_size:
        new_size = min(max_batch_size, int(current_batch_size * (1 + (1 - growth_factor))))
        logger.info(
            f"Low memory usage ({current_memory_usage:.1%}), "
            f"increasing batch size to {new_size}"
        )
        return new_size
    return current_batch_size

def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)

def log_system_info(prefix: str = "") -> None:
    """Log system resource information."""
    resources = get_system_resources()
    logger.info(
        f"{prefix}System resources:\n"
        f"- CPUs: {resources.cpu_count}\n"
        f"- Total Memory: {resources.total_memory_gb:.1f}GB\n"
        f"- Available Memory: {resources.available_memory_gb:.1f}GB\n"
        f"- GPU Memory: {resources.gpu_memory_total:.1f}GB total, "
        f"{resources.gpu_memory_used:.1f}GB used" if resources.gpu_memory_total else "- No GPU detected"
    )

def calculate_chunk_size(
    total_items: int,
    optimal_workers: int,
    min_chunk_size: int = 100,
    items_per_gb: int = 50,
    memory_buffer: float = 0.2  # Keep 20% memory buffer
) -> int:
    """Calculate optimal chunk size based on system resources."""
    resources = get_system_resources()
    available_memory = resources.available_memory_gb * (1 - memory_buffer)
    
    return min(
        max(min_chunk_size, total_items // (optimal_workers * 4)),  # CPU-based size
        max(1, int((available_memory / optimal_workers) * items_per_gb))  # Memory-based size
    )

def calculate_optimal_batch_size(
    device: torch.device,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    target_memory_usage: float = 0.8,
    growth_factor: float = 0.1,
    safety_margin: float = 0.1
) -> int:
    """Calculate optimal batch size with additional safety margin."""
    if device.type != "cuda":
        return max_batch_size
        
    current_memory = get_gpu_memory_usage(device)
    available_memory = (1 - current_memory) * (1 - safety_margin)
    
    # Use growth factor to scale the initial batch size
    optimal_size = min(
        max_batch_size,
        max(
            min_batch_size,
            int(available_memory / target_memory_usage * max_batch_size * (1 + growth_factor))
        )
    )
    
    return optimal_size

class MemoryCache:
    """Memory cache with size limit and automatic cleanup."""
    
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with automatic cleanup."""
        self.cache[key] = value
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up cache if it exceeds size limit."""
        if len(self.cache) > self.max_items:
            # Remove oldest items
            remove_keys = list(self.cache.keys())[:-self.max_items]
            for key in remove_keys:
                del self.cache[key]
            logger.debug(
                f"Cache cleanup: removed {len(remove_keys)} items, "
                f"{len(self.cache)} remaining"
            )
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_items,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        } 