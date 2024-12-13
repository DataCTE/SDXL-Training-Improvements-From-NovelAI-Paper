import torch
import psutil
import logging
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from src.data.processors.utils.thread_config import get_optimal_cpu_threads
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakValueDictionary
import gc
import time
from src.utils.logging.metrics import log_system_metrics, log_error_with_context

logger = logging.getLogger(__name__)

class CacheValue:
    """Wrapper class for cache values to enable weak references."""
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"CacheValue({self.value})"

@dataclass
class SystemResources:
    """System resource information."""
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_memory_total: Optional[float] = None
    gpu_memory_used: Optional[float] = None

def get_system_resources() -> SystemResources:
    """Get current system resource information with logging."""
    try:
        resources = SystemResources(
            cpu_count=psutil.cpu_count(logical=True),
            total_memory_gb=psutil.virtual_memory().total / (1024**3),  # GB
            available_memory_gb=psutil.virtual_memory().available / (1024**3),  # GB
            gpu_memory_total=None,
            gpu_memory_used=None
        )
        
        # Get GPU info if available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            resources.gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            resources.gpu_memory_used = torch.cuda.memory_reserved() / (1024**3)
        
        # Log system metrics
        log_system_metrics(prefix="System Resources: ")
        
        return resources
        
    except Exception as e:
        log_error_with_context(e, "Error getting system resources")
        raise

def get_optimal_workers(memory_per_worker_gb: float = 1.0) -> int:
    """Calculate optimal number of worker processes based on system resources."""
    try:
        resources = get_system_resources()
        
        # Calculate workers based on CPU cores
        cpu_workers = max(1, resources.cpu_count - 1)  # Leave one core free
        
        # Calculate workers based on available memory
        memory_workers = max(1, int(resources.available_memory_gb / memory_per_worker_gb))
        
        # Use the minimum of CPU and memory-based workers
        optimal_workers = min(cpu_workers, memory_workers)
        
        logger.info(
            f"Calculated optimal workers:\n"
            f"- CPU-based: {cpu_workers}\n"
            f"- Memory-based: {memory_workers}\n"
            f"- Final: {optimal_workers}"
        )
        
        return optimal_workers
        
    except Exception as e:
        logger.error(f"Error calculating optimal workers: {e}")
        return max(1, (psutil.cpu_count(logical=True) or 2) - 1)

def get_gpu_memory_usage(device: torch.device) -> float:
    """Get current GPU memory usage as a fraction."""
    if device.type == "cuda":
        return torch.cuda.memory_reserved() / torch.cuda.get_device_properties(device).total_memory
    return 0.0

def create_thread_pool(num_workers: int, **kwargs) -> ThreadPoolExecutor:
    """Create a thread pool with proper error handling."""
    try:
        return ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix='batch_worker'
        )
    except TypeError:
        # Fallback for Python versions that don't support thread_name_prefix
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
    """Memory cache with size limit, automatic cleanup, and weak references."""
    
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.cache: WeakValueDictionary = WeakValueDictionary()
        self.hits = 0
        self.misses = 0
        self._last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        self._auto_cleanup()
        try:
            cached = self.cache.get(key)
            if cached is not None:
                self.hits += 1
                return cached.value
        except KeyError:
            pass
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with automatic cleanup."""
        self._auto_cleanup()
        try:
            # Wrap value in CacheValue if it's a dict or other non-referenceable type
            if isinstance(value, (dict, list, set, tuple)):
                cache_value = CacheValue(value)
            else:
                cache_value = value
            self.cache[key] = cache_value
            self._cleanup_if_needed()
        except Exception as e:
            logger.warning(f"Failed to cache item: {e}")
    
    def _auto_cleanup(self) -> None:
        """Periodically force garbage collection."""
        current_time = time.time()
        if current_time - self._last_cleanup > 600:
            self.clear(force_gc=True)
            self._last_cleanup = current_time
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit."""
        if len(self.cache) > self.max_items:
            # Remove oldest items
            remove_keys = list(self.cache.keys())[:-self.max_items]
            for key in remove_keys:
                try:
                    del self.cache[key]
                except KeyError:
                    pass
            logger.debug(
                f"Cache cleanup: removed {len(remove_keys)} items, "
                f"{len(self.cache)} remaining"
            )
    
    def clear(self, force_gc: bool = False) -> None:
        """Clear the cache and optionally force garbage collection."""
        self.cache.clear()
        if force_gc:
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_items,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "memory_usage_mb": get_memory_usage_gb() * 1024
        } 

async def cleanup_processor(obj):
    """Clean up processor resources with logging."""
    try:
        logger.info(f"Starting cleanup for {obj.__class__.__name__}")
        
        cleanup_stats = {
            'tensors_cleared': 0,
            'cache_items_cleared': 0,
            'gpu_memory_start': get_gpu_memory_usage(obj.device) if hasattr(obj, 'device') else 0
        }

        # Thread pool
        if hasattr(obj, 'executor'):
            obj.executor.shutdown(wait=True)
            logger.debug("Thread pool shutdown complete")

        # Clear caches
        if hasattr(obj, '_tensor_cache'):
            cleanup_stats['tensors_cleared'] = len(obj._tensor_cache)
            obj._tensor_cache.clear()
            
        if hasattr(obj, 'memory_cache'):
            cleanup_stats['cache_items_cleared'] = len(obj.memory_cache)
            obj.memory_cache.clear()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleanup_stats['gpu_memory_end'] = get_gpu_memory_usage(obj.device) if hasattr(obj, 'device') else 0
            
        # Force garbage collection
        gc.collect()

        # Log cleanup stats
        logger.info(
            f"Cleanup completed for {obj.__class__.__name__}:\n"
            f"- Tensors cleared: {cleanup_stats['tensors_cleared']}\n"
            f"- Cache items cleared: {cleanup_stats['cache_items_cleared']}\n"
            f"- GPU memory freed: {cleanup_stats['gpu_memory_start'] - cleanup_stats['gpu_memory_end']:.1%}"
        )
        
    except Exception as e:
        log_error_with_context(e, f"Error during cleanup of {obj.__class__.__name__}") 