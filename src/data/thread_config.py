# src/data/thread_config.py
import multiprocessing
from dataclasses import dataclass

@dataclass
class ThreadConfig:
    """Global thread configuration"""
    num_threads: int
    chunk_size: int
    prefetch_factor: int

def get_optimal_cpu_threads() -> ThreadConfig:
    """Calculate optimal thread configuration using 90% of CPU resources"""
    cpu_count = multiprocessing.cpu_count()
    num_threads = max(1, int(cpu_count * 0.9))
    
    return ThreadConfig(
        num_threads=num_threads,
        chunk_size=max(1, num_threads // 2),
        prefetch_factor=2
    )