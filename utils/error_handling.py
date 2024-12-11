import sys
import traceback
from functools import wraps
import logging
from typing import Optional, Callable
import torch.distributed as dist

def setup_logger(name: str, rank: Optional[int] = None):
    """Setup logger with rank-aware formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    rank_prefix = f"[Rank {rank}] " if rank is not None else ""
    formatter = logging.Formatter(
        f'{rank_prefix}%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = f"error_rank{rank}.log" if rank is not None else "error.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def error_handler(func: Callable) -> Callable:
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if dist.is_initialized():
                # Ensure all processes know about the error
                is_error = torch.tensor(1, device="cuda")
                dist.all_reduce(is_error)
                dist.barrier()  # Synchronize before cleanup
                dist.destroy_process_group()
            raise
    return wrapper 