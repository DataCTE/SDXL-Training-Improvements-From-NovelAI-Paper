import os
import torch
import torch.distributed as dist
from typing import Optional
from utils.error_handling import error_handler
import datetime
import time
import logging

logger = logging.getLogger(__name__)

@error_handler
def setup_distributed(
    local_rank: int,
    world_size: Optional[int] = None,
    port: str = "12355"
) -> torch.device:
    """Setup distributed training"""
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    
    # Initialize process group
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dist.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=local_rank,
                timeout=datetime.timedelta(minutes=30)
            )
            # Test communication
            test_tensor = torch.zeros(1, device=f"cuda:{local_rank}")
            dist.all_reduce(test_tensor)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to initialize process group after {max_retries} attempts")
            time.sleep(1)
    
    # Set device
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    # Set memory allocator settings
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats(device=None)
    # Optional: limit maximum allocated memory
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some headroom
    device = torch.device(f"cuda:{local_rank}")
    
    if torch.cuda.is_available():
        # Set memory allocator for better fragmentation handling
        torch.cuda.set_per_process_memory_fraction(0.95)
        # Enable memory caching
        torch.cuda.memory.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.95)
    
    # Add NCCL backend check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, NCCL backend requires CUDA")
    
    if not hasattr(torch.distributed, 'NCCL_BLOCKING_WAIT'):
        logger.warning("NCCL_BLOCKING_WAIT not available, this might cause hangs")
    
    return device

@error_handler
def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        try:
            # Ensure all processes reach cleanup
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logger.error(f"Error during distributed cleanup: {e}")
            # Force cleanup even if barrier fails
            dist.destroy_process_group()

@error_handler
def is_main_process() -> bool:
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0 

@error_handler
def check_process_group_health():
    """Check if process group is healthy"""
    if not dist.is_initialized():
        return True
    try:
        # Test communication
        test_tensor = torch.zeros(1, device=f"cuda:{dist.get_rank()}")
        dist.all_reduce(test_tensor)
        return True
    except Exception as e:
        logger.error(f"Process group health check failed: {e}")
        return False

@error_handler
def check_gpu_topology():
    """Check GPU topology and warn about suboptimal configurations"""
    if not torch.cuda.is_available():
        return
        
    try:
        # Check NVLink connectivity
        nvlink_output = os.popen('nvidia-smi nvlink -s').read()
        if 'No NVLink found' in nvlink_output:
            logger.warning("No NVLink detected - communication may be slower")
            
        # Check GPU topology
        topo_output = os.popen('nvidia-smi topo -m').read()
        if 'PCIe' in topo_output:
            logger.warning("GPUs connected via PCIe - performance may be impacted")
    except Exception as e:
        logger.warning(f"Failed to check GPU topology: {e}")

@error_handler
def validate_env_vars():
    """Validate required environment variables"""
    required_vars = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK")
    }
    
    missing = [k for k, v in required_vars.items() if v is None]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

@error_handler
def recover_from_timeout():
    """Attempt to recover from process group timeout"""
    if not dist.is_initialized():
        return False
        
    try:
        # Test if process group is still functional
        if not check_process_group_health():
            # Reinitialize process group
            dist.destroy_process_group()
            setup_distributed(
                local_rank=int(os.environ.get("LOCAL_RANK")),
                world_size=int(os.environ.get("WORLD_SIZE"))
            )
            return True
    except Exception as e:
        logger.error(f"Failed to recover from timeout: {e}")
        return False

@error_handler
def log_memory_stats(rank: int):
    """Log memory statistics for each GPU"""
    if not torch.cuda.is_available():
        return
        
    try:
        allocated = torch.cuda.memory_allocated(rank)
        reserved = torch.cuda.memory_reserved(rank)
        max_allocated = torch.cuda.max_memory_allocated(rank)
        
        logger.info(
            f"[Rank {rank}] Memory Stats:\n"
            f"  Allocated: {allocated / 1e9:.2f}GB\n"
            f"  Reserved: {reserved / 1e9:.2f}GB\n"
            f"  Max Allocated: {max_allocated / 1e9:.2f}GB"
        )
    except Exception as e:
        logger.warning(f"Failed to log memory stats: {e}")