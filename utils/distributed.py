import os
import torch
import torch.distributed as dist
from typing import Optional

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
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return device

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0 