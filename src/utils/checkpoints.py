import torch
import torch.distributed as dist
import os
import logging
from typing import Dict, Any
import traceback

logger = logging.getLogger(__name__)

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
    global_step: int,
    current_epoch: int,
    half_precision: bool = True
) -> None:
    """Save checkpoint efficiently with atomic operation.
    
    Args:
        path: Path to save checkpoint to
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        config: Training configuration
        global_step: Current global step
        current_epoch: Current epoch
        half_precision: Whether to save in half precision
    """
    try:
        if not dist.is_initialized() or dist.get_rank() == 0:  # Only save on main process
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'step': global_step,
                'epoch': current_epoch,
            }
            
            # Only save scheduler state if it exists
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            
            # Save in half precision for smaller file size
            if half_precision:
                for k, v in checkpoint['model'].items():
                    if v.dtype in [torch.float32, torch.float64]:
                        checkpoint['model'][k] = v.half()
                    
            # Save atomically
            tmp_path = path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)  # Atomic operation
            
            logger.info(f"Saved checkpoint to {path}")
            
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """Load checkpoint with error handling.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        map_location: Device to load tensors onto
        
    Returns:
        Dict containing checkpoint info (config, step, epoch)
    """
    try:
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
        # Optionally load optimizer state
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        # Optionally load scheduler state
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        logger.info(f"Loaded checkpoint from {path}")
        
        return {
            'config': checkpoint.get('config'),
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0)
        }
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 