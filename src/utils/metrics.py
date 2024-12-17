import torch
import logging
import wandb
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_grad_norm(model: torch.nn.Module, grad_norm_buffer: torch.Tensor) -> float:
    """Compute gradient norm using pre-allocated buffer."""
    grad_norm_buffer.zero_()
    
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            grad_norm_buffer[i] = p.grad.data.norm(2).item()
    
    return torch.sqrt(torch.sum(grad_norm_buffer * grad_norm_buffer)).item()

def log_metrics(metrics: Dict[str, Any], global_step: int, is_main_process: bool, 
                use_wandb: bool, step_type: str = "step"):
    """Unified logging function for all metrics."""
    try:
        if not is_main_process:
            return
        
        if use_wandb:
            wandb.log(metrics, step=global_step)
        
        if step_type == "step":
            logger.info(
                f"Step {global_step} | "
                f"Loss: {metrics.get('train/loss', 0):.4f} | "
                f"LR: {metrics.get('train/learning_rate', 0):.2e}"
            )
        elif step_type == "epoch":
            logger.info(
                f"Epoch {metrics.get('epoch', 0)} | "
                f"Avg Loss: {metrics.get('train/epoch_loss', 0):.4f}"
            )
        
    except Exception as e:
        logger.warning(f"Error logging metrics: {str(e)}") 