import torch
import logging
import wandb
import traceback
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_grad_norm(model: torch.nn.Module, grad_norm_buffer: torch.Tensor) -> float:
    """Compute gradient norm using pre-allocated buffer."""
    try:
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected model to be torch.nn.Module, got {type(model)}")
        if not isinstance(grad_norm_buffer, torch.Tensor):
            raise ValueError(f"Expected grad_norm_buffer to be torch.Tensor, got {type(grad_norm_buffer)}")
            
        grad_norm_buffer.zero_()
        
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                try:
                    grad_norm_buffer[i] = p.grad.data.norm(2).item()
                except IndexError as e:
                    logger.error(f"Buffer size mismatch at index {i}")
                    raise
                except RuntimeError as e:
                    logger.error(f"Error computing gradient norm for parameter {i}: {str(e)}")
                    raise
        
        return torch.sqrt(torch.sum(grad_norm_buffer * grad_norm_buffer)).item()
    
    except Exception as e:
        logger.error(f"Error in compute_grad_norm: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def log_metrics(metrics: Dict[str, Any], global_step: int, is_main_process: bool, 
                use_wandb: bool, step_type: str = "step"):
    """Unified logging function for all metrics."""
    try:
        if not isinstance(metrics, dict):
            raise ValueError(f"Expected metrics to be dict, got {type(metrics)}")
        if not isinstance(global_step, int):
            raise ValueError(f"Expected global_step to be int, got {type(global_step)}")
        
        if not is_main_process:
            return
        
        if use_wandb:
            try:
                wandb.log(metrics, step=global_step)
            except Exception as e:
                logger.error(f"Error logging to wandb: {str(e)}")
                logger.error(traceback.format_exc())
        
        try:
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
            else:
                logger.warning(f"Unknown step_type: {step_type}")
        except Exception as e:
            logger.error(f"Error formatting log message: {str(e)}")
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Error in log_metrics: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}") 