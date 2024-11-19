"""Ultra-optimized training steps with maximum GPU acceleration."""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from src.training.metrics import MetricsManager
from src.training.loss_functions import forward_pass
import weakref
from functools import lru_cache
from src.models.StateTracker import StateTracker

logger = logging.getLogger(__name__)


# Pre-allocated buffers for training steps
_buffers: Dict[str, torch.Tensor] = {}


class GradientAccumulator:
    """Efficient gradient accumulation with weak references."""
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self._step = 0
        self._stored_grads = weakref.WeakKeyDictionary()
    
    def _accumulate_grad(self, grad: torch.Tensor, stored: Optional[torch.Tensor]) -> torch.Tensor:
        """Optimized gradient accumulation."""
        if stored is None:
            return grad.clone()
        return stored + grad

    def store_grad(self, param: torch.nn.Parameter, grad: torch.Tensor) -> None:
        """Store gradient with weak reference."""
        if param not in self._stored_grads:
            self._stored_grads[param] = grad.clone()
        else:
            self._stored_grads[param] = self._accumulate_grad(grad, self._stored_grads[param])
    
    def apply_accumulated_grads(self) -> bool:
        """Apply accumulated gradients efficiently."""
        self._step += 1
        if self._step >= self.accumulation_steps:
            self._step = 0
            for param, grad in self._stored_grads.items():
                if param.grad is None:
                    param.grad = grad / self.accumulation_steps
                else:
                    param.grad.copy_(grad / self.accumulation_steps)
            self._stored_grads.clear()
            return True
        return False

def update_ema(ema_param: torch.Tensor, model_param: torch.Tensor, decay: float) -> None:
    """Optimized EMA update."""
    ema_param.copy_(ema_param * decay + model_param * (1 - decay))

def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999
) -> None:
    """Update EMA model parameters efficiently."""
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            if model_param.requires_grad:
                update_ema(ema_param.data, model_param.data, decay)

@autocast('cuda')
def train_step(
    args: Any,
    model_dict: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    grad_accumulator: Optional[GradientAccumulator] = None,
    scaler: Optional[GradScaler] = None,
    state_tracker: Optional[StateTracker] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute optimized training step with enhanced metrics."""
    try:
        # Forward pass with detailed metrics
        loss, forward_metrics = forward_pass(
            args=args,
            model_dict=model_dict,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        
        # Scale loss for gradient accumulation
        if grad_accumulator is not None:
            loss = loss / grad_accumulator.accumulation_steps
        
        # Backward pass with gradient tracking
        grad_metrics = {}
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Track gradient norms before clipping
            if hasattr(args, 'log_gradients') and args.log_gradients:
                for name, param in model_dict["unet"].named_parameters():
                    if param.grad is not None:
                        grad_metrics[f"grad_norm/{name}"] = param.grad.norm().item()
                        grad_metrics[f"weight_norm/{name}"] = param.norm().item()
            
            # Unscale and clip gradients
            if args.max_grad_norm is not None:
                scaler.unscale_(optimizers["unet"])
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_dict["unet"].parameters(),
                    args.max_grad_norm
                )
                grad_metrics["grad_norm/total"] = grad_norm.item()
        
        # Gather comprehensive metrics
        metrics_dict = {
            "loss/total": loss.item(),
            "loss/v_pred": forward_metrics.get("v_pred_loss", 0.0),
            "learning_rate": schedulers["unet"].get_last_lr()[0],
            "batch_size": batch["pixel_values"].shape[0],
            **grad_metrics,
            **forward_metrics
        }
        
        # Update state tracker
        if state_tracker is not None:
            state_tracker.update_metrics(metrics_dict)
        
        return loss, metrics_dict
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        raise

def clear_caches() -> None:
    """Clear all training step caches and buffers."""
    _buffers.clear()
