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

logger = logging.getLogger(__name__)

# Pre-allocated buffers for common operations
_buffers = {}

def _get_or_create_buffer(key: str, shape: tuple, device: torch.device) -> torch.Tensor:
    """Get or create a pre-allocated buffer."""
    global _buffers
    buffer_key = f"{key}_{shape}_{device}"
    if buffer_key not in _buffers:
        _buffers[buffer_key] = torch.empty(shape, device=device)
    return _buffers[buffer_key]

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
    batch_idx: int,
    metrics: MetricsManager,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    grad_accumulator: Optional[GradientAccumulator] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute optimized training step with GPU acceleration."""
    try:
        # Determine precision context
        if args.mixed_precision == "fp16":
            dtype = torch.float16
        elif args.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Forward pass with mixed precision
        loss = forward_pass(
            args=args,
            model_dict=model_dict,
            batch=batch,
            device=device,
            dtype=dtype,
            components={"metrics": metrics}
        )
        
        # Scale loss for gradient accumulation if needed
        if grad_accumulator is not None:
            loss = loss / grad_accumulator.accumulation_steps
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip if needed
            if args.max_grad_norm is not None:
                scaler.unscale_(optimizers["unet"])
                torch.nn.utils.clip_grad_norm_(
                    model_dict["unet"].parameters(),
                    args.max_grad_norm
                )
            
            # Step optimizer with gradient scaling
            if grad_accumulator is None or grad_accumulator.apply_accumulated_grads():
                scaler.step(optimizers["unet"])
                scaler.update()
                optimizers["unet"].zero_grad(set_to_none=True)
        else:
            loss.backward()
            
            # Clip gradients if needed
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model_dict["unet"].parameters(),
                    args.max_grad_norm
                )
            
            # Step optimizer
            if grad_accumulator is None or grad_accumulator.apply_accumulated_grads():
                optimizers["unet"].step()
                optimizers["unet"].zero_grad(set_to_none=True)
        
        # Update EMA model if present
        if "ema_unet" in model_dict:
            update_ema_model(
                model_dict["ema_unet"],
                model_dict["unet"],
                getattr(args, "ema_decay", 0.9999)
            )
        
        # Update learning rate
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Gather metrics
        metrics_dict = {
            "loss": loss.item(),
            "lr": schedulers["unet"].get_last_lr()[0]
        }
        
        return loss, metrics_dict
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        raise

def clear_caches() -> None:
    """Clear all caches and buffers."""
    global _buffers
    _buffers.clear()
