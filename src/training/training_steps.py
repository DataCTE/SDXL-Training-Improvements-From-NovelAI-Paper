"""Ultra-optimized training steps implementing NAI's SDXL improvements."""

import torch
import logging
from typing import Dict, Any, Tuple
from torch.amp import autocast, GradScaler
from src.training.loss_functions import forward_pass
from src.models.StateTracker import StateTracker

logger = logging.getLogger(__name__)

@autocast('cuda')
def train_step(
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, Any],
    scaler: GradScaler,
    state_tracker: StateTracker,
    num_inference_steps: int = 1000,  # NAI's full schedule length
    min_snr_gamma: float = 5.0,
    scale_factor: float = 1.0,
    max_grad_norm: float = 1.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Execute training step with NAI improvements:
    1. v-prediction parameterization (Section 2.1)
    2. ZTSNR noise schedule (Section 2.2)
    3. Resolution-based sigma scaling (Section 2.3)
    4. MinSNR loss weighting (Section 2.4)
    
    Args:
        model_dict: Dictionary containing UNet model
        batch: Training batch with pixel_values, embeddings and time_ids
        optimizers: Dictionary of optimizers
        schedulers: Dictionary of learning rate schedulers
        scaler: Gradient scaler for mixed precision
        state_tracker: Training state tracker
        num_inference_steps: Number of inference steps (NAI uses 1000)
        min_snr_gamma: SNR clamping value
        scale_factor: Loss scaling factor
        max_grad_norm: Maximum gradient norm
        device: Compute device
        dtype: Compute dtype
    Returns:
        loss: Training loss
        metrics: Training metrics
    """
    # Clear gradients
    optimizers["unet"].zero_grad(set_to_none=True)
    
    # Forward pass with NAI improvements
    loss, forward_metrics = forward_pass(
        model_dict=model_dict,
        batch=batch,
        num_inference_steps=num_inference_steps,
        min_snr_gamma=min_snr_gamma,
        scale_factor=scale_factor,
        device=device,
        dtype=dtype
    )
    
    # Scale loss and compute gradients
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Unscale gradients for clipping
    scaler.unscale_(optimizers["unet"])
    
    # Compute gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model_dict["unet"].parameters(),
        max_grad_norm
    )
    
    # Update weights with gradient clipping
    scaler.step(optimizers["unet"])
    scaler.update()
    
    # Update learning rate
    schedulers["unet"].step()
    
    # Collect metrics
    metrics = {
        "loss/total": loss.item(),
        "grad_norm/total": grad_norm.item(),
        "grad_norm/clipped": min(grad_norm.item(), max_grad_norm),
        "learning_rate": schedulers["unet"].get_last_lr()[0],
        "batch_size": batch["pixel_values"].shape[0],
        **forward_metrics
    }
    
    # Update state tracker
    if state_tracker is not None:
        height, width = batch["pixel_values"].shape[-2:]
        state_tracker.update_metrics({
            **metrics,
            "sigma_max": 20000.0,  # NAI's practical infinity
            "effective_batch_size": batch["pixel_values"].shape[0],
            "resolution_scale": (height * width) / (1024 * 1024)  # Relative to base res
        })
    
    return loss, metrics

def clear_caches() -> None:
    """Clear all training step caches and buffers."""
    pass  # No caches in this implementation
