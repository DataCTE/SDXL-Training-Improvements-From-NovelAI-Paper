"""Ultra-optimized loss functions implementing NAI's SDXL improvements."""

import torch
import torch.nn.functional as F
from torch.amp import autocast
import logging
from typing import Dict, Any, Optional, Tuple
import math

logger = logging.getLogger(__name__)

# Pre-allocated buffers
_buffers: Dict[str, torch.Tensor] = {}
_sigma_cache: Dict[str, torch.Tensor] = {}

@torch.jit.script
def _compute_v_target(noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute v-prediction target per NAI paper section 2.1.
    Transitions from ε-prediction to x0-prediction as SNR changes.
    Args:
        noise: Noise tensor [B, C, H, W]
        sigma: Noise levels [B]
    Returns:
        v-target tensor [B, C, H, W]
    """
    sigma_data = 1.0  # Fixed data variance
    return noise * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5

@torch.jit.script
def _compute_snr(sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio (SNR).
    Args:
        sigma: Noise levels [B]
    Returns:
        SNR tensor [B]
    """
    sigma_data = 1.0
    return (sigma_data ** 2) / (sigma ** 2)

@torch.jit.script
def _compute_min_snr_weights(sigma: torch.Tensor, min_snr_gamma: float, scale: float) -> torch.Tensor:
    """
    Compute MinSNR loss weights per NAI paper section 2.4.
    Args:
        sigma: Noise levels [B]
        min_snr_gamma: SNR clamping value (default 5.0)
        scale: Optional scaling factor
    Returns:
        Loss weights tensor [B, 1, 1, 1]
    """
    snr = _compute_snr(sigma)
    weights = torch.minimum(snr, torch.tensor(min_snr_gamma, device=sigma.device))
    weights = weights / snr
    weights = weights * scale
    return weights.view(-1, 1, 1, 1)

@torch.jit.script
def _compute_resolution_scale(height: int, width: int) -> float:
    """
    Compute resolution-based sigma scaling per NAI paper section 2.3.
    Args:
        height: Image height
        width: Image width
    Returns:
        Resolution scale factor
    """
    base_res = 1024.0
    return math.sqrt((float(height) * float(width)) / (base_res * base_res))

def _get_resolution_scaled_sigma_schedule(
    num_steps: int,
    height: int,
    width: int,
    sigma_min: float = 0.0292,
    rho: float = 7.0,
) -> torch.Tensor:
    """
    Compute sigma schedule with NAI's resolution scaling and ZTSNR.
    Args:
        num_steps: Number of diffusion steps
        height: Image height
        width: Image width
        sigma_min: Minimum noise level
        rho: Karras schedule parameter
    Returns:
        Tensor of sigma values [num_steps]
    """
    # Resolution-based scaling (NAI section 2.3)
    scale = _compute_resolution_scale(height, width)
    scaled_sigma_min = sigma_min * math.sqrt(scale)
    
    # Use fixed ZTSNR value (NAI appendix A.2)
    sigma_max = 20000.0
    
    # Generate timesteps
    steps = torch.linspace(0, num_steps - 1, num_steps, dtype=torch.float32)
    sigmas = torch.exp(-steps / rho)
    sigmas = sigmas * (sigma_max - scaled_sigma_min) + scaled_sigma_min
    
    return sigmas

@autocast('cuda')
def get_sigmas(
    num_inference_steps: int,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get optimized sigma schedule with NAI improvements."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_key = f"sigmas_{num_inference_steps}_{height}_{width}"
    if cache_key not in _sigma_cache:
        _sigma_cache[cache_key] = _get_resolution_scaled_sigma_schedule(
            num_inference_steps,
            height,
            width
        )
    
    return _sigma_cache[cache_key].to(device)

@autocast('cuda')
def training_loss_v_prediction(
    model: torch.nn.Module,
    x_0: torch.Tensor,  # [B, C, H, W]
    sigma: torch.Tensor,  # [B]
    text_embeddings: torch.Tensor,  # [B, 77, D]
    noise: torch.Tensor,  # [B, C, H, W]
    added_cond_kwargs: Dict[str, torch.Tensor],  # {"text_embeds": [B, 1, D], "time_ids": [B, 6]}
    min_snr_gamma: float = 5.0,
    scale_factor: float = 1.0,
    return_metrics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute v-prediction loss with NAI improvements.
    Args:
        model: UNet model
        x_0: Input images
        sigma: Noise levels
        text_embeddings: Text condition embeddings
        noise: Gaussian noise
        added_cond_kwargs: Additional conditioning (text_embeds, time_ids)
        min_snr_gamma: SNR clamping value
        scale_factor: Loss scaling factor
        return_metrics: Whether to return additional metrics
    Returns:
        loss: Training loss
        metrics: Optional dict of metrics if return_metrics=True
    """
    # Scale sigma based on resolution [B]
    height, width = x_0.shape[-2:]
    sigma = sigma * _compute_resolution_scale(height, width)
    
    # Add noise to input [B, C, H, W]
    noised = x_0 + sigma.view(-1, 1, 1, 1) * noise
    
    # Forward pass [B, C, H, W]
    v_pred = model(
        noised,
        sigma,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs
    ).sample
    
    # Compute v-target [B, C, H, W]
    v_target = _compute_v_target(noise, sigma)
    
    # Compute MinSNR weights [B, 1, 1, 1]
    weights = _compute_min_snr_weights(sigma, min_snr_gamma, scale_factor)
    
    # Compute weighted MSE loss
    loss = F.mse_loss(
        v_pred * weights.sqrt(),
        v_target * weights.sqrt(),
        reduction='none'
    ).mean()

    if not return_metrics:
        return loss, {}
        
    # Compute metrics
    with torch.no_grad():
        metrics = {
            "target_mean": v_target.mean().item(),
            "target_std": v_target.std().item(),
            "pred_mean": v_pred.mean().item(),
            "pred_std": v_pred.std().item(),
            "error": torch.abs(v_pred - v_target).mean().item(),
            "snr": _compute_snr(sigma).mean().item(),
            "weight_mean": weights.mean().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_std": sigma.std().item(),
            "v_pred_loss": loss.item(),
        }
    
    return loss, metrics

def clear_caches() -> None:
    """Clear all caches and buffers."""
    _buffers.clear()
    _sigma_cache.clear()