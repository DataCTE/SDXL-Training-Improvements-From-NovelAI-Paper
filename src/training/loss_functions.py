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
    """
    sigma_data = 1.0  # Fixed data variance per NAI
    return noise * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5

@torch.jit.script
def _compute_snr(sigma: torch.Tensor) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio (SNR)."""
    sigma_data = 1.0  # Fixed data variance per NAI
    return (sigma_data ** 2) / (sigma ** 2)

@torch.jit.script
def _compute_min_snr_weights(sigma: torch.Tensor, min_snr_gamma: float, scale: float) -> torch.Tensor:
    """
    Compute MinSNR loss weights per NAI paper section 2.4.
    Balances learning across timesteps.
    """
    snr = _compute_snr(sigma)
    weights = torch.minimum(snr, torch.tensor(min_snr_gamma, device=sigma.device))
    weights = weights / snr  # Normalize by SNR
    weights = weights * scale
    return weights.view(-1, 1, 1, 1)

@torch.jit.script
def _compute_resolution_scale(height: int, width: int) -> float:
    """
    Compute resolution-based sigma scaling per NAI paper section 2.3.
    Doubles sigma_max when canvas area quadruples.
    """
    base_res = 1024.0  # SDXL base resolution
    return math.sqrt((float(height) * float(width)) / (base_res * base_res))

def get_cosine_schedule_with_warmup(
    num_training_steps: int,
    num_warmup_steps: int,
    height: int,
    width: int,
    sigma_min: float = 0.0292,
    sigma_max: float = 20000.0,  # NAI Appendix A.2: practical ZTSNR approximation
    rho: float = 7.0,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Get ZTSNR noise schedule with cosine warmup per NAI paper.
    Args:
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        height: Image height
        width: Image width
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level (NAI uses 20000.0 as ∞ approximation)
        rho: Karras schedule parameter
        device: Compute device
    Returns:
        Tensor of sigma values [num_training_steps]
    """
    # Resolution-based scaling (NAI section 2.3)
    scale = _compute_resolution_scale(height, width)
    scaled_sigma_min = sigma_min * math.sqrt(scale)
    
    # Generate timesteps
    steps = torch.linspace(0, num_training_steps - 1, num_training_steps, device=device)
    
    # Compute warmup schedule
    warmup_steps = torch.arange(num_warmup_steps, device=device)
    warmup_factor = warmup_steps / max(1, num_warmup_steps)
    
    # Generate Karras schedule (NAI section 2.2)
    sigmas = torch.exp(-steps / rho)
    sigmas = sigmas * (sigma_max - scaled_sigma_min) + scaled_sigma_min
    
    # Apply warmup
    if num_warmup_steps > 0:
        sigmas[:num_warmup_steps] *= warmup_factor
    
    return sigmas

@autocast('cuda')
def get_sigmas(
    num_inference_steps: int,
    height: int,
    width: int,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Get ZTSNR schedule with resolution scaling."""
    cache_key = f"sigmas_{num_inference_steps}_{height}_{width}"
    
    if cache_key not in _sigma_cache:
        _sigma_cache[cache_key] = get_cosine_schedule_with_warmup(
            num_training_steps=num_inference_steps,
            num_warmup_steps=0,  # No warmup for inference
            height=height,
            width=width,
            sigma_min=0.0292,
            sigma_max=20000.0,  # NAI ZTSNR approximation
            rho=7.0,
            device=torch.device("cpu")  # Cache on CPU
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
    Compute v-prediction loss with NAI improvements:
    1. v-prediction parameterization (Section 2.1)
    2. ZTSNR noise schedule (Section 2.2)
    3. Resolution-based sigma scaling (Section 2.3)
    4. MinSNR loss weighting (Section 2.4)
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

@autocast('cuda')
def forward_pass(
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    num_inference_steps: int,
    min_snr_gamma: float = 5.0,
    scale_factor: float = 1.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Forward pass implementing NAI's SDXL improvements.
    Args:
        model_dict: Dictionary containing UNet model
        batch: Training batch containing:
            - pixel_values: [B, C, H, W]
            - text_embeddings: [B, 77, D]
            - pooled_text_embeddings: [B, D]
            - time_ids: [B, 6]
        num_inference_steps: Number of inference steps
        min_snr_gamma: SNR clamping value
        scale_factor: Loss scaling factor
        device: Compute device
        dtype: Compute dtype
    Returns:
        loss: Training loss
        metrics: Training metrics
    """
    # Get models
    unet = model_dict["unet"]
    
    # Process inputs
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)  # [B, C, H, W]
    text_embeddings = batch["text_embeddings"].to(device=device)  # [B, 77, D]
    pooled_text_embeddings = batch["pooled_text_embeddings"].to(device=device)  # [B, D]
    time_ids = batch["time_ids"].to(device=device, dtype=dtype)  # [B, 6]
    
    # Get batch dimensions
    batch_size, _, height, width = pixel_values.shape
    
    # Format conditioning inputs
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeddings.unsqueeze(1),  # [B, 1, D]
        "time_ids": time_ids  # [B, 6]
    }
    
    # Get noise schedule
    sigmas = get_sigmas(
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        device=device
    )
    sigma = sigmas[0]  # Use highest noise level
    
    # Generate noise
    noise = torch.randn_like(pixel_values)  # [B, C, H, W]
    
    # Compute loss with NAI improvements
    loss, metrics = training_loss_v_prediction(
        model=unet,
        x_0=pixel_values,
        sigma=sigma.expand(batch_size),  # [B]
        text_embeddings=text_embeddings,
        noise=noise,
        added_cond_kwargs=added_cond_kwargs,
        min_snr_gamma=min_snr_gamma,
        scale_factor=scale_factor,
        return_metrics=True
    )
    
    return loss, metrics

def clear_caches() -> None:
    """Clear all caches and buffers."""
    _buffers.clear()
    _sigma_cache.clear()