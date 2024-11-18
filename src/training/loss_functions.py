"""Ultra-optimized loss functions with maximum GPU acceleration."""

import torch
import torch.nn.functional as F
from torch.amp import autocast
import logging
from typing import Dict, Any, Optional, Tuple, Union
import weakref
from functools import lru_cache
import math

logger = logging.getLogger(__name__)

# Pre-allocated buffers for common operations
_buffers = {}
_sigma_cache = {}

def _get_or_create_buffer(key: str, shape: tuple, device: torch.device) -> torch.Tensor:
    """Get or create a pre-allocated buffer."""
    global _buffers
    buffer_key = f"{key}_{shape}_{device}"
    if buffer_key not in _buffers:
        _buffers[buffer_key] = torch.empty(shape, device=device)
    return _buffers[buffer_key]

@torch.jit.script
def _compute_v_target(noise: torch.Tensor, sigma: torch.Tensor, sigma_data: float = 1.0) -> torch.Tensor:
    """JIT-optimized v-prediction target computation."""
    return noise * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5

@torch.jit.script
def _compute_snr(timesteps: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """JIT-optimized SNR computation."""
    return (sigma_data ** 2) / (timesteps ** 2)

@torch.jit.script
def _compute_loss_weights(snr: torch.Tensor, min_snr_gamma: float, scale: float) -> torch.Tensor:
    """JIT-optimized loss weight computation."""
    return torch.minimum(snr, torch.tensor(min_snr_gamma)).div(snr).mul(scale)

@lru_cache(maxsize=32)
def _get_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Cached computation of sigma schedule."""
    rho = 7.0  # Hardcoded for optimization
    inv_rho = 1.0 / rho
    steps = torch.arange(num_steps, dtype=torch.float32)
    sigmas = torch.exp(-steps * inv_rho) * (sigma_max - sigma_min) + sigma_min
    return sigmas

@autocast('cuda')
def get_sigmas(
    num_inference_steps: int = 28,
    sigma_min: float = 0.0292,
    height: int = 1024,
    width: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get optimized sigma schedule."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use cached schedule
    sigmas = _get_sigma_schedule(num_inference_steps, sigma_min, 14.614)
    return sigmas.to(device)

def apply_model(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, text_embeddings: torch.Tensor, added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Optimized model application."""
    if added_cond_kwargs is None:
        added_cond_kwargs = {}
    return model(x, t, text_embeddings, added_cond_kwargs=added_cond_kwargs).sample

@autocast('cuda')
def training_loss_v_prediction(
    model: torch.nn.Module,
    x_0: torch.Tensor,
    sigma: torch.Tensor,
    text_embeddings: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    use_snr_weighting: bool = True,
    min_snr_gamma: float = 5.0,
    scale_factor: float = 1.0,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute v-prediction loss with maximum optimization."""
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Reuse noise tensor if possible
    if noise is None:
        noise_buffer = _get_or_create_buffer("noise", x_0.shape, device)
        torch.randn_like(x_0, out=noise_buffer)
        noise = noise_buffer
    
    # Optimize timesteps computation
    timesteps_buffer = _get_or_create_buffer("timesteps", (batch_size,), device)
    timesteps_buffer.copy_(sigma)
    
    # Compute noisy samples efficiently
    noisy_samples = x_0 + noise * sigma.view(-1, 1, 1, 1)
    
    # Forward pass with optimization
    v_pred = apply_model(model, noisy_samples, timesteps_buffer, text_embeddings, added_cond_kwargs)
    
    # Compute target
    v_target = _compute_v_target(noise, sigma)
    
    # Compute weights if needed
    if use_snr_weighting:
        snr = _compute_snr(sigma, 1.0)
        weights = _compute_loss_weights(snr, min_snr_gamma, scale_factor)
        weights = weights.view(-1, 1, 1, 1)
    else:
        weights = torch.ones_like(sigma.view(-1, 1, 1, 1))
    
    # Compute loss efficiently
    loss = torch.sum((v_pred - v_target) ** 2 * weights) / batch_size
    
    return loss

@autocast('cuda')
def forward_pass(
    args: Any,
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    components: Dict[str, Any],
) -> torch.Tensor:
    """Execute forward pass with maximum GPU optimization."""
    try:
        # Get models
        unet = model_dict["unet"]
        text_encoder = model_dict.get("text_encoder")
        text_encoder_2 = model_dict.get("text_encoder_2")
        
        # Move batch to device efficiently
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        input_ids_2 = batch.get("input_ids_2", input_ids).to(device=device, non_blocking=True)
        
        # Compute text embeddings with caching
        cache_key = f"{input_ids.shape}_{input_ids_2.shape}"
        if cache_key not in _buffers:
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]
                text_embeddings_2 = text_encoder_2(input_ids_2)[0]
                pooled_text_embeddings = text_embeddings_2[:, -1]
                text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
            _buffers[cache_key] = (text_embeddings, pooled_text_embeddings)
        else:
            text_embeddings, pooled_text_embeddings = _buffers[cache_key]
        
        # Prepare added conditions
        added_cond_kwargs = {"text_embeds": pooled_text_embeddings}
        
        # Get sigmas efficiently
        height, width = pixel_values.shape[-2:]
        sigmas = get_sigmas(
            height=height,
            width=width,
            device=device
        )
        sigma = sigmas[0]
        
        # Compute loss with all optimizations
        loss = training_loss_v_prediction(
            model=unet,
            x_0=pixel_values,
            sigma=sigma,
            text_embeddings=text_embeddings,
            use_snr_weighting=getattr(args, "use_snr_weighting", True),
            min_snr_gamma=getattr(args, "min_snr_gamma", 5.0),
            scale_factor=getattr(args, "scale_factor", 1.0),
            added_cond_kwargs=added_cond_kwargs,
        )
        
        return loss
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise

def clear_caches() -> None:
    """Clear all caches and buffers."""
    global _buffers, _sigma_cache
    _buffers.clear()
    _sigma_cache.clear()
    _get_sigma_schedule.cache_clear()