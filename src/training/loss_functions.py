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

# Pre-allocated buffers for common operations - properly typed and initialized
_buffers: Dict[str, torch.Tensor] = {}
_sigma_cache: Dict[str, torch.Tensor] = {}
_scheduler_buffers: Dict[str, torch.Tensor] = {}

def _get_or_create_buffer(key: str, shape: Union[tuple, Tuple], device: torch.device) -> torch.Tensor:
    """Get or create a pre-allocated buffer."""
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

@torch.jit.script
def _compute_resolution_scale(height: int, width: int, base_resolution: float = 1024.0) -> float:
    """JIT-optimized resolution scale computation."""
    return math.sqrt((float(height) * float(width)) / (base_resolution * base_resolution))

@lru_cache(maxsize=32)
def _get_resolution_scaled_sigma_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    height: int,
    width: int,
    rho: float = 7.0,  # Configurable rho parameter
    min_snr_gamma: float = 5.0,
) -> torch.Tensor:
    """Cached computation of resolution-scaled sigma schedule with NovelAI improvements."""
    # Compute resolution scale
    scale = _compute_resolution_scale(height, width)
    
    # Apply NovelAI's improved sigma scaling
    scaled_sigma_max = sigma_max * scale
    scaled_sigma_min = sigma_min * math.sqrt(scale)  # Square root scaling for min sigma
    
    # Generate timesteps
    steps = torch.arange(num_steps, dtype=torch.float32)
    inv_rho = 1.0 / rho
    
    # Compute sigmas with improved scaling
    sigmas = torch.exp(-steps * inv_rho) * (scaled_sigma_max - scaled_sigma_min) + scaled_sigma_min
    
    # Apply SNR clamping
    snr = _compute_snr(sigmas, 1.0)
    sigmas = torch.where(
        snr > min_snr_gamma,
        torch.sqrt(torch.tensor(1.0 + min_snr_gamma) / snr) * sigmas,
        sigmas
    )
    
    return sigmas

@autocast('cuda')
def get_sigmas(
    num_inference_steps: int = 28,
    sigma_min: float = 0.0292,
    sigma_max: float = 14.614,  # SDXL default
    height: int = 1024,
    width: int = 1024,
    device: Optional[torch.device] = None,
    min_snr_gamma: float = 5.0,
) -> torch.Tensor:
    """Get optimized sigma schedule with SDXL defaults and NovelAI improvements."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use cached schedule with resolution scaling
    cache_key = f"sigmas_{num_inference_steps}_{sigma_min}_{sigma_max}_{height}_{width}_{min_snr_gamma}"
    if cache_key not in _sigma_cache:
        _sigma_cache[cache_key] = _get_resolution_scaled_sigma_schedule(
            num_inference_steps,
            sigma_min,
            sigma_max,
            height,
            width,
            min_snr_gamma=min_snr_gamma
        )
    
    return _sigma_cache[cache_key].to(device)

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
    return_metrics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute v-prediction loss with SDXL and NovelAI optimizations."""
    batch_size = x_0.shape[0]
    device = x_0.device
    height, width = x_0.shape[-2:]
    
    # Generate noise efficiently
    if noise is None:
        noise_buffer = _get_or_create_buffer("noise", x_0.shape, device)
        noise_buffer.normal_()
        noise = noise_buffer
    
    # Optimize timesteps computation
    timesteps_buffer = _get_or_create_buffer("timesteps", (batch_size,), device)
    timesteps_buffer.copy_(sigma)
    timesteps_buffer = timesteps_buffer.squeeze()
    
    # Compute noisy samples efficiently
    noisy_samples = x_0 + noise * sigma.view(-1, 1, 1, 1)
    
    # Generate time embeddings for SDXL
    if added_cond_kwargs is None:
        added_cond_kwargs = {}
    
    if "time_ids" not in added_cond_kwargs:
        time_ids = _get_add_time_ids(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=text_embeddings.dtype,
            device=device
        )
        added_cond_kwargs["time_ids"] = time_ids
    
    # Ensure text_embeds has correct shape [batch_size, dim]
    if "text_embeds" in added_cond_kwargs:
        text_embeds = added_cond_kwargs["text_embeds"]
        if len(text_embeds.shape) == 3:  # If shape is [batch_size, 1, dim]
            added_cond_kwargs["text_embeds"] = text_embeds.squeeze(1)
    
    # Forward pass with optimization
    v_pred = apply_model(model, noisy_samples, timesteps_buffer, text_embeddings, added_cond_kwargs)
    
    # Compute target with improved SNR weighting
    v_target = _compute_v_target(noise, sigma)
    
    # Compute weights with NovelAI's improvements
    if use_snr_weighting:
        snr = _compute_snr(sigma, 1.0)
        # Apply min SNR gamma clamping
        snr = torch.clamp(snr, max=min_snr_gamma)
        # Compute weights with scale factor
        weights = _compute_loss_weights(snr, min_snr_gamma, scale_factor)
        weights = weights.view(-1, 1, 1, 1)
    else:
        weights = torch.ones_like(sigma.view(-1, 1, 1, 1))
    
    # Compute loss efficiently
    loss = torch.sum((v_pred - v_target) ** 2 * weights) / batch_size
    
    if return_metrics:
        return loss, {
            "target_mean": v_target.mean().item(),
            "target_std": v_target.std().item(),
            "pred_mean": v_pred.mean().item(),
            "pred_std": v_pred.std().item(),
            "error": torch.abs(v_pred - v_target).mean().item(),
            "snr": snr.mean().item() if use_snr_weighting else 0.0,
            "weight_mean": weights.mean().item(),
        }
    
    return loss

@torch.jit.script
def _compute_warmup_factor(current_step: int, num_warmup_steps: int) -> float:
    """JIT-optimized warmup factor computation."""
    return float(current_step) / float(max(1, num_warmup_steps))

@torch.jit.script
def _compute_cosine_decay(progress: float, num_cycles: float) -> float:
    """JIT-optimized cosine decay computation."""
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

@lru_cache(maxsize=16)
def _get_scheduler_buffer(num_steps: int, device: torch.device) -> torch.Tensor:
    """Cached computation of progress tensor for scheduler."""
    return torch.arange(num_steps, dtype=torch.float32, device=device) / max(1, num_steps)

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    device: Optional[torch.device] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create an ultra-optimized cosine learning rate schedule with warmup."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get pre-computed progress tensor from buffer
    progress = _get_scheduler_buffer(num_training_steps, device)
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Use pre-computed warmup factors if available
            if "warmup_factors" in _scheduler_buffers:
                return _scheduler_buffers["warmup_factors"][current_step].item()
            return _compute_warmup_factor(current_step, num_warmup_steps)
            
        # Use pre-computed progress values
        step_progress = progress[current_step].item()
        return _compute_cosine_decay(step_progress, num_cycles)
    
    # Pre-compute warmup factors if not already cached
    if "warmup_factors" not in _scheduler_buffers:
        _scheduler_buffers["warmup_factors"] = torch.tensor(
            [_compute_warmup_factor(i, num_warmup_steps) for i in range(num_warmup_steps)],
            device=device
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

@autocast('cuda')
def forward_pass(
    args: Any,
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute forward pass with detailed metrics."""
    # Get models
    unet = model_dict["unet"]
    text_encoder = model_dict.get("text_encoder")
    text_encoder_2 = model_dict.get("text_encoder_2")
    
    # Validate batch contents
    required_keys = ["pixel_values"]
    missing_keys = [key for key in required_keys if key not in batch]
    if missing_keys:
        raise ValueError(f"Missing required keys in batch: {missing_keys}")
        
    # Move batch to device efficiently
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
    
    # Handle text embeddings if present in batch
    if "text_embeddings" in batch and "pooled_text_embeddings" in batch:
        # Use pre-computed embeddings
        text_embeddings = batch["text_embeddings"].to(device=device, non_blocking=True)
        pooled_text_embeddings = batch["pooled_text_embeddings"].to(device=device, non_blocking=True)
    else:
        # Check for input IDs
        if "input_ids" not in batch:
            raise KeyError("Batch must contain either 'text_embeddings' or 'input_ids'")
            
        # Get input IDs
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
    
    # Prepare added conditions with time_ids
    batch_size = pixel_values.shape[0]
    height, width = pixel_values.shape[-2:]
    
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeddings,
        "time_ids": _get_add_time_ids(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device
        )
    }
    
    # Get sigmas efficiently
    sigmas = get_sigmas(
        height=height,
        width=width,
        device=device
    )
    sigma = sigmas[0]
    
    # Compute loss with detailed metrics
    loss, v_pred_metrics = training_loss_v_prediction(
        model=unet,
        x_0=pixel_values,
        sigma=sigma,
        text_embeddings=text_embeddings,
        use_snr_weighting=getattr(args, "use_snr_weighting", True),
        min_snr_gamma=getattr(args, "min_snr_gamma", 5.0),
        scale_factor=getattr(args, "scale_factor", 1.0),
        added_cond_kwargs=added_cond_kwargs,
        return_metrics=True
    )
    
    # Gather forward pass metrics
    metrics = {
        "v_pred/target_mean": v_pred_metrics["target_mean"],
        "v_pred/target_std": v_pred_metrics["target_std"],
        "v_pred/pred_mean": v_pred_metrics["pred_mean"],
        "v_pred/pred_std": v_pred_metrics["pred_std"],
        "v_pred/error": v_pred_metrics["error"],
        "sigma/value": sigma.mean().item(),
        "embeddings/norm": text_embeddings.norm().item(),
    }
    
    return loss, metrics

def clear_caches() -> None:
    """Clear all caches and buffers."""
    _buffers.clear()
    _sigma_cache.clear()
    _scheduler_buffers.clear()
    _get_scheduler_buffer.cache_clear()
    _get_sigma_schedule.cache_clear()
    _get_resolution_scaled_sigma_schedule.cache_clear()  # Added for completeness

def _get_add_time_ids(
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Generate time_ids tensor for SDXL conditioning following the official implementation."""
    if original_size is None:
        original_size = (height, width)
    if target_size is None:
        target_size = (height, width)

    # SDXL base expects 6 time_ids:
    # 1-2. Original image size (h, w)
    # 3-4. Crop coordinates (top, left)
    # 5-6. Target size (h, w)
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
    
    # Create tensor with correct shape [batch_size, 6]
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)

    return add_time_ids