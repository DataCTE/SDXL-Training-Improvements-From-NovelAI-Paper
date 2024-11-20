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
    """Compute sigma schedule with NAI3's resolution-based scaling."""
    # Resolution-based scaling (NAI3 improvement)
    scale = _compute_resolution_scale(height, width)
    
    # Apply NAI3's adaptive sigma scaling
    scaled_sigma_max = sigma_max * scale
    scaled_sigma_min = sigma_min * math.sqrt(scale)  # Square root scaling for min sigma
    
    # Generate timesteps with improved spacing
    steps = torch.linspace(0, num_steps - 1, num_steps, dtype=torch.float32)
    inv_rho = 1.0 / rho
    
    # Compute sigmas with NAI3's improvements
    sigmas = torch.exp(-steps * inv_rho)
    sigmas = sigmas * (scaled_sigma_max - scaled_sigma_min) + scaled_sigma_min
    
    # Apply dynamic SNR clamping (NAI3 improvement)
    snr = _compute_snr(sigmas, 1.0)
    sigmas = torch.where(
        snr > min_snr_gamma,
        torch.sqrt((1.0 + min_snr_gamma) / snr) * sigmas,
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
    """Compute v-prediction loss with SDXL and NovelAI improvements."""
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Generate noise if not provided
    if noise is None:
        noise_buffer = _get_or_create_buffer("noise", x_0.shape, device)
        noise_buffer.normal_()
        noise = noise_buffer
    
    # Compute timesteps with improved precision
    timesteps = sigma.squeeze()
    
    # Compute v-target with improved numerical stability
    v_target = _compute_v_target(noise, sigma)
    
    # Add noise to input
    noisy_samples = x_0 + noise * sigma.view(-1, 1, 1, 1)
    
    # Forward pass
    v_pred = apply_model(model, noisy_samples, timesteps, text_embeddings, added_cond_kwargs)
    
    # Compute SNR-based loss weights (NAI3 improvement)
    if use_snr_weighting:
        # Compute SNR with improved stability
        snr = _compute_snr(sigma, 1.0)
        
        # Apply NAI3's dynamic SNR clamping
        snr = torch.clamp(snr, max=min_snr_gamma)
        
        # Compute weights with scale factor (NAI3 improvement)
        weights = scale_factor * torch.minimum(snr, torch.tensor(min_snr_gamma, device=device)) / snr
        weights = weights.view(-1, 1, 1, 1)
    else:
        weights = torch.ones_like(sigma.view(-1, 1, 1, 1))
        snr = torch.zeros_like(sigma)  # For metrics
    
    # Compute loss with improved numerical stability
    loss = F.mse_loss(v_pred * weights.sqrt(), v_target * weights.sqrt(), reduction='sum') / batch_size

    if return_metrics:
        return loss, {
            "target_mean": v_target.mean().item(),
            "target_std": v_target.std().item(),
            "pred_mean": v_pred.mean().item(),
            "pred_std": v_pred.std().item(),
            "error": torch.abs(v_pred - v_target).mean().item(),
            "snr": snr.mean().item() if use_snr_weighting else 0.0,
            "weight_mean": weights.mean().item(),
            "weight_std": weights.std().item(),  # Added for monitoring
            "effective_loss": (weights * (v_pred - v_target) ** 2).mean().item(),  # Added for monitoring
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

def clear_caches() -> None:
    """Clear all caches and buffers."""
    _buffers.clear()
    _sigma_cache.clear()
    _scheduler_buffers.clear()
    _get_scheduler_buffer.cache_clear()
    _get_sigma_schedule.cache_clear()
    _get_resolution_scaled_sigma_schedule.cache_clear()  # Added for completeness

def apply_model(
    model: torch.nn.Module,
    x: torch.Tensor,          # Shape: [B, C, H, W]
    t: torch.Tensor,          # Shape: [B]
    text_embeddings: torch.Tensor,  # Shape: [B, S, D] where S=sequence length, D=embedding dim
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor:
    """
    From SDXL paper section 2.1: UNet receives concatenated text embeddings from 
    CLIP ViT-L (768-dim) and OpenCLIP ViT-bigG (1280-dim) for 2048 total dim.
    Additionally conditions on pooled embeddings and time_ids.
    """
    if added_cond_kwargs is None:
        added_cond_kwargs = {}
        
    # Per paper section 2.2: Additional micro-conditioning requires matching dims
    if "time_ids" in added_cond_kwargs:
        time_ids = added_cond_kwargs["time_ids"]  # Shape: [B, D_time]
        # Add sequence dimension to match text embeddings
        if time_ids.ndim == 2 and text_embeddings.ndim == 3:
            # Expand to [B, S, D_time] to match text embedding sequence length
            time_ids = time_ids.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        added_cond_kwargs["time_ids"] = time_ids
        
    # Text embeddings from second encoder need same treatment
    if "text_embeds" in added_cond_kwargs:
        text_embeds = added_cond_kwargs["text_embeds"]  # Shape: [B, D_pooled]
        if text_embeds.ndim == 2 and text_embeddings.ndim == 3:
            # Expand to [B, S, D_pooled]
            text_embeds = text_embeds.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        added_cond_kwargs["text_embeds"] = text_embeds
    
    return model(x, t, text_embeddings, added_cond_kwargs=added_cond_kwargs).sample

def _get_add_time_ids(
    batch_size: int,
    height: int, 
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    hidden_dim: int,  # 2048 per paper section 2.1
    text_encoder_projection_dim: Optional[int] = None
) -> torch.Tensor:
    """
    From section 2.2: Micro-conditioning includes original size, crop coordinates,
    and target size, projected to match embedding dimension.
    """
    # Per paper: Condition on original and target image dimensions
    add_time_ids = torch.tensor([
        [height, width,  # original size
         0, 0,          # crop coordinates (default to 0,0)
         height, width] # target size
    ], dtype=dtype, device=device)  # Shape: [1, 6]
    
    # Repeat for batch
    add_time_ids = add_time_ids.repeat(batch_size, 1)  # Shape: [B, 6]
    
    # Project to embedding space
    total_dim = text_encoder_projection_dim or hidden_dim
    time_embed = torch.nn.Linear(add_time_ids.shape[1], total_dim, device=device, dtype=dtype)
    # Final shape: [B, D_time] where D_time = total_dim
    return time_embed(add_time_ids)

@autocast('cuda')
def forward_pass(
    args: Any,
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Main forward pass implementing architecture from SDXL paper section 2.1"""
    unet = model_dict["unet"]
    text_encoder = model_dict.get("text_encoder")     # CLIP ViT-L
    text_encoder_2 = model_dict.get("text_encoder_2") # OpenCLIP ViT-bigG
    
    # 2048-dim cross attention per paper
    hidden_dim = unet.config.cross_attention_dim
    
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
    batch_size = pixel_values.shape[0]
    height, width = pixel_values.shape[-2:]
    
    # Handle text embeddings - concatenate both encoder outputs
    if "text_embeddings" in batch and "pooled_text_embeddings" in batch:
        text_embeddings = batch["text_embeddings"].to(device=device, non_blocking=True)
        pooled_text_embeddings = batch["pooled_text_embeddings"].to(device=device, non_blocking=True)
    else:
        with torch.no_grad():
            # Get embeddings from both encoders
            text_embeddings = text_encoder(batch["input_ids"].to(device))[0]      # [B, S, 768]
            text_embeddings_2 = text_encoder_2(batch["input_ids_2"].to(device))[0] # [B, S, 1280] 
            pooled_text_embeddings = text_embeddings_2[:, -1]  # [B, 1280]
            # Concatenate to 2048 dim per paper
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)

    # Get time embeddings with proper dimensions
    time_ids = _get_add_time_ids(
        batch_size=batch_size,
        height=height,
        width=width, 
        dtype=dtype,
        device=device,
        hidden_dim=hidden_dim,
        text_encoder_projection_dim=text_encoder_2.config.projection_dim
    )

    # Match sequence dimension for concatenation 
    if time_ids.ndim == 2 and text_embeddings.ndim == 3:
        time_ids = time_ids.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)

    # Prepare all conditioning inputs
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeddings,  # [B, 1280]
        "time_ids": time_ids                    # [B, S, D_time]
    }

    # Get sigmas and compute loss
    sigmas = get_sigmas(height=height, width=width, device=device)
    sigma = sigmas[0]

    return training_loss_v_prediction(
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