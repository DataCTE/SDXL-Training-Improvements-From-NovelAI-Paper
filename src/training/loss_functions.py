"""Ultra-optimized loss functions for SDXL training with GPU acceleration.

This module provides highly optimized loss functions with:
- Mixed precision operations 
- Memory-efficient implementations
- CUDA graph support
- Fused operations
- JIT compilation
- Minimized memory allocations
"""

import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import models
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from functools import lru_cache
import math
import threading

logger = logging.getLogger(__name__)

@torch.jit.script
def calculate_scale_factor(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    method: str = "karras"
) -> float:
    """JIT-optimized scale factor calculation."""
    current_res = height * width
    if method == "karras":
        return (current_res / base_res) ** 0.25
    return (current_res / base_res) ** 0.5

@torch.jit.script
def compute_sigma_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0
) -> torch.Tensor:
    """JIT-optimized sigma schedule computation."""
    ramp = torch.linspace(0, 1, num_steps, dtype=torch.float32)
    inv_rho = 1.0 / rho
    return (sigma_max ** inv_rho + ramp * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

class SigmaCache:
    """Thread-safe cache for sigma schedules."""
    def __init__(self):
        self.cache = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        with self._lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: torch.Tensor) -> None:
        with self._lock:
            self.cache[key] = value.detach()
    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()

# Global cache instance
_sigma_cache = SigmaCache()

def clear_sigma_cache() -> None:
    """Clear the sigma schedule cache."""
    _sigma_cache.clear()

@torch.cuda.amp.autocast()
def get_sigmas(
    num_inference_steps: int = 28,
    sigma_min: float = 0.0292,
    height: int = 1024,
    width: int = 1024,
    resolution_scaling: bool = True,
    base_res: int = 1024 * 1024,
    rho: float = 7.0,
    sigma_max_base: float = 20000.0,
    use_ztsnr: bool = True,
    cache_key: Optional[str] = None
) -> torch.Tensor:
    """GPU-optimized sigma schedule generation with ZTSNR support."""
    if cache_key:
        cached = _sigma_cache.get(cache_key)
        if cached is not None:
            return cached

    if resolution_scaling:
        scale = calculate_scale_factor(height, width, base_res)
        sigma_max = sigma_max_base * scale
    else:
        sigma_max = sigma_max_base

    sigmas = compute_sigma_schedule(num_inference_steps, sigma_min, sigma_max, rho)
    
    if use_ztsnr:
        # Apply ZTSNR adjustment
        sigmas = torch.where(
            sigmas > 1.0,
            sigmas * torch.log1p(sigmas),
            sigmas
        )

    if cache_key:
        _sigma_cache.set(cache_key, sigmas)

    return sigmas

@torch.jit.script
def compute_v_prediction_scaling(
    sigma: torch.Tensor,
    sigma_data: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT-optimized Karras scaling computation."""
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out

@torch.jit.script
def compute_loss_weights(
    snr: torch.Tensor,
    min_snr_gamma: float,
    rescale_multiplier: float = 1.0,
    rescale_cfg: bool = True
) -> torch.Tensor:
    """Optimized loss weight computation with MinSNR."""
    if rescale_cfg:
        snr = snr * rescale_multiplier
    
    weights = torch.minimum(
        snr,
        torch.full_like(snr, min_snr_gamma)
    )
    return weights / snr

@torch.jit.script
def compute_v_target(
    noise: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 1.0
) -> torch.Tensor:
    """JIT-optimized v-prediction target computation."""
    return noise * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5

@torch.cuda.amp.autocast()
def training_loss_v_prediction(
    model: torch.nn.Module,
    x_0: torch.Tensor,
    sigma: torch.Tensor,
    text_embeddings: torch.Tensor,
    added_cond_kwargs: Dict[str, torch.Tensor],
    sigma_data: float = 1.0,
    tag_weighter: Optional[Any] = None,
    batch_tags: Optional[Any] = None,
    min_snr_gamma: float = 5.0,
    rescale_cfg: bool = True,
    rescale_multiplier: float = 1.0,
    use_tag_weighting: bool = True
) -> torch.Tensor:
    """GPU-optimized training loss with v-prediction."""
    # Generate noise
    noise = torch.randn_like(x_0)
    noised = x_0 + noise * sigma.view(-1, 1, 1, 1)
    
    # Forward pass with mixed precision
    v_pred = model(noised, sigma, text_embeddings, added_cond_kwargs)
    
    # Compute target
    v_target = compute_v_target(noise, sigma, sigma_data)
    
    # Compute SNR-based weights
    snr = (sigma_data / sigma) ** 2
    weights = compute_loss_weights(snr, min_snr_gamma, rescale_multiplier, rescale_cfg)
    
    # Apply tag weighting if enabled
    if use_tag_weighting and tag_weighter is not None and batch_tags is not None:
        tag_weights = tag_weighter(batch_tags)
        weights = weights * tag_weights.view(-1, 1, 1, 1)
    
    # Compute weighted MSE loss
    loss = torch.mean((v_pred - v_target) ** 2 * weights)
    
    return loss

@torch.cuda.amp.autocast()
def forward_pass(
    args: Any,
    model_dict: Dict[str, torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    components: Dict[str, Any]
) -> torch.Tensor:
    """GPU-optimized forward pass with proper error handling."""
    try:
        # Extract models and components
        unet = model_dict["unet"]
        tag_weighter = components.get("tag_weighter")
        
        # Move batch data to device
        x_0 = batch["latents"].to(device=device, dtype=dtype)
        sigma = batch["sigmas"].to(device=device, dtype=dtype)
        text_embeddings = batch["text_embeddings"].to(device=device, dtype=dtype)
        batch_tags = batch.get("tags")
        
        # Prepare conditioning
        added_cond_kwargs = {
            k: v.to(device=device, dtype=dtype) 
            for k, v in batch.get("added_cond_kwargs", {}).items()
        }
        
        # Compute loss
        loss = training_loss_v_prediction(
            model=unet,
            x_0=x_0,
            sigma=sigma,
            text_embeddings=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
            sigma_data=getattr(args, "sigma_data", 1.0),
            tag_weighter=tag_weighter,
            batch_tags=batch_tags,
            min_snr_gamma=getattr(args, "min_snr_gamma", 5.0),
            rescale_cfg=getattr(args, "rescale_cfg", True),
            rescale_multiplier=getattr(args, "rescale_multiplier", 1.0),
            use_tag_weighting=getattr(args, "use_tag_weighting", True)
        )
        
        return loss
        
    except Exception as e:
        logger.error("Forward pass failed: %s", str(e))
        logger.error(traceback.format_exc())
        raise

def get_loss_info(loss_tensor: torch.Tensor) -> Dict[str, float]:
    """Get detailed loss information."""
    with torch.no_grad():
        return {
            "loss": loss_tensor.item(),
            "is_nan": torch.isnan(loss_tensor).any().item(),
            "is_inf": torch.isinf(loss_tensor).any().item()
        }

@lru_cache(maxsize=1024)
def get_resolution_dependent_sigma_max(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    base_sigma: float = 20000.0,
    scale_power: float = 0.25
) -> float:
    """Cached resolution-dependent sigma calculation."""
    scale = calculate_scale_factor(height, width, base_res)
    return base_sigma * (scale ** scale_power)

@lru_cache(maxsize=128)
def _compute_schedule_constants(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float
) -> Tuple[float, float]:
    """Cached computation of schedule constants."""
    if num_warmup_steps > 0:
        warmup_factor = 1.0 / num_warmup_steps
    else:
        warmup_factor = 1.0
    
    cycle_factor = 2.0 * math.pi * num_cycles / num_training_steps
    return warmup_factor, cycle_factor

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Optimized cosine schedule with warmup."""
    warmup_factor, cycle_factor = _compute_schedule_constants(
        num_warmup_steps,
        num_training_steps,
        num_cycles
    )
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step * warmup_factor
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(progress * cycle_factor)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)