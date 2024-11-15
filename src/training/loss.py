import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, Callable
from functools import partial, lru_cache
import math

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def calculate_scale_factor(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    method: str = "karras"
) -> float:
    """Cached calculation of resolution-dependent scale factor"""
    try:
        current_res = height * width
        if method == "karras":
            return float(current_res / base_res)
        return float(np.sqrt(current_res / base_res))
    except Exception as e:
        logger.error(f"Scale factor calculation failed: {str(e)}")
        raise

@torch.jit.script
def compute_sigma_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float
) -> torch.Tensor:
    """JIT-compiled sigma schedule computation"""
    t = torch.linspace(0, 1, num_steps, dtype=torch.float32)
    inv_rho = 1.0 / rho
    return (sigma_max ** inv_rho + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

def get_sigmas(
    num_inference_steps: int = 28, 
    sigma_min: float = 0.0292, 
    height: int = 1024, 
    width: int = 1024,
    resolution_scaling: bool = True,
    scale_method: str = "karras",
    verbose: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    base_res: int = 1024 * 1024,
    rho: float = 7.0,
    sigma_max_base: float = 20000.0,
    cache_key: Optional[str] = None
) -> torch.Tensor:
    """Enhanced sigma schedule generation with caching and optimizations"""
    try:
        # Input validation
        if num_inference_steps < 1:
            raise ValueError(f"Invalid number of steps: {num_inference_steps}")
        if sigma_min <= 0:
            raise ValueError(f"Invalid minimum sigma: {sigma_min}")
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("Height and width must be integers")
        if scale_method not in ["karras", "simple"]:
            raise ValueError(f"Invalid scaling method: {scale_method}")
        
        # Use cache if available
        if cache_key is not None:
            cache_dict = getattr(get_sigmas, '_cache', {})
            if cache_key in cache_dict:
                return cache_dict[cache_key]
        
        # Calculate scale factor
        scale_factor = (
            calculate_scale_factor(height, width, base_res, scale_method)
            if resolution_scaling else 1.0
        )
        
        # Compute sigma max
        sigma_max = sigma_max_base * scale_factor
        
        # Log configuration if verbose
        if verbose:
            log_config = {
                "Resolution": f"{width}x{height} (scale: {scale_factor:.3f})",
                "Scaling method": scale_method,
                "Sigma range": f"{sigma_min:.4f} to {sigma_max:.1f}",
                "Steps": num_inference_steps,
                "Rho": rho
            }
            logger.info("Generating sigma schedule:")
            for key, value in log_config.items():
                logger.info(f"- {key}: {value}")
        
        # Compute schedule
        sigmas = compute_sigma_schedule(num_inference_steps, sigma_min, sigma_max, rho)
        
        # Move to device/dtype if specified
        if device is not None:
            sigmas = sigmas.to(device)
        if dtype is not None:
            sigmas = sigmas.to(dtype)
        
        # Log schedule details if verbose
        if verbose:
            logger.info(f"- First 3 sigmas: {sigmas[:3].tolist()}")
            logger.info(f"- Last 3 sigmas: {sigmas[-3:].tolist()}")
        
        # Cache result if requested
        if cache_key is not None:
            if not hasattr(get_sigmas, '_cache'):
                get_sigmas._cache = {}
            get_sigmas._cache[cache_key] = sigmas
        
        return sigmas
        
    except Exception as e:
        logger.error(f"Failed to generate sigmas: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def clear_sigma_cache():
    """Clear the sigma schedule cache"""
    if hasattr(get_sigmas, '_cache'):
        get_sigmas._cache.clear()
    calculate_scale_factor.cache_clear()

@torch.jit.script
def compute_v_prediction_scaling(
    sigma: torch.Tensor,
    sigma_data: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized Karras scaling computation for v-prediction"""
    sigma_sq = sigma * sigma
    sigma_data_sq = sigma_data * sigma_data
    denominator = torch.sqrt(sigma_sq + sigma_data_sq)
    c_out = -sigma * sigma_data / denominator
    c_in = 1.0 / denominator
    return c_out, c_in

@torch.jit.script
def compute_loss_weights(
    snr: torch.Tensor,
    min_snr_gamma: float,
    scale_method: str,
    rescale_multiplier: float,
    rescale_cfg: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized loss weight computation"""
    snr_weight = torch.clamp(snr / min_snr_gamma, max=1.0)
    
    if rescale_cfg:
        snr_plus_one = 1.0 + snr
        cfg_scale = rescale_multiplier * (
            torch.sqrt(snr_plus_one) if scale_method == "karras"
            else snr_plus_one
        )
    else:
        cfg_scale = torch.ones_like(snr)
    
    return snr_weight, cfg_scale

@torch.compile()
def training_loss_v_prediction(
    model: torch.nn.Module,
    x_0: torch.Tensor,
    sigma: torch.Tensor,
    text_embeddings: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, Any]] = None,
    sigma_data: float = 1.0,
    tag_weighter: Optional[Any] = None,
    batch_tags: Optional[Any] = None,
    min_snr_gamma: float = 5.0,
    rescale_cfg: bool = True,
    rescale_multiplier: float = 0.7,
    scale_method: str = "karras",
    use_tag_weighting: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Ultra-optimized v-prediction loss with minimal memory operations"""
    try:
        # Input validation
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be a torch.nn.Module")
        if not torch.is_tensor(x_0) or not torch.is_tensor(sigma) or not torch.is_tensor(text_embeddings):
            raise ValueError("Inputs must be torch tensors")
        if scale_method not in ["karras", "simple"]:
            raise ValueError(f"Invalid scale method: {scale_method}")
            
        # Ensure contiguous tensors for better memory access
        x_0 = x_0.contiguous()
        sigma = sigma.contiguous()
        text_embeddings = text_embeddings.contiguous()
        
        # Single contiguous memory allocation for noise
        noise = torch.randn_like(x_0, memory_format=torch.contiguous_format)
        
        # Move tensors to device/dtype if specified
        if device is not None or dtype is not None:
            tensors_to_move = [noise, sigma, x_0, text_embeddings]
            for tensor in tensors_to_move:
                tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
        
        # Pre-compute common terms (fused operations)
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        sigma_sq = sigma * sigma
        sigma_data_sq = sigma_data * sigma_data
        
        # Compute scaling factors with fused operations
        inv_denominator = torch.rsqrt(sigma_sq + sigma_data_sq)
        c_out = -sigma * sigma_data * inv_denominator
        c_in = inv_denominator
        
        # Compute noised input with fused operations
        noised = torch.addcmul(x_0, sigma_expanded, noise)
        model_input = noised * c_in.view(-1, 1, 1, 1)
        
        # Forward pass with error handling
        try:
            model_output = model(
                model_input,
                sigma_expanded,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            ).sample
        except Exception as e:
            logger.error(f"Model forward pass failed: {str(e)}")
            raise
        
        # Compute loss components with fused operations
        target = noise * c_out.view(-1, 1, 1, 1)
        mse = torch.square(model_output - target).mean(dim=(1, 2, 3))
        
        # Compute SNR and weights with optimized operations
        snr = torch.square(sigma_data * inv_denominator * sigma)
        snr_weight, cfg_scale = compute_loss_weights(
            snr, min_snr_gamma, scale_method, rescale_multiplier, rescale_cfg
        )
        
        # Compute final loss with fused operations
        loss = torch.mean(mse * snr_weight) * cfg_scale.mean()
        
        # Apply tag weighting if enabled
        if use_tag_weighting and tag_weighter is not None and batch_tags is not None:
            try:
                tag_weight = tag_weighter(batch_tags).mean()
                loss = loss * tag_weight
            except Exception as e:
                logger.warning(f"Tag weighting failed, skipping: {str(e)}")
        
        return loss
        
    except Exception as e:
        logger.error(f"Loss computation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_loss_info(loss_tensor: torch.Tensor) -> dict:
    """Get detailed information about loss values"""
    return {
        'mean': float(loss_tensor.mean()),
        'std': float(loss_tensor.std()),
        'min': float(loss_tensor.min()),
        'max': float(loss_tensor.max()),
        'non_finite': int(torch.sum(~torch.isfinite(loss_tensor)))
    }

@lru_cache(maxsize=128)
def get_resolution_dependent_sigma_max(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    base_sigma: float = 20000.0,
    scale_power: float = 0.25
) -> float:
    """Enhanced resolution-dependent sigma calculation with caching"""
    try:
        current_res = height * width
        scale_factor = (current_res / base_res) ** scale_power
        return base_sigma * scale_factor
    except Exception as e:
        logger.error(f"Failed to calculate resolution-dependent sigma: {str(e)}")
        raise

@lru_cache(maxsize=1)
def _compute_schedule_constants(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float
) -> Tuple[float, float, float]:
    """Cached computation of schedule constants."""
    return (
        1.0 / max(1, num_warmup_steps),  # warmup_scale
        1.0 / max(1, num_training_steps - num_warmup_steps),  # progress_scale
        2.0 * math.pi * num_cycles  # pi_cycles
    )

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Optimized cosine schedule with warmup based on NovelAI's training methodology.
    
    Key optimizations:
    1. Cached constants for faster runtime calculations
    2. Vectorized operations where possible
    3. Minimized redundant computations in lr_lambda
    4. Optimized math operations for better numerical stability
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (default: 0.5 as per NovelAI paper)
        last_epoch: Last epoch number for resuming
        
    Returns:
        LambdaLR scheduler with optimized cosine warmup schedule
    """
    try:
        # Validate inputs
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
        if num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative")
        if num_training_steps <= 0:
            raise ValueError("num_training_steps must be positive")
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive")
            
        # Get cached constants
        warmup_scale, progress_scale, pi_cycles = _compute_schedule_constants(
            num_warmup_steps, num_training_steps, num_cycles
        )
        
        def lr_lambda(current_step: int) -> float:
            try:
                # Warmup phase
                if current_step < num_warmup_steps:
                    return float(current_step * warmup_scale)
                    
                # Cosine decay phase with optimized computation
                progress = float((current_step - num_warmup_steps) * progress_scale)
                cosine = math.cos(pi_cycles * progress)
                return max(0.0, 0.5 * (1.0 + cosine))
                
            except Exception as e:
                logger.error(f"Error in lr_lambda calculation: {str(e)}")
                return 0.0  # Safe fallback
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
        
    except Exception as e:
        logger.error(f"Failed to create cosine schedule: {str(e)}")
        logger.error(traceback.format_exc())
        raise