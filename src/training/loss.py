import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models
import numpy as np
from typing import Optional, Dict, Any, Tuple
from typing import Union, Optional, Callable
import numpy as np
import math
import torch
from functools import partial
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from functools import lru_cache
import math


logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = 0.0,
    warmup_init_lr: Optional[float] = None
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Enhanced cosine schedule with warmup, minimum LR, and better initialization
    
    Args:
        optimizer: Torch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        last_epoch: Last epoch number for resuming
        min_lr: Minimum learning rate ratio
        warmup_init_lr: Initial warmup learning rate
        
    Returns:
        LambdaLR scheduler
    """
    
    def validate_inputs():
        """Validate input parameters"""
        if num_warmup_steps < 0:
            raise ValueError(f"Invalid warmup steps: {num_warmup_steps}")
        if num_training_steps <= 0:
            raise ValueError(f"Invalid training steps: {num_training_steps}")
        if num_cycles <= 0:
            raise ValueError(f"Invalid number of cycles: {num_cycles}")
        if not 0.0 <= min_lr <= 1.0:
            raise ValueError(f"Invalid minimum learning rate: {min_lr}")
    
    def get_warmup_lr_ratio(
        current_step: int,
        num_warmup_steps: int,
        init_lr: Optional[float] = None
    ) -> float:
        """Calculate warmup learning rate ratio"""
        if init_lr is not None:
            base_lr = optimizer.param_groups[0]['initial_lr']
            warmup_initial_lr = min(init_lr, base_lr)
            alpha = float(current_step) / float(max(1, num_warmup_steps))
            return warmup_initial_lr / base_lr + (1.0 - warmup_initial_lr / base_lr) * alpha
        
        return float(current_step) / float(max(1, num_warmup_steps))
    
    def get_cosine_decay_ratio(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr: float = 0.0
    ) -> float:
        """Calculate cosine decay learning rate ratio"""
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        cosine_decay = max(0.0, math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        decayed_lr_ratio = 0.5 * (1.0 + cosine_decay)
        
        # Apply minimum learning rate
        return min_lr + (1.0 - min_lr) * decayed_lr_ratio
    
    @torch.jit.script
    def lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
        min_lr: float,
        warmup_init_lr: Optional[float] = None
    ) -> float:
        """JIT-compiled learning rate calculation"""
        if current_step < num_warmup_steps:
            return get_warmup_lr_ratio(current_step, num_warmup_steps, warmup_init_lr)
        return get_cosine_decay_ratio(
            current_step, num_warmup_steps, num_training_steps, num_cycles, min_lr
        )
    
    # Validate inputs
    validate_inputs()
    
    # Create partial function with fixed parameters
    lr_lambda_partial = partial(
        lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr=min_lr,
        warmup_init_lr=warmup_init_lr
    )
    
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda_partial,
        last_epoch
    )

# Additional utility functions
def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr: float = 0.0
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear learning rate schedule with warmup"""
    
    @torch.jit.script
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Polynomial decay schedule with warmup"""
    
    @torch.jit.script
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        lr_range = 1.0 - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * (pct_remaining ** power) + lr_end
        return decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)




logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def calculate_scale_factor(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    method: str = "karras"
) -> float:
    """Cached calculation of resolution-dependent scale factor"""
    current_res = height * width
    if method == "karras":
        return float(current_res / base_res)
    return float(np.sqrt(current_res / base_res))

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
    """
    Enhanced sigma schedule generation with caching and optimizations
    
    Args:
        num_inference_steps: Number of steps (default: 28)
        sigma_min: Minimum sigma (default: 0.0292)
        height: Image height
        width: Image width
        resolution_scaling: Enable resolution scaling
        scale_method: Scaling method ("karras" or "simple")
        verbose: Enable verbose logging
        device: Target device for tensor
        dtype: Target dtype for tensor
        base_res: Base resolution (default: 1024x1024)
        rho: EDM rho parameter (default: 7.0)
        sigma_max_base: Base maximum sigma (default: 20000.0)
        cache_key: Optional key for result caching
        
    Returns:
        torch.Tensor: Computed sigma schedule
    """
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

def clear_sigma_cache():
    """Clear the sigma schedule cache"""
    if hasattr(get_sigmas, '_cache'):
        get_sigmas._cache.clear()
    calculate_scale_factor.cache_clear()

def get_sigma_schedule_info(sigmas: torch.Tensor) -> dict:
    """Get information about a sigma schedule"""
    return {
        'steps': len(sigmas),
        'min': float(sigmas.min()),
        'max': float(sigmas.max()),
        'mean': float(sigmas.mean()),
        'std': float(sigmas.std()),
        'dynamic_range': float(sigmas.max() / sigmas.min())
    }


@torch.jit.script
def compute_v_prediction_scaling(
    sigma: torch.Tensor,
    sigma_data: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute v-prediction scaling factors (c_out, c_in)"""
    # Pre-compute common terms
    sigma_sq = sigma ** 2
    sigma_data_sq = sigma_data ** 2
    denominator = torch.sqrt(sigma_sq + sigma_data_sq)
    
    # Output and input scaling only (no skip connection needed for v-prediction)
    c_out = (-sigma * sigma_data) / denominator
    c_in = 1.0 / denominator
    
    return c_out, c_in

def v_prediction_scaling_factors(
    sigma: Union[torch.Tensor, float], 
    sigma_data: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced v-prediction scaling factors computation with caching and optimizations
    
    Args:
        sigma: Noise level (tensor or scalar)
        sigma_data: Data standard deviation (default: 1.0)
        device: Target device for tensors
        dtype: Target dtype for tensors
        use_cache: Enable result caching for scalar inputs
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (c_out, c_in) scaling factors
        
    References:
        NovelAI paper equations (11)-(13)
        Karras preconditioner for v-prediction parameterization
    """
    # Input validation
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"Invalid sigma value: {sigma}")
        if use_cache:
            # Use cached computation for scalar inputs
            c_out, c_in = compute_v_prediction_scaling(float(sigma), sigma_data)
            if device is not None or dtype is not None:
                # Convert to tensors with specified device/dtype
                return (
                    torch.tensor(c_out, device=device, dtype=dtype),
                    torch.tensor(c_in, device=device, dtype=dtype)
                )
            return (
                torch.tensor(c_out),
                torch.tensor(c_in)
            )
        # Convert scalar to tensor
        sigma = torch.tensor(sigma)
    
    # Validate tensor input
    if torch.any(sigma <= 0):
        raise ValueError("Sigma values must be positive")
    
    # Move to specified device/dtype if needed
    if device is not None:
        sigma = sigma.to(device)
    if dtype is not None:
        sigma = sigma.to(dtype)
    
    # Compute scaling factors using JIT-compiled function
    return compute_v_prediction_scaling(sigma, sigma_data)

def clear_scaling_factors_cache():
    """Clear the scaling factors computation cache"""
    calculate_scale_factor.cache_clear()

def get_scaling_factors_info(
    c_out: torch.Tensor,
    c_in: torch.Tensor
) -> dict:
    """Get information about computed scaling factors"""
    return {
        'c_out': {
            'min': float(c_out.min()),
            'max': float(c_out.max()),
            'mean': float(c_out.mean())
        },
        'c_in': {
            'min': float(c_in.min()),
            'max': float(c_in.max()),
            'mean': float(c_in.mean())
        }
    }


@torch.jit.script
def compute_mse_components(
    model_output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none"
) -> torch.Tensor:
    """Optimized MSE computation"""
    return F.mse_loss(model_output, target, reduction=reduction)

@torch.jit.script
def compute_loss_weights(
    snr: torch.Tensor,
    min_snr_gamma: float,
    scale_method: str,
    rescale_multiplier: float,
    rescale_cfg: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute SNR weights and CFG scaling"""
    # MinSNR weighting
    snr_weight = (snr / min_snr_gamma).clamp(max=1.0)
    
    # CFG rescaling
    if rescale_cfg:
        if scale_method == "karras":
            cfg_scale = rescale_multiplier * torch.sqrt(1 + snr)
        else:  # simple
            cfg_scale = rescale_multiplier * (1 + snr)
    else:
        cfg_scale = torch.ones_like(snr)  # Default to 1.0 when rescaling is disabled
    
    return snr_weight, cfg_scale

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
    verbose: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Enhanced v-prediction loss with optimized computations and better organization
    
    Args:
        model: UNet model
        x_0: Clean data (latents)
        sigma: Noise level
        text_embeddings: Text condition embeddings
        added_cond_kwargs: Additional conditioning
        sigma_data: Data standard deviation
        tag_weighter: Tag weighting module
        batch_tags: Batch tag information
        min_snr_gamma: MinSNR gamma parameter
        rescale_cfg: Enable CFG rescaling
        rescale_multiplier: CFG rescale multiplier
        scale_method: CFG scale method
        use_tag_weighting: Enable tag weighting
        verbose: Enable verbose logging
        device: Target device
        dtype: Target dtype
    """
    try:
        # Input validation
        if scale_method not in ["karras", "simple"]:
            raise ValueError(f"Invalid scale method: {scale_method}")
        if min_snr_gamma <= 0:
            raise ValueError(f"Invalid MinSNR gamma: {min_snr_gamma}")
        
        # Generate noise
        noise = torch.randn_like(x_0)
        
        # Move to device/dtype if specified
        if device is not None or dtype is not None:
            noise = noise.to(device=device, dtype=dtype)
            sigma = sigma.to(device=device, dtype=dtype)
        
        # Add noise with broadcasting
        noised = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get scaling factors
        c_out, c_in = v_prediction_scaling_factors(
            sigma, sigma_data, device=device, dtype=dtype
        )
        
        # Scale input
        model_input = c_in.view(-1, 1, 1, 1) * noised
        
        # Forward pass
        with torch.set_grad_enabled(True):
            model_output = model(
                model_input,
                sigma.view(-1, 1, 1, 1),
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            ).sample
        
        # Compute target
        target = c_out.view(-1, 1, 1, 1) * noise
        
        # Compute MSE components
        mse = compute_mse_components(
            model_output, target, reduction="none"
        ).mean(dim=(1, 2, 3))
        
        # Compute SNR and weights
        snr = sigma_data**2 / sigma**2
        snr_weight, cfg_scale = compute_loss_weights(
            snr, min_snr_gamma, scale_method, rescale_multiplier, rescale_cfg
        )
        
        # Compute weighted loss
        loss = (mse * snr_weight).mean()
        
        # Apply CFG scaling
        if cfg_scale is not None:
            loss = loss * cfg_scale.mean()
        
        # Apply tag weighting
        if use_tag_weighting and tag_weighter is not None and batch_tags is not None:
            tag_weights = tag_weighter(batch_tags)
            loss = loss * tag_weights.mean()
        
        # Log components if verbose
        if verbose:
            log_components = {
                "Base MSE": mse.mean().item(),
                "SNR weight": snr_weight.mean().item(),
                "CFG scale": cfg_scale.mean().item() if cfg_scale is not None else None,
                "Tag weight": tag_weights.mean().item() if 'tag_weights' in locals() else None,
                "Final loss": loss.item()
            }
            logger.info("Loss components:")
            for name, value in log_components.items():
                if value is not None:
                    logger.info(f"- {name}: {value:.4e}")
        
        return loss
        
    except Exception as e:
        logger.error(f"Error in training_loss_v_prediction: {str(e)}")
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

class PerceptualLoss(nn.Module):
    """Enhanced VGG-based perceptual loss with optimizations"""
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
        layers: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        cache_size: int = 32
    ):
        """
        Initialize perceptual loss module
        
        Args:
            device: Target device
            dtype: Target dtype
            layers: Custom layer mapping
            weights: Layer-wise loss weights
            cache_size: Feature cache size
        """
        super().__init__()
        
        # Initialize VGG model
        self.vgg = self._setup_vgg(device, dtype)
        
        # Layer configuration
        self.layers = layers or {
            '3': 'relu1_2',
            '8': 'relu2_2', 
            '15': 'relu3_3',
            '22': 'relu4_3'
        }
        
        # Layer weights
        self.weights = weights or {name: 1.0 for name in self.layers.values()}
        
        # Setup normalization
        self.register_buffer(
            'mean',
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )
        
        # Feature cache
        self.cache_size = cache_size
        self._feature_cache = {}
    
    def _setup_vgg(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype]
    ) -> nn.Module:
        """Setup and optimize VGG model"""
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval().requires_grad_(False)
        
        # Move to device/dtype
        if dtype is not None:
            vgg = vgg.to(dtype)
        return vgg.to(device)
    
    @torch.no_grad()
    def get_features(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Extract features with caching"""
        # Check cache
        cache_key = hash(x.data_ptr())
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Normalize input
        if normalize:
            x = (x - self.mean) / self.std
        
        # Extract features
        features = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x.detach()
        
        # Update cache
        if len(self._feature_cache) >= self.cache_size:
            # Remove oldest entry
            self._feature_cache.pop(next(iter(self._feature_cache)))
        self._feature_cache[cache_key] = features
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = True,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted images
            target: Target images
            normalize: Apply ImageNet normalization
            reduction: Loss reduction method
        """
        # Handle grayscale inputs
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Get features
        pred_features = self.get_features(pred, normalize)
        target_features = self.get_features(target, normalize)
        
        # Compute weighted loss
        losses = []
        for key in pred_features:
            loss = F.mse_loss(
                pred_features[key],
                target_features[key],
                reduction='none'
            ).mean(dim=(1, 2, 3))
            losses.append(loss * self.weights[key])
        
        # Combine losses
        total_loss = torch.stack(losses).sum(dim=0)
        
        # Apply reduction
        if reduction == 'mean':
            return total_loss.mean()
        elif reduction == 'sum':
            return total_loss.sum()
        return total_loss
    
    def clear_cache(self) -> None:
        """Clear feature cache"""
        self._feature_cache.clear()

@lru_cache(maxsize=128)
def get_resolution_dependent_sigma_max(
    height: int,
    width: int,
    base_res: int = 1024 * 1024,
    base_sigma: float = 20000.0,
    scale_power: float = 0.25
) -> float:
    """
    Enhanced resolution-dependent sigma calculation with caching
    
    Args:
        height: Image height
        width: Image width
        base_res: Base resolution
        base_sigma: Base sigma value
        scale_power: Scaling power factor
    """
    current_res = height * width
    scale_factor = (current_res / base_res) ** scale_power
    return base_sigma * scale_factor