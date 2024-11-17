import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import models
import numpy as np
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
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
        logger.error("Scale factor calculation failed: %s", str(e))
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

class SigmaCache:
    """Cache manager for sigma schedules."""
    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached sigma schedule."""
        return self.cache.get(key)

    def set(self, key: str, value: torch.Tensor) -> None:
        """Cache sigma schedule."""
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

# Global cache instance
_sigma_cache = SigmaCache()

def clear_sigma_cache():
    """Clear the sigma schedule cache"""
    _sigma_cache.clear()

def get_sigmas(
    num_inference_steps: int = 28,
    sigma_min: float = 0.0292,
    height: int = 1024,
    width: int = 1024,
    resolution_scaling: bool = True,
    verbose: bool = True,
    base_res: int = 1024 * 1024,
    rho: float = 7.0,
    sigma_max_base: float = 20000.0,  # Increased from 14.6 for ZTSNR
    use_ztsnr: bool = True,  # Enable ZTSNR by default
    cache_key: Optional[str] = None
) -> torch.Tensor:
    """Enhanced sigma schedule generation with ZTSNR support"""
    # Check cache first
    if cache_key is not None:
        cached_sigmas = _sigma_cache.get(cache_key)
        if cached_sigmas is not None:
            return cached_sigmas

    # Calculate sigma_max based on resolution if needed
    sigma_max = sigma_max_base
    if resolution_scaling:
        # Scale sigma_max based on resolution as per NovelAI paper
        current_res = height * width
        scale_factor = (current_res / base_res) ** 0.25  # Rule of thumb: double sigma_max when quadrupling resolution
        sigma_max = sigma_max_base * scale_factor

    # Generate sigma schedule
    sigmas = compute_sigma_schedule(num_inference_steps, sigma_min, sigma_max, rho)
    
    # Add infinite noise step for ZTSNR if enabled
    if use_ztsnr:
        ztsnr_sigma = torch.tensor([20000.0], device=sigmas.device, dtype=sigmas.dtype)  # Practical approximation of infinity
        sigmas = torch.cat([ztsnr_sigma, sigmas])

    # Cache if key provided
    if cache_key is not None:
        _sigma_cache.set(cache_key, sigmas)

    return sigmas

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
    rescale_multiplier: float,
    rescale_cfg: bool,
) -> torch.Tensor:
    """Compute loss weights with MinSNR support"""
    # Implement MinSNR loss weighting
    snr_clipped = torch.minimum(snr, torch.full_like(snr, min_snr_gamma))
    weights = snr_clipped / snr
    
    if rescale_cfg:
        weights = weights * rescale_multiplier
        
    return weights

@torch.jit.script
def compute_v_target(noise: torch.Tensor, sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Compute v-prediction target"""
    return noise * sigma_data / torch.sqrt(sigma * sigma + sigma_data * sigma_data)

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
    """Training loss using v-prediction with MinSNR and ZTSNR support.
    
    Args:
        model: UNet model for prediction
        x_0: Input latent tensor
        sigma: Noise level tensor
        text_embeddings: Text condition embeddings
        added_cond_kwargs: Additional conditioning arguments
        sigma_data: Data sigma parameter (default: 1.0)
        tag_weighter: Optional tag weighting function
        batch_tags: Optional batch tags for weighting
        min_snr_gamma: Minimum SNR gamma for loss weighting (default: 5.0)
        rescale_cfg: Whether to rescale classifier-free guidance (default: True)
        rescale_multiplier: Multiplier for CFG rescaling (default: 1.0)
        use_tag_weighting: Whether to use tag-based loss weighting (default: True)
    
    Returns:
        torch.Tensor: Computed training loss
    """
    # Add noise to input
    noise = torch.randn_like(x_0)
    noised = x_0 + noise * sigma.view(-1, 1, 1, 1)

    # Compute v target (velocity)
    v_target = compute_v_target(noise, sigma, sigma_data)

    # Get model prediction
    v_pred = model(noised, sigma, text_embeddings, added_cond_kwargs=added_cond_kwargs)

    # Compute SNR for loss weighting
    snr = (sigma_data / sigma) ** 2
    weights = compute_loss_weights(
        snr,
        min_snr_gamma=min_snr_gamma,
        rescale_multiplier=rescale_multiplier,
        rescale_cfg=rescale_cfg,
    )

    # Compute weighted MSE loss
    loss = torch.mean((v_pred - v_target) ** 2 * weights)

    # Apply tag weighting if enabled
    if use_tag_weighting and tag_weighter is not None and batch_tags is not None:
        tag_weights = tag_weighter(batch_tags)
        loss = loss * tag_weights.mean()

    return loss

def forward_pass(args, model_dict, batch, device, dtype, components) -> torch.Tensor:
    """Execute forward pass with proper error handling.
    
    Args:
        args: Training configuration arguments
        model_dict: Dictionary containing models (unet, vae, etc.)
        batch: Training batch data
        device: Device to run on
        dtype: Data type for computations
        components: Training components (tag_weighter, vae_finetuner, etc.)
        
    Returns:
        torch.Tensor: Computed loss value
        
    Raises:
        ValueError: If required batch data is missing
        RuntimeError: If forward pass computation fails
        KeyError: If required model or component is missing
    """
    try:
        # Move batch to device
        batch = {k: v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get model predictions
        latents = batch["latents"]
        timesteps = batch["timesteps"]
        
        # Get noise prediction using v-prediction loss
        loss = training_loss_v_prediction(
            model=model_dict["unet"],
            x_0=latents,
            sigma=timesteps,
            text_embeddings=batch["prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": batch["pooled_prompt_embeds"],
                "time_ids": batch["add_text_embeds"],
            },
            sigma_data=args.training.sigma_data,
            tag_weighter=components.get("tag_weighter"),
            batch_tags=batch.get("tags"),
            min_snr_gamma=args.training.min_snr_gamma,
            rescale_cfg=args.training.rescale_cfg,
            rescale_multiplier=args.training.rescale_multiplier,
            use_tag_weighting=args.data.use_tag_weighting,
        )

        # Apply VAE finetuning loss if enabled
        if components.get("vae_finetuner") is not None:
            vae_loss = components["vae_finetuner"].compute_loss(batch)
            loss = loss + vae_loss * args.vae_loss_weight

        return loss

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error("Forward pass failed: %s", str(e))
        raise type(e)(f"Failed during forward pass: {str(e)}") from e

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
        logger.error("Failed to calculate resolution-dependent sigma: %s", str(e))
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
                
            except (TypeError, ValueError, ArithmeticError) as e:
                logger.error("Learning rate calculation failed: %s (type: %s)", str(e), type(e).__name__)
                raise
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
        
    except Exception as e:
        logger.error("Failed to create cosine schedule: %s", str(e))
        logger.error(traceback.format_exc())
        raise

class PerceptualLoss(torch.nn.Module):
    """Perceptual loss using VGG16 features with improved memory efficiency."""
    def __init__(
        self,
        resize: bool = True,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.resize = resize
        self.normalize = normalize
        
        try:
            # Initialize VGG with optimizations
            vgg = models.vgg16(pretrained=True).features
            vgg.eval()
            vgg.requires_grad_(False)
            
            if device is not None:
                vgg = vgg.to(device)
            if dtype is not None:
                vgg = vgg.to(dtype)
            
            # Cache feature layers
            self.layers = {
                '3': 'relu1_2',
                '8': 'relu2_2',
                '15': 'relu3_3',
                '22': 'relu4_3'
            }
            
            self.blocks = torch.nn.ModuleList([])
            curr_block = []
            
            # Build feature extraction blocks
            for name, layer in vgg.named_children():
                curr_block.append(layer)
                if name in self.layers:
                    self.blocks.append(torch.nn.Sequential(*curr_block))
                    curr_block = []
            
            # Register preprocessing
            if self.normalize:
                self.register_buffer(
                    'mean', 
                    torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
                )
                self.register_buffer(
                    'std', 
                    torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
                )
            
            logger.info("Perceptual loss initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize perceptual loss: %s", str(e))
            logger.error(traceback.format_exc())
            raise
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input images."""
        try:
            if self.resize:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            if self.normalize:
                x = (x - self.mean) / self.std
                
            return x
            
        except Exception as e:
            logger.error("Preprocessing failed: %s", str(e))
            raise
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Perceptual loss value
        """
        try:
            if x.shape != y.shape:
                raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")
            
            # Preprocess inputs
            x = self.preprocess(x)
            y = self.preprocess(y)
            
            # Compute features and loss
            loss = 0.0
            for block in self.blocks:
                with torch.cuda.amp.autocast(enabled=False):
                    x = block(x.float())
                    y = block(y.float())
                    loss = loss + F.mse_loss(x, y)
            
            return loss
            
        except Exception as e:
            logger.error("Forward pass failed: %s", str(e))
            logger.error(traceback.format_exc())
            raise
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input tensor."""
        try:
            x = self.preprocess(x)
            features = {}
            
            for i, block in enumerate(self.blocks):
                with torch.cuda.amp.autocast(enabled=False):
                    x = block(x.float())
                    features[self.layers[str(i*7 + 3)]] = x
            
            return features
            
        except Exception as e:
            logger.error("Feature extraction failed: %s", str(e))
            raise