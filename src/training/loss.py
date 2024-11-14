import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def get_sigmas(num_inference_steps: int = 28, 
               sigma_min: float = 0.0292, 
               height: int = 1024, 
               width: int = 1024,
               verbose: bool = True) -> torch.Tensor:
    """
    Generate sigmas for ZTSNR with resolution-dependent scaling using 28-step native schedule.
    Implements NovelAI's quadratic scaling for redundant signal (section 2.3).
    
    Args:
        num_inference_steps: Number of inference steps (default: 28 from NovelAI paper)
        sigma_min: Minimum sigma value (default: 0.0292 from SDXL paper)
        height: Image height for resolution scaling
        width: Image width for resolution scaling
        verbose: Whether to log sigma schedule details
    
    Returns:
        torch.Tensor: Generated sigma schedule
    """
    # Calculate resolution-dependent sigma_max with quadratic scaling
    base_res = 1024 * 1024
    current_res = height * width
    scale_factor = (current_res / base_res)  # Quadratic scaling for redundant signal
    
    # Use 20000 as practical infinity approximation for ZTSNR (appendix A.2)
    sigma_max = 20000.0 * scale_factor
    
    if verbose:
        logger.info(f"Generating sigma schedule:")
        logger.info(f"- Resolution: {width}x{height} (scale factor: {scale_factor:.3f})")
        logger.info(f"- Sigma range: {sigma_min:.4f} to {sigma_max:.1f}")
    
    # Use non-linear spacing with EDM-style karras schedule
    rho = 7.0  # EDM's recommended value
    t = torch.linspace(0, 1, num_inference_steps)
    inv_rho = 1.0 / rho
    
    # Log-space interpolation with sigma conversion
    sigmas = (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    
    if verbose:
        logger.info(f"- Schedule steps: {num_inference_steps}")
        logger.info(f"- First 3 sigmas: {sigmas[:3].tolist()}")
        logger.info(f"- Last 3 sigmas: {sigmas[-3:].tolist()}")
    
    return sigmas

def v_prediction_scaling_factors(sigma: torch.Tensor, 
                               sigma_data: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute v-prediction scaling factors according to NovelAI paper equations (11)-(13).
    Implements the Karras preconditioner for v-prediction parameterization.
    
    Args:
        sigma: Noise level
        sigma_data: Data standard deviation (default: 1.0 for normalized latents)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (c_skip, c_out, c_in) scaling factors
    """
    # Skip factor for zero terminal SNR case (eq. 11)
    c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
    
    # Output scaling with negative sign for v-prediction (eq. 12)
    c_out = (-sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)
    
    # Input scaling for proper normalization (eq. 13)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return c_skip, c_out, c_in

def training_loss_v_prediction(model: torch.nn.Module,
                             x_0: torch.Tensor,
                             sigma: torch.Tensor,
                             text_embeddings: torch.Tensor,
                             added_cond_kwargs: Optional[Dict[str, Any]] = None,
                             sigma_data: float = 1.0,
                             tag_weighter: Optional[Any] = None,
                             batch_tags: Optional[Any] = None,
                             min_snr_gamma: float = 5.0,
                             verbose: bool = False) -> torch.Tensor:
    """
    Calculate v-prediction loss with MinSNR weighting and SDXL architecture support.
    Implements NovelAI's v-prediction and MinSNR improvements (sections 2.1, 2.4).
    
    Args:
        model: UNet model (must match SDXL architecture)
        x_0: Clean data
        sigma: Noise level
        text_embeddings: Combined CLIP embeddings (SDXL format)
        added_cond_kwargs: Additional conditioning (time_ids, text_embeds)
        sigma_data: Data standard deviation
        tag_weighter: Optional tag weighting function
        batch_tags: Optional batch tags for weighting
        min_snr_gamma: Minimum SNR value (γ) for loss weighting
        verbose: Whether to log detailed loss information
    
    Returns:
        torch.Tensor: Computed loss value
    """
    try:
        dtype = x_0.dtype
        device = x_0.device
        
        # Log input shapes for debugging
        if verbose:
            logger.info(f"Input shapes:")
            logger.info(f"x_0: {x_0.shape}, dtype: {x_0.dtype}")
            logger.info(f"sigma: {sigma.shape}, dtype: {sigma.dtype}")
            logger.info(f"text_embeddings: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")
        
        # Ensure consistent dtype and device
        sigma = sigma.to(dtype=dtype, device=device)
        text_embeddings = text_embeddings.to(dtype=dtype, device=device)
        
        # Calculate noise
        noise = torch.randn_like(x_0)
        
        # Get v-prediction scaling factors
        c_skip, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)
        
        # Calculate noised input
        x_noisy = c_skip * x_0 + c_out * noise
        
        # Scale model input for proper normalization
        model_input = c_in * x_noisy
        
        # Forward pass through SDXL UNet with detailed error tracking
        try:
            model_output = model(
                model_input,
                sigma,
                text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            )
        except Exception as model_error:
            logger.error("Error during model forward pass:")
            logger.error(f"Error type: {type(model_error).__name__}")
            logger.error(f"Error message: {str(model_error)}")
            
            # Print shapes of all inputs
            logger.error("\nInput tensor shapes:")
            logger.error(f"model_input: {model_input.shape}")
            logger.error(f"sigma: {sigma.shape}")
            logger.error(f"text_embeddings: {text_embeddings.shape}")
            if added_cond_kwargs:
                for key, value in added_cond_kwargs.items():
                    if hasattr(value, 'shape'):
                        logger.error(f"{key}: {value.shape}")
            
            # Print full traceback
            tb = traceback.format_exc()
            logger.error("\nFull traceback:")
            for line in tb.split('\n'):
                logger.error(line)
            
            raise RuntimeError("Model forward pass failed - see logs for details") from model_error
        
        # Calculate target
        target = c_in * x_0
        
        # Calculate SNR for loss weighting (MinSNR)
        snr = (sigma_data / sigma) ** 2
        min_snr = torch.full_like(snr, min_snr_gamma)
        loss_weights = (snr / min_snr).clamp(max=1.0)
        
        # Calculate weighted MSE loss with nan/inf checking
        loss = F.mse_loss(model_output, target, reduction='none')
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error("NaN or Inf detected in loss computation")
            logger.error(f"model_output stats: min={model_output.min()}, max={model_output.max()}, mean={model_output.mean()}")
            logger.error(f"target stats: min={target.min()}, max={target.max()}, mean={target.mean()}")
            raise ValueError("NaN or Inf values detected in loss computation")
            
        loss = loss.mean(dim=(1, 2, 3))
        loss = (loss * loss_weights).mean()
        
        # Apply tag weighting if provided
        if tag_weighter is not None and batch_tags is not None:
            tag_weights = tag_weighter(batch_tags)
            loss = loss * tag_weights.mean()
        
        if verbose:
            logger.info(f"Loss computation details:")
            logger.info(f"- Input shape: {x_0.shape}")
            logger.info(f"- Current sigma: {sigma.mean().item():.4f}")
            logger.info(f"- SNR: {snr.mean().item():.4f}")
            logger.info(f"- Loss weights: {loss_weights.mean().item():.4f}")
            logger.info(f"- Raw loss: {loss.item():.4f}")
        
        return loss
        
    except Exception as e:
        logger.error("\nUnexpected error in loss calculation:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        
        # Get the full stack trace
        tb = traceback.format_exc()
        logger.error("\nFull traceback:")
        for line in tb.split('\n'):
            logger.error(line)
            
        # Log tensor states
        logger.error("\nTensor states at error:")
        try:
            logger.error(f"x_0: shape={x_0.shape}, dtype={x_0.dtype}, device={x_0.device}")
            logger.error(f"sigma: shape={sigma.shape}, dtype={sigma.dtype}, device={sigma.device}")
            logger.error(f"text_embeddings: shape={text_embeddings.shape}, dtype={text_embeddings.dtype}, device={text_embeddings.device}")
            if 'model_output' in locals():
                logger.error(f"model_output: shape={model_output.shape}, dtype={model_output.dtype}, device={model_output.device}")
            if 'target' in locals():
                logger.error(f"target: shape={target.shape}, dtype={target.dtype}, device={target.device}")
        except Exception as debug_error:
            logger.error(f"Error while logging tensor states: {str(debug_error)}")
        
        raise RuntimeError("Loss calculation failed - see logs for details") from e

class PerceptualLoss:
    def __init__(self, device="cuda"):
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = self.vgg.eval().to(device).to(torch.float32)
        self.vgg.requires_grad_(False)
        
        self.layers = {
            '3': 'relu1_2',
            '8': 'relu2_2', 
            '15': 'relu3_3',
            '22': 'relu4_3'
        }
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def get_features(self, x):
        x = self.normalize(x)
        features = {}
        
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
                
        return features

    def __call__(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)
        
        loss = 0.0
        for key in pred_features:
            loss += F.mse_loss(
                pred_features[key],
                target_features[key],
                reduction='mean'
            )
            
        return loss

def get_resolution_dependent_sigma_max(height, width):
    """Calculate resolution-dependent maximum sigma value with conservative scaling"""
    base_res = 1024 * 1024
    current_res = height * width
    # More conservative scaling for higher resolutions
    scale_factor = (current_res / base_res) ** 0.5  # Changed from 0.5 to 0.25
    base_sigma_max = 20000.0  # Practical infinity for ZTSNR
    return base_sigma_max * scale_factor