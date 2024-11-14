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
        model: UNet model
        x_0: Clean data (latents)
        sigma: Noise level
        text_embeddings: Text condition embeddings
        added_cond_kwargs: Additional conditioning for SDXL (time_ids, etc.)
        sigma_data: Data standard deviation (default: 1.0)
        tag_weighter: Optional tag weighting module
        batch_tags: Optional batch tag information
        min_snr_gamma: MinSNR gamma parameter (default: 5.0)
        verbose: Whether to log detailed information
    """
    try:
        # Add noise to input
        noise = torch.randn_like(x_0)
        noised = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get v-prediction scaling factors
        c_skip, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)
        
        # Scale model input
        model_input = c_in.view(-1, 1, 1, 1) * noised
        
        # Forward pass with proper conditioning
        model_output = model(
            model_input,
            sigma.view(-1, 1, 1, 1),
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Compute target
        target = c_out.view(-1, 1, 1, 1) * noise
        
        # Calculate weighted MSE loss with MinSNR
        snr = sigma_data**2 / sigma**2
        mse = (model_output - target).pow(2).mean(dim=(1, 2, 3))
        
        # Apply MinSNR weighting (section 2.4)
        snr_weight = (snr / min_snr_gamma).clamp(max=1.0)
        loss = (mse * snr_weight).mean()
        
        # Apply tag weighting if provided
        if tag_weighter is not None and batch_tags is not None:
            tag_weights = tag_weighter(batch_tags)
            loss = loss * tag_weights.mean()
        
        if verbose:
            logger.info(f"Loss components:")
            logger.info(f"- Base MSE: {mse.mean().item():.4e}")
            logger.info(f"- SNR weight: {snr_weight.mean().item():.4f}")
            if tag_weighter is not None:
                logger.info(f"- Tag weight: {tag_weights.mean().item():.4f}")
            logger.info(f"Final loss: {loss.item():.4e}")
        
        return loss
        
    except Exception as e:
        logger.error("Error in training_loss_v_prediction:")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise

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