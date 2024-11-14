import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models
import numpy as np

logger = logging.getLogger(__name__)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, height=1024, width=1024):
    """
    Generate sigmas for ZTSNR with resolution-dependent scaling using 28-step native schedule
    Modified for better noise control and ZTSNR support
    """
    # Calculate resolution-dependent sigma_max with quadratic scaling (section 2.3)
    base_res = 1024 * 1024
    current_res = height * width
    scale_factor = (current_res / base_res)  # Quadratic scaling for redundant signal
    
    # Use 20000 as practical infinity approximation for ZTSNR (appendix A.2)
    sigma_max = 20000.0 * scale_factor
    
    # Use non-linear spacing with EDM-style karras schedule
    rho = 7.0  # EDM's recommended value
    t = torch.linspace(0, 1, num_inference_steps)
    inv_rho = 1.0 / rho
    
    # Log-space interpolation with sigma conversion
    sigmas = (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    
    return sigmas

def v_prediction_scaling_factors(sigma, sigma_data=1.0):
    """
    Compute v-prediction scaling factors according to paper equations (11)-(13)
    """
    # Skip factor for zero terminal SNR case
    c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
    
    # Output scaling with negative sign for v-prediction
    c_out = (-sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)
    
    # Input scaling for proper normalization
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return c_skip, c_out, c_in

def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs=None, 
                             sigma_data=1.0, tag_weighter=None, batch_tags=None, min_snr_gamma=5.0):
    """
    Calculate v-prediction loss with MinSNR weighting and optimized performance
    Args:
        min_snr_gamma: Minimum SNR value for loss weighting (default: 5.0 from paper)
    """
    try:
        dtype = x_0.dtype
        device = x_0.device
        
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
        
        # Forward pass
        model_output = model(
            model_input,
            sigma,
            text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        )
        
        # Calculate target
        target = c_in * x_0
        
        # Calculate SNR for loss weighting
        snr = (sigma_data / sigma) ** 2
        min_snr = torch.full_like(snr, min_snr_gamma)
        loss_weights = (snr / min_snr).clamp(max=1.0)
        
        # Calculate weighted MSE loss
        loss = F.mse_loss(model_output, target, reduction='none')
        loss = loss.mean(dim=(1, 2, 3))
        loss = (loss * loss_weights).mean()
        
        # Apply tag weighting if provided
        if tag_weighter is not None and batch_tags is not None:
            tag_weights = tag_weighter(batch_tags)
            loss = loss * tag_weights.mean()
        
        return loss
    
    except Exception as e:
        logger.error(f"Error in v-prediction loss calculation: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

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