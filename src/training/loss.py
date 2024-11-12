import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models

logger = logging.getLogger(__name__)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, height=1024, width=1024):
    """
    Generate sigmas for ZTSNR with resolution-dependent scaling using 28-step native schedule
    """
    # Calculate resolution-dependent sigma_max
    base_res = 1024 * 1024
    current_res = height * width
    scale_factor = (current_res / base_res) ** 0.5
    
    # Use 20000 as infinity approximation for ZTSNR
    sigma_max = 20000.0 * scale_factor
    
    # Uniform linear spacing over 1000-step ZTSNR schedule
    # but sampled at 28 points for efficiency
    t = torch.linspace(0, 1, num_inference_steps)
    
    # Linear interpolation in log-space
    sigmas = torch.exp(t * torch.log(torch.tensor(sigma_min)) + 
                      (1-t) * torch.log(torch.tensor(sigma_max)))
    
    return sigmas

def v_prediction_scaling_factors(sigma, sigma_data=1.0):
    """
    Compute v-prediction scaling factors according to paper equations (12), (13)
    """
    c_out = (-sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return c_out, c_in

def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs=None, sigma_data=1.0):
    """
    Calculate v-prediction loss with MinSNR weighting
    """
    try:
        # Get model dtype and device
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        
        # Convert inputs to model dtype
        x_0 = x_0.to(dtype=dtype)
        sigma = sigma.to(dtype=dtype)
        text_embeddings = text_embeddings.to(dtype=dtype)
        
        # Handle conditional inputs
        if added_cond_kwargs is not None:
            added_cond_kwargs = {
                k: v.to(dtype=dtype) if torch.is_tensor(v) else v
                for k, v in added_cond_kwargs.items()
            }

        # Generate noise
        noise = torch.randn_like(x_0, dtype=dtype)
        
        # Create noisy input
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get model prediction
        v_pred = model(x_t, sigma, text_embeddings, added_cond_kwargs=added_cond_kwargs).sample
        
        # Calculate scaling factors
        c_skip, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)


        # Keep these for the loss calculation:
        scaled_output = c_out.view(-1, 1, 1, 1) * v_pred
        v_target = c_in.view(-1, 1, 1, 1) * noise
        
        # Calculate SNR for loss weighting
        snr = (sigma_data / sigma) ** 2
        min_snr_gamma = 5.0
        loss_weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=device, dtype=dtype))
        
        # Calculate weighted loss
        mse_loss = F.mse_loss(scaled_output, v_target, reduction='none')
        weighted_loss = mse_loss * loss_weight.view(-1, 1, 1, 1)
        loss = weighted_loss.mean()
        
        # Collect metrics for monitoring
        loss_metrics = {
            'loss/total': loss.item(),
            'loss/mse_raw': mse_loss.mean().item(),
            'loss/weight_mean': loss_weight.mean().item(),
            'model/sigma_mean': sigma.mean().item(),
            'model/sigma_std': sigma.std().item(),
            'model/v_pred_norm': v_pred.norm().item(),
            'model/snr_mean': snr.mean().item()
        }
        
        return loss, loss_metrics

    except Exception as e:
        logger.error(f"Error in loss calculation: {str(e)}")
        logger.error(f"Shapes: x_0={x_0.shape}, sigma={sigma.shape}")
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
    """Calculate resolution-dependent maximum sigma value"""
    base_res = 1024 * 1024
    current_res = height * width
    # Scale based on total pixels with EDM-style adjustment
    scale_factor = (current_res / base_res) ** 0.5
    base_sigma_max = 20000.0  # Approximate infinity for ZTSNR
    return base_sigma_max * scale_factor