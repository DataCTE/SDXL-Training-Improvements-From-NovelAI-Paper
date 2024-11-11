import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models

logger = logging.getLogger(__name__)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, height=1024, width=1024):
    """
    Generate sigmas using a schedule that supports Zero Terminal SNR (ZTSNR)
    Args:
        num_inference_steps: Number of inference steps
        sigma_min: Minimum sigma value (≈0.0292 from paper)
        height: Image height for resolution-dependent sigma_max
        width: Image width for resolution-dependent sigma_max
    Returns:
        Tensor of sigma values
    """
    # Calculate resolution-dependent sigma_max
    sigma_max = get_resolution_dependent_sigma_max(height, width)
    
    rho = 7.0  # Use rho=7.0 as specified in the paper
    t = torch.linspace(1, 0, num_inference_steps)
    # Karras schedule with ZTSNR modifications
    sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigmas

def v_prediction_scaling_factors(sigma, sigma_data=1.0):
    """Compute scaling factors for v-prediction with improved stability"""
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    
    # Modified scaling factors with better numerical stability
    c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2 + eps)
    c_out = (-sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2 + eps)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2 + eps)
    
    return c_skip, c_out, c_in

def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs=None, sigma_data=1.0):
    try:
        # Get model dtype
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        
        # Convert inputs to model dtype
        x_0 = x_0.to(dtype=dtype)
        sigma = sigma.to(dtype=dtype)
        text_embeddings = text_embeddings.to(dtype=dtype)
        
        if added_cond_kwargs is not None:
            added_cond_kwargs = {
                k: v.to(dtype=dtype) if torch.is_tensor(v) else v
                for k, v in added_cond_kwargs.items()
            }

        # Scale sigma based on resolution as per Section 2.3
        _, _, height, width = x_0.shape
        total_pixels = height * width
        base_res = 1024 * 1024
        
        # Paper's resolution scaling with stability cap
        scale_factor = min((total_pixels / base_res) ** 0.5, 4.0)
        
        # Set sigma range as per Section 2.2 with practical upper bound
        sigma_min = 0.002
        sigma_max = 20000.0  # Practical approximation of inf for ZTSNR
        sigma = torch.clamp(sigma * scale_factor, sigma_min, sigma_max)
        
        # Generate noise with explicit dtype
        noise = torch.randn_like(x_0, dtype=dtype)
        
        # Create noisy input
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get v-prediction
        v_pred = model(x_t, sigma, text_embeddings, added_cond_kwargs=added_cond_kwargs).sample
        
        # v-prediction scaling factors with numerical stability
        eps = 1e-8  # Small epsilon to prevent division by zero
        c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2 + eps)
        c_out = (-sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2 + eps)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2 + eps)
        
        # Scale prediction and target
        scaled_output = c_out.view(-1, 1, 1, 1) * v_pred
        v_target = c_in.view(-1, 1, 1, 1) * noise
        
        # MinSNR loss weighting with stability bounds
        snr = (sigma_data / (sigma + eps)) ** 2
        snr = torch.clamp(snr, min=1e-5, max=1e5)  # Prevent extreme values
        min_snr_gamma = 1.0  # As per paper
        loss_weight = torch.minimum(snr, torch.tensor(min_snr_gamma, device=snr.device, dtype=snr.dtype))
        
        # Calculate weighted MSE loss
        mse_loss = F.mse_loss(scaled_output, v_target, reduction='none')
        weighted_loss = mse_loss * loss_weight.view(-1, 1, 1, 1)
        loss = weighted_loss.mean()
        
        # More aggressive stability checks
        if (not torch.isfinite(loss) or 
            loss < 1e-6 or  # Increased threshold
            torch.abs(v_pred).max() > 1e3 or  # Check for extreme predictions
            torch.abs(weighted_loss).max() > 1e3):  # Check for extreme loss values
            logger.warning(f"Skipping batch - Loss: {loss.item()}, Max v_pred: {torch.abs(v_pred).max().item()}, Max weighted_loss: {torch.abs(weighted_loss).max().item()}")
            return None, None
            
        # Collect metrics
        loss_metrics = {
            'loss/total': loss.item(),
            'loss/mse_raw': mse_loss.mean().item(),
            'loss/weight_mean': loss_weight.mean().item(),
            'model/sigma_mean': sigma.mean().item(),
            'model/sigma_std': sigma.std().item(),
            'model/v_pred_norm': v_pred.norm().item(),
        }
        
        return loss, loss_metrics

    except Exception as e:
        logger.error(f"Error in loss calculation: {str(e)}")
        logger.error(f"Shapes: x_0={x_0.shape}, sigma={sigma.shape}")
        logger.error(f"Values: sigma={sigma}, loss_weight={loss_weight}")
        raise

def get_resolution_dependent_sigma_max(height, width):
    """Scale sigma_max based on image resolution as described in section 2.3"""
    base_res = 1024 * 1024  # Base resolution
    current_res = height * width
    scale_factor = (current_res / base_res) ** 0.5
    return 20000.0 * scale_factor  # Base sigma_max * scale factor

class PerceptualLoss:
    def __init__(self):
        # Use pre-trained VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to("cuda")
        # Convert VGG to bfloat16
        self.vgg = self.vgg.to(dtype=torch.bfloat16)
        self.vgg.requires_grad_(False)
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def get_features(self, x):
        # Ensure input is in bfloat16
        x = x.to(dtype=torch.bfloat16)
        x = self.normalize(x)
        features = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def __call__(self, pred, target):
        # Ensure inputs are in bfloat16
        pred = pred.to(dtype=torch.bfloat16)
        target = target.to(dtype=torch.bfloat16)
        
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)

        loss = 0.0
        for key in pred_features:
            loss += F.mse_loss(pred_features[key], target_features[key])
        return loss