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
    """Training loss using v-prediction with MinSNR weighting as described in NovelAI V3 paper"""
    try:
        # Get model's dtype for consistency
        dtype = next(model.parameters()).dtype
        
        # Convert inputs to model's dtype
        x_0 = x_0.to(dtype=dtype)
        sigma = sigma.to(dtype=dtype)
        text_embeddings = text_embeddings.to(dtype=dtype)
        
        # Convert added_cond_kwargs to model's dtype
        if added_cond_kwargs is not None:
            added_cond_kwargs = {
                k: v.to(dtype=dtype) if torch.is_tensor(v) else v
                for k, v in added_cond_kwargs.items()
            }
        
        # Generate noise with matching dtype
        noise = torch.randn_like(x_0, dtype=dtype)
        
        # Add gradient scaling for numerical stability
        scale_factor = min((x_0.shape[-1] * x_0.shape[-2] / (1024 * 1024)) ** 0.5, 4.0)  # Cap scaling
        sigma = sigma * scale_factor
        
        # Create noisy input
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get model prediction
        v_pred = model(
            x_t,
            sigma,
            text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Channel-wise normalization for stability (as per paper)
        B, C, H, W = v_pred.shape
        v_pred = v_pred.view(B, C, -1)
        v_pred = v_pred / (v_pred.norm(dim=-1, keepdim=True) + 1e-8)
        v_pred = v_pred.view(B, C, H, W)
        
        # Conservative clamping (as per paper's practical implementation)
        v_pred = torch.clamp(v_pred, -0.5, 0.5)  # Reduced range for stability
        
        # Improved v-prediction scaling factors
        c_skip, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)
        
        # Scale prediction and target
        scaled_output = c_out.view(-1, 1, 1, 1) * v_pred
        v_target = c_in.view(-1, 1, 1, 1) * noise
        
        # Improved MinSNR weighting with better numerical stability
        snr = (sigma_data / (sigma + 1e-8)) ** 2
        snr = torch.clamp(snr, min=1e-5, max=1e2)  # Tighter bounds
        min_snr = 1.0
        snr_clipped = torch.minimum(snr, torch.tensor(min_snr, device=snr.device, dtype=snr.dtype))
        loss_weight = torch.clamp(snr_clipped / snr, min=0.1, max=10.0)
        
        # Calculate loss with improved stability
        mse_loss = F.mse_loss(scaled_output, v_target, reduction='none')
        weighted_loss = mse_loss * loss_weight.view(-1, 1, 1, 1)
        loss = weighted_loss.mean()
        
        # Add gradient norm clipping threshold
        max_grad_norm = 1.0
        if loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Skip if loss is too small
        if loss.item() < 1e-8:
            logger.warning(f"Skipping batch due to very small loss: {loss.item()}")
            return None, None
            
        # Add L2 regularization to prevent extreme predictions
        l2_reg = 1e-4 * (v_pred ** 2).mean()
        loss = loss + l2_reg
        
        # Collect detailed metrics for monitoring
        loss_metrics = {
            'loss/mse_mean': loss.item(),
            'loss/mse_std': torch.nn.functional.mse_loss(scaled_output, v_target).std().item(),
            'loss/snr_mean': snr.mean().item(),
            'loss/min_snr_gamma_mean': loss_weight.mean().item(),
            'model/v_pred_std': v_pred.std().item(),
            'model/v_target_std': v_target.std().item(),
            'model/alpha_t_mean': (1 / torch.sqrt(1 + sigma**2)).mean().item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item(),
        }
        
        # Skip batch if loss is unstable
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e5:
            logger.warning(f"Skipping batch due to unstable loss: {loss.item()}")
            return None, None  # Return None to indicate batch should be skipped
            
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Input shapes and dtypes:")
        logger.error(f"x_0: {x_0.shape}, {x_0.dtype}")
        logger.error(f"sigma: {sigma.shape}, {sigma.dtype}")
        logger.error(f"text_embeddings: {text_embeddings.shape}, {text_embeddings.dtype}")
        logger.error(f"added_cond_kwargs: {added_cond_kwargs}")
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