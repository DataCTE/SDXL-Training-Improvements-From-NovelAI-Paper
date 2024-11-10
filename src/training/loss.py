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
    """Compute scaling factors for v-prediction as described in paper section 2.1"""
    # α_t = 1/√(1 + σ²) from paper
    alpha_t = 1 / torch.sqrt(1 + sigma**2)
    
    # Scaling factors from paper appendix A.1
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = -sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return alpha_t, c_skip, c_out, c_in

def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs):
    """Training loss using v-prediction with MinSNR weighting"""
    try:
        # Get image dimensions for sigma scaling
        _, _, height, width = x_0.shape
        
        # Scale sigma based on resolution if not already scaled
        if sigma.ndim == 1:  # Only scale if sigma hasn't been scaled yet
            sigma_max = get_resolution_dependent_sigma_max(height, width)
            sigma = sigma * (sigma_max / 20000.0)  # Scale relative to base sigma_max
        
        # Validate text embeddings shape
        expected_embed_dim = 2048  # SDXL context dimension
        if text_embeddings.shape[-1] != expected_embed_dim:
            logger.warning(f"Unexpected text embedding dimension: {text_embeddings.shape[-1]}, expected {expected_embed_dim}")

        # Handle additional conditioning
        if "time_ids" in added_cond_kwargs:
            time_ids = added_cond_kwargs["time_ids"]
            expected_time_dim = 2816
            if time_ids.shape[1] != expected_time_dim:
                logger.info(f"Padding time_ids from {time_ids.shape[1]} to {expected_time_dim}")
                # Reshape time_ids to match expected dimensions
                time_ids = time_ids.view(time_ids.shape[0], -1)  # Flatten except batch dim
                if time_ids.shape[1] < expected_time_dim:
                    # Pad if needed
                    time_ids = F.pad(time_ids, (0, expected_time_dim - time_ids.shape[1]))
                else:
                    # Truncate if too long
                    time_ids = time_ids[:, :expected_time_dim]
                added_cond_kwargs["time_ids"] = time_ids

        # Generate noise and noisy input
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Calculate scaling factors
        alpha_t = 1 / torch.sqrt(1 + sigma**2)
        
        # Get model prediction
        v_pred = model(
            x_t,
            sigma,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        # Calculate loss
        target = noise
        loss = F.mse_loss(v_pred, target, reduction="none")
        loss = loss.mean([1, 2, 3])
        loss = loss.mean()
        
        # Add resolution info to metrics
        loss_metrics = {
            'loss/current': loss.item(),
            'model/alpha_t_mean': alpha_t.mean().item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item(),
            'resolution/height': height,
            'resolution/width': width,
            'resolution/sigma_max': sigma_max.item() if torch.is_tensor(sigma_max) else sigma_max
        }
        
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        # Log additional debugging info
        logger.error(f"Input shapes:")
        logger.error(f"x_0: {x_0.shape}")
        logger.error(f"sigma: {sigma.shape}")
        logger.error(f"text_embeddings: {text_embeddings.shape}")
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