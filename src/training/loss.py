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
        # Validate input dimensions
        batch_size, channels, height, width = x_0.shape
        embed_dim = text_embeddings.shape[-1]  # Get just the embedding dimension
        
        # Validate text embedding dimensions
        if embed_dim != 2048:
            # Check if we need to concatenate embeddings
            if embed_dim == 768:
                logger.warning("Got 768-dim embeddings, expecting concatenated SDXL embeddings (2048-dim)")
                logger.warning("Please ensure both text encoders' outputs are being concatenated")
                raise ValueError(
                    f"Text embedding context dimension ({embed_dim}) must be 2048 for SDXL. "
                    "Make sure to concatenate both text encoders' outputs."
                )
            else:
                raise ValueError(
                    f"Unexpected embedding dimension: {embed_dim}. "
                    "SDXL requires 2048-dim embeddings (768 from encoder 1 + 1280 from encoder 2)"
                )
        
        # Get latent dimensions and validate
        batch_size, channels, height, width = x_0.shape
        
        # Validate channel dimension (SDXL uses 4 channels in latent space)
        if channels != 4:
            raise ValueError(
                f"Input must have 4 channels for SDXL latent space. "
                f"Got {channels} channels. Shape: {x_0.shape}"
            )
            
        # Validate UNet architecture requirements (in latent space)
        min_size = 32  # 256 pixels / 8 (VAE scaling)
        max_size = 256  # 2048 pixels / 8 (VAE scaling)
        
        if height < min_size or width < min_size:
            raise ValueError(
                f"Latent dimensions too small. Minimum size is {min_size}x{min_size} "
                f"(256x256 in pixel space). Got {height}x{width}"
            )
            
        if height > max_size or width > max_size:
            raise ValueError(
                f"Latent dimensions too large. Maximum size is {max_size}x{max_size} "
                f"(2048x2048 in pixel space). Got {height}x{width}"
            )
        
        # Calculate total pixels in latent space
        total_pixels = height * width
        
        # Check aspect ratio but don't raise error
        aspect_ratio = width / height
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:  # Values from SDXL paper
            logger.warning(
                f"Batch contains latents with aspect ratio ({aspect_ratio:.2f}) "
                "outside supported range (0.25 to 4.0). Skipping batch."
            )
            # Return None to indicate batch should be skipped
            return None, None
        
        # Validate text embedding context dimension (SDXL requirement)
        if text_embeddings.shape[-1] != 2048:
            raise ValueError(
                f"Text embedding context dimension ({text_embeddings.shape[-1]}) must be 2048 for SDXL"
            )
            
        # Validate batch dimensions match
        if sigma.ndim == 1 and len(sigma) != batch_size:
            raise ValueError(
                f"Sigma length ({len(sigma)}) must match batch size ({batch_size})"
            )
            
        if text_embeddings.shape[0] != batch_size:
            raise ValueError(
                f"Text embedding batch size ({text_embeddings.shape[0]}) must match image batch size ({batch_size})"
            )
        
        # Ensure sigma is in same dtype as model
        sigma = sigma.to(dtype=x_0.dtype)
        
        # Scale sigma based on resolution as per paper section 2.3
        if sigma.ndim == 1:
            base_res = 1024 * 1024  # Base resolution from paper
            scale_factor = (total_pixels / base_res) ** 0.5
            sigma_max = 20000.0 * scale_factor  # Using practical ZTSNR approximation
            sigma = sigma * (sigma_max / 20000.0)
        
        # Generate noise and normalize it
        noise = torch.randn_like(x_0)
        noise = noise / (noise.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8)
        
        # Create noisy input with normalized noise
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Get scaling factors (we'll only use c_out and c_in)
        _, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)
        
        # Get model prediction
        v_pred = model(
            x_t,
            sigma,
            text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Normalize prediction before clamping
        v_pred = v_pred / (v_pred.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8)
        v_pred = torch.clamp(v_pred, -1.0, 1.0)  # Clamp to [-1,1] after normalization
        
        # Scale output and compute target (using unnormalized noise)
        scaled_output = c_out.view(-1, 1, 1, 1) * v_pred 
        v_target = c_in.view(-1, 1, 1, 1) * noise

        # Modified SNR weighting with paper-specified values
        snr = torch.clamp((sigma_data / (sigma + 1e-8)) ** 2, 1e-5, 1e2)
        min_snr = 1.0  # As per NovelAI V3 paper
        snr_clipped = torch.minimum(snr, torch.tensor(min_snr))
        loss_weight = torch.clamp(snr_clipped / snr, 0.1, 10.0)

        # Calculate loss with stability improvements
        mse_loss = F.mse_loss(scaled_output, v_target, reduction='none')
        weighted_loss = (mse_loss * loss_weight.view(-1, 1, 1, 1))
        loss = weighted_loss.mean()

        # Add L2 regularization to prevent extreme predictions
        l2_reg = 1e-4 * (v_pred ** 2).mean()
        loss = loss + l2_reg
        
        # Collect metrics
        loss_metrics = {
            # Loss metrics
            'loss/mse_mean': loss.item(),
            'loss/mse_std': torch.nn.functional.mse_loss(scaled_output, v_target).std().item(),
            'loss/snr_mean': snr.mean().item(),
            'loss/min_snr_gamma_mean': loss_weight.mean().item(),
            
            # Model metrics
            'model/v_pred_std': v_pred.std().item(),
            'model/v_target_std': v_target.std().item(),
            'model/alpha_t_mean': (1 / torch.sqrt(1 + sigma**2)).mean().item(),
            
            # Noise metrics
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item(),
        }
        
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error(traceback.format_exc())
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