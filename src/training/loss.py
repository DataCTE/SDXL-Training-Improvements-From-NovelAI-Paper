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
    """Training loss using v-prediction with MinSNR weighting as described in NovelAI V3 paper"""
    try:
        # Get image dimensions and validate
        _, _, height, width = x_0.shape
        
        # Validate resolution bounds (NovelAI V3 section 4.1)
        if height * width < 256 * 256 or height * width > 2048 * 2048:
            raise ValueError(f"Resolution ({height}x{width}) outside supported range (256x256 to 2048x2048)")
            
        # Validate text embedding context dimension (SDXL requirement)
        if text_embeddings.shape[-1] != 2048:
            raise ValueError(f"Text embedding context dimension ({text_embeddings.shape[-1]}) must be 2048")
            
        # Validate sigma shape matches batch dimension
        if sigma.ndim == 1 and len(sigma) != x_0.shape[0]:
            raise ValueError(f"Sigma length ({len(sigma)}) must match batch size ({x_0.shape[0]})")
            
        # Validate text embeddings shape
        if text_embeddings.shape[0] != x_0.shape[0]:
            raise ValueError(f"Text embedding batch size ({text_embeddings.shape[0]}) must match image batch size ({x_0.shape[0]})")
        
        # Validate aspect ratio is within supported range
        aspect_ratio = width / height
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:  # Values from SDXL paper
            raise ValueError(f"Aspect ratio ({aspect_ratio:.2f}) outside supported range (0.25 to 4.0)")
        
        # Scale sigma based on resolution as per paper section 2.3
        # "rule of thumb: if you double the canvas length (quadrupling the canvas area): 
        # you should double σmax (quadrupling the noise variance) to maintain SNR"
        if sigma.ndim == 1:
            base_res = 1024 * 1024  # Base resolution from paper
            current_res = height * width
            scale_factor = (current_res / base_res) ** 0.5
            sigma_max = 20000.0 * scale_factor  # Using practical ZTSNR approximation from paper appendix A.2
            sigma = sigma * (sigma_max / 20000.0)
        
        # Generate noise and noisy input
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        
        # Calculate v-prediction scaling factors
        sigma_data = 1.0  # Using σdata = 1.0 as per EDM formulation
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = -sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        
        # Scale model input
        model_input = c_in.view(-1, 1, 1, 1) * x_t
        
        # Get model prediction and apply scaling
        v_pred = model(model_input, sigma, text_embeddings, added_cond_kwargs)
        scaled_output = c_skip.view(-1, 1, 1, 1) * x_t + c_out.view(-1, 1, 1, 1) * v_pred
        
        # Calculate target
        v_target = c_in.view(-1, 1, 1, 1) * noise
        
        # Calculate SNR for MinSNR weighting as described in paper section 2.4
        snr = (sigma_data / sigma) ** 2
        min_snr = 1.0  # Minimum SNR threshold
        snr_clipped = torch.minimum(snr, torch.tensor(min_snr))
        loss_weight = snr_clipped / snr
        
        # Calculate weighted MSE loss
        loss = torch.nn.functional.mse_loss(scaled_output, v_target, reduction='none')
        loss = (loss * loss_weight.view(-1, 1, 1, 1)).mean()
        
        # Add metrics including MinSNR details
        loss_metrics = {
            'loss/current': loss.item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item(),
            'resolution/height': height,
            'resolution/width': width,
            'resolution/sigma_max': sigma_max.item() if torch.is_tensor(sigma_max) else sigma_max,
            'snr/current': snr.mean().item(),
            'snr/weight': loss_weight.mean().item()
        }
        
        # Validate model output shape
        if v_pred.shape != x_t.shape:
            raise ValueError(f"Model output shape ({v_pred.shape}) must match input shape ({x_t.shape})")
        
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