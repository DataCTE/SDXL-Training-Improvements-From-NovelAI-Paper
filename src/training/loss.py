import torch
import torch.nn.functional as F
import logging
import traceback
from torchvision import transforms, models

logger = logging.getLogger(__name__)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, sigma_max=20000.0):
    """
    Generate sigmas using a schedule that supports Zero Terminal SNR (ZTSNR)
    Args:
        num_inference_steps: Number of inference steps
        sigma_min: Minimum sigma value (≈0.0292 from paper)
        sigma_max: Maximum sigma value (set to 20000 for practical ZTSNR)
    Returns:
        Tensor of sigma values
    """
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
    """
    Training loss using v-prediction with MinSNR weighting (sections 2.1 and 2.4)
    
    Args:
        model (UNet2DConditionModel): The model being trained
        x_0 (tensor): [B, 4, H, W] latent representations
        sigma (tensor): [B] noise levels
        text_embeddings (tensor): [B, 77, 2048] Combined CLIP embeddings
        added_cond_kwargs (dict): Additional conditioning including:
            - text_embeds: [B, 1280] Pooled embeddings
            - time_ids: [B, 6] Time embedding IDs
    
    Returns:
        tuple: (loss, metrics_dict)
    """
    try:
        logger.debug("\n=== Starting v-prediction training loss calculation ===")
        logger.debug("Initial input shapes and values:")
        logger.debug(f"x_0: shape={x_0.shape}, dtype={x_0.dtype}, device={x_0.device}")
        logger.debug(f"sigma: shape={sigma.shape}, dtype={sigma.dtype}, range=[{sigma.min():.6f}, {sigma.max():.6f}]")
        logger.debug(f"text_embeddings: shape={text_embeddings.shape}, dtype={text_embeddings.dtype}")
        
        # Get noise and scaling factors
        logger.debug("\nGenerating noise and computing scaling factors:")
        noise = torch.randn_like(x_0)
        logger.debug(f"noise: shape={noise.shape}, dtype={noise.dtype}, std={noise.std():.6f}")
        
        alpha_t, c_skip, c_out, c_in = v_prediction_scaling_factors(sigma)
        logger.debug(f"alpha_t: range=[{alpha_t.min():.6f}, {alpha_t.max():.6f}]")
        logger.debug(f"c_skip: range=[{c_skip.min():.6f}, {c_skip.max():.6f}]")
        logger.debug(f"c_out: range=[{c_out.min():.6f}, {c_out.max():.6f}]")
        logger.debug(f"c_in: range=[{c_in.min():.6f}, {c_in.max():.6f}]")
        
        # Compute noisy sample x_t = x_0 + σε
        logger.debug("\nComputing noisy sample:")
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        logger.debug(f"x_t: shape={x_t.shape}, range=[{x_t.min():.6f}, {x_t.max():.6f}]")
        
        # Compute v-target = α_t * ε - (1 - α_t) * x_0
        logger.debug("\nComputing v-target:")
        v_target = alpha_t.view(-1, 1, 1, 1) * noise - (1 - alpha_t).view(-1, 1, 1, 1) * x_0
        logger.debug(f"v_target: shape={v_target.shape}, range=[{v_target.min():.6f}, {v_target.max():.6f}]")
        
        # Get model prediction
        logger.debug("\nGetting model prediction:")
        v_pred = model(
            x_t,
            sigma,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        logger.debug(f"v_pred: shape={v_pred.shape}, range=[{v_pred.min():.6f}, {v_pred.max():.6f}]")
        
        # MinSNR weighting
        logger.debug("\nComputing MinSNR weights:")
        snr = 1 / (sigma**2)  # SNR = 1/σ²
        gamma = 1.0  # SNR clipping value
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        logger.debug(f"SNR: range=[{snr.min():.6f}, {snr.max():.6f}]")
        logger.debug(f"min_snr_gamma: range=[{min_snr_gamma.min():.6f}, {min_snr_gamma.max():.6f}]")
        
        # Compute weighted MSE loss
        logger.debug("\nComputing final loss:")
        mse_loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = (min_snr_gamma.view(-1, 1, 1, 1) * mse_loss).mean()
        
        # Collect detailed metrics
        loss_metrics = {
            'loss/total': loss.item(),
            'loss/mse_mean': mse_loss.mean().item(),
            'loss/mse_std': mse_loss.std().item(),
            'loss/snr_mean': snr.mean().item(),
            'loss/min_snr_gamma_mean': min_snr_gamma.mean().item(),
            'model/v_pred_std': v_pred.std().item(),
            'model/v_target_std': v_target.std().item(),
            'model/alpha_t_mean': alpha_t.mean().item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item()
        }
        
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise

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