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
                             sigma_data=1.0, tag_weighter=None, batch_tags=None):
    """
    Calculate v-prediction loss with MinSNR weighting and optimized performance
    """
    try:
        # Optimize device and dtype handling
        dtype = x_0.dtype
        device = x_0.device
        
        # Ensure consistent dtype for all inputs
        sigma = sigma.to(dtype=dtype, device=device)
        text_embeddings = text_embeddings.to(dtype=dtype, device=device)
        
        # Preprocess conditional inputs with performance in mind
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        
        # Handle dual text encoders (CLIP ViT-L and OpenCLIP ViT-bigG)
        if 'time_ids' in added_cond_kwargs:
            time_ids = added_cond_kwargs['time_ids']
            # Get text embedding dimensions for both encoders
            batch_size = text_embeddings.size(0)
            seq_len = text_embeddings.size(1) if text_embeddings.dim() > 2 else 1
            hidden_dim = text_embeddings.size(-1)
            
            # Reshape time_ids to match concatenated embeddings
            if time_ids.dim() == 1:
                # Single time embedding -> expand for both encoders
                time_ids = time_ids.view(batch_size, 1)
                time_ids = time_ids.unsqueeze(-1).expand(-1, seq_len, -1)
            elif time_ids.dim() == 2:
                # Already batch x sequence, ensure proper sequence length
                if time_ids.size(1) == 1:
                    time_ids = time_ids.expand(-1, seq_len)
            elif time_ids.dim() == 4:
                # Handle spatial time embeddings
                time_ids = time_ids.view(batch_size, -1)
                time_ids = time_ids.unsqueeze(1).expand(-1, seq_len, -1)
            
            added_cond_kwargs['time_ids'] = time_ids

        # Use torch.no_grad for inference to reduce memory overhead
        with torch.no_grad():
            # Generate noise more efficiently
            noise = torch.randn_like(x_0, dtype=dtype, device=device)
            
            # Create noisy input with less memory allocation
            x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
            
            # Compute scaling factors with less overhead
            c_skip, c_out, c_in = v_prediction_scaling_factors(sigma, sigma_data)
            
            # Predict with minimal overhead
            v_pred = model(x_t, sigma, text_embeddings, added_cond_kwargs=added_cond_kwargs).sample
            
            # Scale prediction and target more efficiently
            scaled_output = c_skip.view(-1, 1, 1, 1) * x_t + c_out.view(-1, 1, 1, 1) * v_pred
            v_target = c_in.view(-1, 1, 1, 1) * noise
            
            # Compute SNR with tensor operations (section 2.4)
            snr = (sigma_data / sigma) ** 2
            min_snr_gamma = 5.0  # As recommended in the paper
            
            # MinSNR loss weighting with better ZTSNR handling
            loss_weight = torch.where(
                snr > min_snr_gamma,
                torch.ones_like(snr) * min_snr_gamma,
                snr
            )
            
            # Compute loss with reduced memory allocations
            mse_loss = F.mse_loss(scaled_output, v_target, reduction='none')
            weighted_loss = mse_loss * loss_weight.view(-1, 1, 1, 1)
            
            # Optional tag-based weighting with early exit if no weighting
            if tag_weighter is not None and batch_tags is not None:
                batch_weights = []
                for tags, weights, special in zip(batch_tags['tags'], 
                                               batch_tags['tag_weights'],
                                               batch_tags['special_tags']):
                    weight = tag_weighter.calculate_weights(tags, weights, special)
                    batch_weights.append(weight)
                
                tag_weights = torch.tensor(batch_weights, device=device, dtype=dtype)
                weighted_loss = weighted_loss * tag_weights.view(-1, 1, 1, 1)
            
            # Compute final loss
            loss = weighted_loss.mean()
            
            # Collect metrics with minimal overhead
            loss_metrics = {
                'loss/total': loss.item(),
                'loss/mse_raw': mse_loss.mean().item(),
                'loss/weight_mean': loss_weight.mean().item(),
                'model/sigma_mean': sigma.mean().item(),
                'model/sigma_std': sigma.std().item(),
                'model/v_pred_norm': v_pred.norm().item(),
                'model/snr_mean': snr.mean().item()
            }
            
            # Efficiently compute tag metrics
            if tag_weighter is not None and batch_tags is not None:
                loss_metrics.update({
                    'tags/weight_mean': torch.tensor(batch_weights).mean().item() if batch_weights else 0,
                    'tags/weight_std': torch.tensor(batch_weights).std().item() if batch_weights else 0,
                    'tags/niji_count': sum(1 for t in batch_tags['special_tags'] if t.get('niji', False)),
                    'tags/quality_6_count': sum(1 for t in batch_tags['special_tags'] if t.get('quality', 0) == 6),
                    'tags/stylize_mean': np.mean([t.get('stylize', 0) for t in batch_tags['special_tags']]),
                    'tags/chaos_mean': np.mean([t.get('chaos', 0) for t in batch_tags['special_tags']])
                })
            
            return loss, loss_metrics

    except Exception as e:
        logger.error(f"Error in training_loss_v_prediction: {str(e)}\n{traceback.format_exc()}")
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