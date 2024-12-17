import torch
import logging
from diffusers import DDPMScheduler
from src.config.config import Config
import traceback
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def get_karras_scalings(sigmas: torch.Tensor, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Karras noise schedule scalings for given timesteps."""
    timestep_sigmas = sigmas[timestep_indices]
    c_skip = 1 / (timestep_sigmas**2 + 1).sqrt()
    c_out = -timestep_sigmas / (timestep_sigmas**2 + 1).sqrt()
    c_in = 1 / (timestep_sigmas**2 + 1).sqrt()
    return c_skip, c_out, c_in

def get_sigmas(config: Config, device: torch.device) -> torch.Tensor:
    """Generate noise schedule for ZTSNR with optimized scaling."""
    num_timesteps = config.model.num_timesteps
    sigma_min = config.model.sigma_min
    sigma_max = config.model.sigma_max
    rho = config.model.rho
    
    ramp = torch.linspace(0, 1, num_timesteps, device=device)
    min_inv_rho = sigma_min ** (1/rho)
    max_inv_rho = sigma_max ** (1/rho)
    
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas[0] = sigma_min  # First step
    sigmas[-1] = sigma_max  # ZTSNR step
    
    return sigmas

def get_scheduler_parameters(sigmas: torch.Tensor, config: Config, device: torch.device) -> Dict[str, Any]:
    """Compute all scheduler parameters from sigmas."""
    alphas = 1 / (sigmas**2 + 1)
    betas = 1 - alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    snr_values = 1 / (sigmas ** 2)
    
    # Compute SNR weights if needed
    snr_weights = None
    if config.training.snr_gamma is not None:
        snr_weights = torch.minimum(
            snr_values,
            torch.tensor(config.training.snr_gamma, device=device)
        ).float()
    
    # Compute scaling factors based on prediction type
    if config.training.prediction_type == "v_prediction":
        c_skip = 1 / (sigmas**2 + 1).sqrt()
        c_out = -sigmas / (sigmas**2 + 1).sqrt()
        c_in = 1 / (sigmas**2 + 1).sqrt()
    else:  # epsilon prediction
        c_skip = alphas_cumprod.sqrt()
        c_out = (1 - alphas_cumprod).sqrt()
        c_in = torch.ones_like(sigmas)
        
    return {
        'alphas': alphas,
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sigmas': sigmas,
        'snr_values': snr_values,
        'snr_weights': snr_weights,
        'c_skip': c_skip,
        'c_out': c_out,
        'c_in': c_in
    }

def configure_noise_scheduler(config: Config, device: torch.device) -> Dict[str, Any]:
    """Configure noise scheduler with Karras schedule and pre-compute training parameters."""
    try:
        # Initialize scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=config.model.num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=config.training.prediction_type,
            clip_sample=False,
            thresholding=False
        )
        
        # Generate sigmas and compute parameters
        sigmas = get_sigmas(config, device)
        params = get_scheduler_parameters(sigmas, config, device)
        
        # Update scheduler with computed values
        scheduler.alphas = params['alphas']
        scheduler.betas = params['betas']
        scheduler.alphas_cumprod = params['alphas_cumprod']
        scheduler.sigmas = sigmas
        scheduler.init_noise_sigma = sigmas.max()
        
        # Return all parameters including scheduler
        return {'scheduler': scheduler, **params}

    except Exception as e:
        logger.error(f"Failed to configure noise scheduler: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise