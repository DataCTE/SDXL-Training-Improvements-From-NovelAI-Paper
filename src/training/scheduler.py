import torch
import logging
from diffusers import DDPMScheduler
from src.config.config import Config
import traceback
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def get_karras_scalings(sigmas: torch.Tensor, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Karras noise schedule scalings for given timesteps."""
    try:
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError(f"Expected sigmas to be torch.Tensor, got {type(sigmas)}")
        if not isinstance(timestep_indices, torch.Tensor):
            raise ValueError(f"Expected timestep_indices to be torch.Tensor, got {type(timestep_indices)}")
            
        timestep_sigmas = sigmas[timestep_indices]
        c_skip = 1 / (timestep_sigmas**2 + 1).sqrt()
        c_out = -timestep_sigmas / (timestep_sigmas**2 + 1).sqrt()
        c_in = 1 / (timestep_sigmas**2 + 1).sqrt()
        return c_skip, c_out, c_in
        
    except Exception as e:
        logger.error(f"Error in get_karras_scalings: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def get_sigmas(config: Config, device: torch.device) -> torch.Tensor:
    """Generate noise schedule for ZTSNR with optimized scaling."""
    try:
        if not isinstance(config, Config):
            raise ValueError(f"Expected config to be Config, got {type(config)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"Expected device to be torch.device, got {type(device)}")
            
        num_timesteps = config.model.num_timesteps
        sigma_min = config.model.sigma_min
        sigma_max = config.model.sigma_max
        rho = config.model.rho
        
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}")
        
        # Create tensor on CPU first
        ramp = torch.linspace(0, 1, num_timesteps)
        min_inv_rho = sigma_min ** (1/rho)
        max_inv_rho = sigma_max ** (1/rho)
        
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas[0] = sigma_min  # First step
        sigmas[-1] = sigma_max  # ZTSNR step
        
        # Move to device at the end
        sigmas = sigmas.to(device=device)
        logger.info(f"Generated sigmas on device: {sigmas.device}")
        
        return sigmas
        
    except Exception as e:
        logger.error(f"Error in get_sigmas: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def get_scheduler_parameters(sigmas: torch.Tensor, config: Config, device: torch.device) -> Dict[str, Any]:
    """Compute all scheduler parameters from sigmas."""
    try:
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError(f"Expected sigmas to be torch.Tensor, got {type(sigmas)}")
        if not isinstance(config, Config):
            raise ValueError(f"Expected config to be Config, got {type(config)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"Expected device to be torch.device, got {type(device)}")
            
        # Move sigmas to device if needed
        sigmas = sigmas.to(device=device)
        
        # Compute parameters on device
        alphas = (1 / (sigmas**2 + 1)).to(device=device)
        betas = (1 - alphas).to(device=device)
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device=device)
        snr_values = (1 / (sigmas ** 2)).to(device=device)
        
        # Compute SNR weights if needed
        snr_weights = None
        if config.training.snr_gamma is not None:
            try:
                gamma_tensor = torch.tensor(config.training.snr_gamma, device=device)
                snr_weights = torch.minimum(snr_values, gamma_tensor).float()
            except RuntimeError as e:
                logger.error(f"Error computing SNR weights: {str(e)}")
                raise
        
        # Compute scaling factors based on prediction type
        if config.training.prediction_type == "v_prediction":
            c_skip = (1 / (sigmas**2 + 1).sqrt()).to(device=device)
            c_out = (-sigmas / (sigmas**2 + 1).sqrt()).to(device=device)
            c_in = (1 / (sigmas**2 + 1).sqrt()).to(device=device)
        elif config.training.prediction_type == "epsilon":
            c_skip = alphas_cumprod.sqrt()
            c_out = (1 - alphas_cumprod).sqrt()
            c_in = torch.ones_like(sigmas, device=device)
        else:
            raise ValueError(f"Unknown prediction type: {config.training.prediction_type}")
            
        # Verify all tensors are on the correct device
        params = {
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
        
        # Log device information for debugging
        for name, param in params.items():
            if isinstance(param, torch.Tensor):
                logger.info(f"Parameter {name} on device: {param.device}")
        
        return params
        
    except Exception as e:
        logger.error(f"Error in get_scheduler_parameters: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def configure_noise_scheduler(config: Config, device: torch.device) -> Dict[str, Any]:
    """Configure noise scheduler with Karras schedule and pre-compute training parameters."""
    try:
        if not isinstance(config, Config):
            raise ValueError(f"Expected config to be Config, got {type(config)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"Expected device to be torch.device, got {type(device)}")
            
        # Initialize scheduler
        try:
            scheduler = DDPMScheduler(
                num_train_timesteps=config.model.num_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=config.training.prediction_type,
                clip_sample=False,
                thresholding=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize DDPMScheduler: {str(e)}")
            raise
        
        # Generate sigmas and compute parameters
        try:
            sigmas = get_sigmas(config, device)
            params = get_scheduler_parameters(sigmas, config, device)
            
            # Move all parameters to the correct device
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    params[key] = value.to(device=device)
                    logger.info(f"Moved scheduler parameter {key} to device {device}")
            
            # Verify all parameters are on correct device
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    if value.device != device:
                        raise RuntimeError(f"Failed to move scheduler parameter {key} to device {device}")
                    
        except Exception as e:
            logger.error("Failed to generate scheduler parameters")
            raise
        
        # Update scheduler with computed values
        try:
            scheduler.alphas = params['alphas']
            scheduler.betas = params['betas']
            scheduler.alphas_cumprod = params['alphas_cumprod']
            scheduler.sigmas = sigmas
            scheduler.init_noise_sigma = sigmas.max()
            
            # Verify scheduler parameters
            if not isinstance(scheduler.alphas, torch.Tensor) or scheduler.alphas.device != device:
                raise RuntimeError(f"Scheduler alphas not on correct device: {scheduler.alphas.device} vs {device}")
            if not isinstance(scheduler.betas, torch.Tensor) or scheduler.betas.device != device:
                raise RuntimeError(f"Scheduler betas not on correct device: {scheduler.betas.device} vs {device}")
            if not isinstance(scheduler.alphas_cumprod, torch.Tensor) or scheduler.alphas_cumprod.device != device:
                raise RuntimeError(f"Scheduler alphas_cumprod not on correct device: {scheduler.alphas_cumprod.device} vs {device}")
            if not isinstance(scheduler.sigmas, torch.Tensor) or scheduler.sigmas.device != device:
                raise RuntimeError(f"Scheduler sigmas not on correct device: {scheduler.sigmas.device} vs {device}")
                
        except Exception as e:
            logger.error(f"Failed to update scheduler parameters: {str(e)}")
            raise
        
        # Return all parameters including scheduler
        return {'scheduler': scheduler, **params}

    except Exception as e:
        logger.error(f"Failed to configure noise scheduler: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise