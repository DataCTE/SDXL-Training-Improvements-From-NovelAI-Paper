"""Training components for SDXL model training.

This module provides core training functionality including:
- Training loop and epoch management
- Component initialization
- Validation utilities
- Logging configuration
- EMA model management
- VAE finetuning
"""

from .trainer import (
    train_epoch,
    initialize_training_components,
    run_validation,
    _log_optimizer_config,
    _log_vae_config,
    _log_ema_config,
    AverageMeter,
    MetricsManager
)

from .ema import EMAModel
from .vae_finetuner import VAEFinetuner
from .loss import (
    compute_snr_weight,
    compute_loss,
    compute_vae_loss
)

__all__ = [
    # Training core
    'train_epoch',
    'initialize_training_components',
    'run_validation',
    
    # Logging utilities
    '_log_optimizer_config',
    '_log_vae_config',
    '_log_ema_config',
    
    # Metrics
    'AverageMeter',
    'MetricsManager',
    
    # Models
    'EMAModel',
    'VAEFinetuner',
    
    # Loss functions
    'compute_snr_weight',
    'compute_loss',
    'compute_vae_loss'
]
