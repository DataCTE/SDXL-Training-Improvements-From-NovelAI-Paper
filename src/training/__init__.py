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
    AverageMeter,
    MetricsManager
)

from src.utils.logging import (
    log_optimizer_config,
    log_vae_config,
    log_ema_config
)

from .ema import EMAModel
from .vae_finetuner import VAEFineTuner
from .loss import (
    compute_loss_weights,
    training_loss_v_prediction,
    PerceptualLoss
)

__all__ = [
    # Training core
    'train_epoch',
    'initialize_training_components',
    'run_validation',
    
    # Logging utilities
    'log_optimizer_config',
    'log_vae_config',
    'log_ema_config',
    
    # Metrics
    'AverageMeter',
    'MetricsManager',
    
    # Models
    'EMAModel',
    'VAEFineTuner',
    
    # Loss functions
    'compute_loss_weights',
    'training_loss_v_prediction',
    'PerceptualLoss'
]
