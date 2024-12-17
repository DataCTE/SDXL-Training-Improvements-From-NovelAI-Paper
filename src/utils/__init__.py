"""Utilities for SDXL training and model management.

This package provides utilities for:
- Model management and initialization
- Training helpers and checkpointing
- System setup and configuration
- Logging and metrics
- Data processing
"""

# Model utilities
from .model.model import (
    create_unet,
    create_vae,
    setup_model,
    initialize_model_weights,
    configure_model_memory_format,
    is_xformers_installed
)
from .model.embeddings import get_add_time_ids
from .model.noise import generate_noise

# Training utilities
from .training.checkpoints import save_checkpoint, load_checkpoint
from .training.transforms import (
    ensure_three_channels,
    convert_to_bfloat16,
    get_transform
)

# System utilities
from .system.setup import (
    setup_memory_optimizations,
    setup_distributed,
    cleanup_distributed,
    verify_memory_optimizations
)

# Logging utilities
from .logging.metrics import (
    setup_logging,
    log_system_info,
    compute_grad_norm,
    log_metrics,
    cleanup_logging
)

__all__ = [
    # Model
    'create_unet',
    'create_vae',
    'setup_model',
    'initialize_model_weights',
    'configure_model_memory_format',
    'is_xformers_installed',
    'get_add_time_ids',
    'generate_noise',
    
    # Training
    'save_checkpoint',
    'load_checkpoint',
    'ensure_three_channels',
    'convert_to_bfloat16',
    'get_transform',
    
    # System
    'setup_memory_optimizations',
    'setup_distributed',
    'cleanup_distributed',
    'verify_memory_optimizations',
    
    # Logging
    'setup_logging',
    'log_system_info', 
    'compute_grad_norm',
    'log_metrics',
    'cleanup_logging'
] 