"""Device management utilities for SDXL model training.

This module provides utilities for managing device placement and memory
for PyTorch models during training. It handles CUDA, MPS, and CPU devices
with proper error handling and memory management.

Key Features:
- Automatic device selection
- Memory usage tracking
- Safe device movement
- Resource cleanup
"""

import gc
import contextlib
import logging
import traceback
from typing import Generator

import torch
import torch.nn as nn
from torch import device as torch_device

logger = logging.getLogger(__name__)

def get_device() -> torch_device:
    """Get the best available device for model training.
    
    Returns:
        torch.device: Best available device (CUDA > MPS > CPU)
        
    Note:
        Logs device information and available memory for CUDA devices
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory/1e9
        logger.info("Using CUDA device: %s", device_name)
        logger.info("Available CUDA memory: %.2f GB", memory_gb)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

@contextlib.contextmanager
def to_device(
    model: nn.Module,
    device: torch_device
) -> Generator[nn.Module, None, None]:
    """Temporarily move model to specified device with safe cleanup.
    
    Args:
        model: PyTorch model to move
        device: Target device
        
    Yields:
        nn.Module: Model on target device
        
    Note:
        Automatically moves model back to original device after use
    """
    original_device = next(model.parameters()).device
    try:
        model.to(device)
        yield model
    finally:
        model.to(original_device)

def cleanup_memory(
    models: dict[str, nn.Module],
    force_gc: bool = True
) -> None:
    """Clean up GPU memory and disable memory-intensive features.
    
    Args:
        models: Dictionary of model components
        force_gc: Whether to force garbage collection
        
    Note:
        - Empties CUDA cache if available
        - Disables gradient checkpointing
        - Forces garbage collection if requested
    """
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                allocated_gb = torch.cuda.memory_allocated()/1e9
                logger.info("CUDA memory after cleanup: %.2f GB", allocated_gb)
        
        # Disable gradient checkpointing
        for _, model in models.items():
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            elif hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()
                
        # Force garbage collection if requested
        if force_gc:
           
            gc.collect()
            
        logger.info("Memory cleanup completed successfully")
        
    except (RuntimeError, AttributeError, TypeError) as e:
        logger.error("Error during memory cleanup: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        # Don't raise - cleanup errors shouldn't halt execution
