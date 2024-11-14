import torch
import contextlib
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

@contextlib.contextmanager
def to_device(model, device):
    """Temporarily move model to device"""
    original_device = next(model.parameters()).device
    try:
        model.to(device)
        yield model
    finally:
        model.to(original_device) 

def cleanup(models, args):
    """Cleanup after training"""
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Final CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Disable gradient checkpointing for cleanup
        for model_name, model in models.items():
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            elif hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()
        
    except Exception as cleanup_error:
        logger.error(f"Error during cleanup: {cleanup_error}")

