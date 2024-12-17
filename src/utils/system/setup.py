import torch
import os
import logging
import gc
from typing import Tuple, Dict, Any
from src.config.config import Config
import traceback
from src.utils.model.model import configure_model_memory_format, is_xformers_installed
import torch.distributed as dist

logger = logging.getLogger(__name__)

def setup_memory_optimizations(
    model: torch.nn.Module,
    config: Config,
    device: torch.device,
    batch_size: int,
    micro_batch_size: int
) -> Dict[str, Any]:
    """Setup essential memory optimizations for SDXL."""
    try:
        # Basic cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Enable xformers if available
        if config.system.enable_xformers and is_xformers_installed():
            try:
                model.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Failed to enable xformers: {e}")
        
        # Enable gradient checkpointing
        if config.system.gradient_checkpointing:
            try:
                model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing")
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")
        
        return {'model': model}

    except Exception as e:
        logger.error(f"Error in memory optimizations: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def setup_distributed():
    """Setup distributed training with proper error handling."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, distributed training disabled")
            return False
            
        if not dist.is_available():
            logger.warning("torch.distributed not available, distributed training disabled")
            return False

        # Check for required environment variables
        required_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.warning(f"Distributed training environment variables missing: {missing_vars}")
            logger.warning("Distributed training disabled")
            return False
            
        if not dist.is_initialized():
            logger.info("Initializing distributed training...")
            
            # Initialize process group
            dist.init_process_group(backend='nccl')
            
            # Set device
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            
            logger.info(f"Distributed training initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to setup distributed training: {str(e)}")
        logger.warning("Falling back to single GPU training")
        return False

def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def verify_memory_optimizations(
    model: torch.nn.Module,
    config: Config,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, bool]:
    """Verify essential memory optimizations are active."""
    optimization_states = {}
    
    try:
        # Verify xformers if enabled
        if config.system.enable_xformers and is_xformers_installed():
            has_xformers = False
            for module in model.modules():
                if hasattr(module, '_use_memory_efficient_attention_xformers'):
                    if module._use_memory_efficient_attention_xformers:
                        has_xformers = True
                        break
            if not has_xformers:
                logger.warning("xformers attention was not enabled")
            optimization_states['xformers'] = has_xformers
            
        # Verify gradient checkpointing
        if config.system.gradient_checkpointing:
            checkpointing_enabled = False
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    if module.gradient_checkpointing:
                        checkpointing_enabled = True
                        break
            if not checkpointing_enabled:
                logger.warning("Gradient checkpointing was not enabled")
            optimization_states['gradient_checkpointing'] = checkpointing_enabled
                
    except Exception as e:
        logger.warning(f"Error verifying memory optimizations: {e}")
        optimization_states['error'] = str(e)
        
    return optimization_states