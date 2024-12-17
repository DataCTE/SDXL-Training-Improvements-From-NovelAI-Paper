import torch
import os
import logging
import gc
from accelerate import Accelerator
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from typing import Tuple, Dict, Any
from src.config.config import Config
import traceback
from src.utils.model import configure_model_memory_format, is_xformers_installed
from diffusers import DDPMScheduler
import torch.distributed as dist
import torch.multiprocessing as mp

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

def verify_buffer_states(
    buffers: Dict[str, torch.Tensor],
    micro_batch_size: int,
    model_dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, bool]:
    """Verify buffer states and properties.
    Returns dict of buffer states that can be checked by trainer.
    """
    buffer_states = {}
    critical_errors = []
    warnings = []
    
    try:
        # Verify all required buffers exist
        required_buffers = [
            'noise_template', 'grad_norm_buffer', 'noise_buffer',
            'latent_buffer', 'timestep_buffer'
        ]
        
        missing_buffers = [buf for buf in required_buffers if buf not in buffers]
        if missing_buffers:
            critical_errors.append(f"Missing required buffers: {missing_buffers}")
            buffer_states['missing_buffers'] = missing_buffers
            
        # Verify buffer properties
        for name, buffer in buffers.items():
            if name == 'model':  # Skip the model key
                continue
                
            buffer_states[name] = True  # Start optimistic
                
            # Type check
            if not isinstance(buffer, torch.Tensor):
                critical_errors.append(f"Buffer {name} is not a tensor")
                buffer_states[name] = False
                continue
                
            # Device type check (cuda vs cpu)
            if buffer.device.type != device.type:
                warnings.append(f"Buffer {name} device type mismatch: expected {device.type}, got {buffer.device.type}")
                # Don't fail for device mismatch, we can fix this
                
            # For CUDA devices, verify index if specified
            if device.type == 'cuda':
                expected_index = device.index if device.index is not None else 0
                if buffer.device.index != expected_index:
                    warnings.append(f"Buffer {name} CUDA device index mismatch: expected {expected_index}, got {buffer.device.index}")
                    # Don't fail for index mismatch, we can fix this
                
            # Dtype check
            if name != 'timestep_buffer' and buffer.dtype != model_dtype:
                warnings.append(f"Buffer {name} has wrong dtype: expected {model_dtype}, got {buffer.dtype}")
                # Don't fail for dtype mismatch, we can fix this
                
            # Contiguity check
            if not buffer.is_contiguous():
                warnings.append(f"Buffer {name} lost contiguity")
                # Don't fail for contiguity, we can fix this
                
            # Shape check for batch dimension
            if name in ['noise_buffer', 'latent_buffer', 'timestep_buffer']:
                if buffer.shape[0] != micro_batch_size:
                    critical_errors.append(f"Buffer {name} has wrong batch dimension: expected {micro_batch_size}, got {buffer.shape[0]}")
                    buffer_states[name] = False
                    
        # Log all warnings
        for warning in warnings:
            logger.warning(warning)
            
        # Log critical errors
        if critical_errors:
            for error in critical_errors:
                logger.error(error)
            buffer_states['critical_errors'] = critical_errors
            
        # Only consider buffers invalid if there are critical errors
        buffer_states['valid'] = len(critical_errors) == 0
            
    except Exception as e:
        logger.error(f"Error verifying buffer states: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        buffer_states['error'] = str(e)
        buffer_states['valid'] = False
        
    return buffer_states

def verify_scheduler_parameters(
    scheduler_params: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, bool]:
    """Verify scheduler parameters are valid.
    Returns dict of parameter states that can be checked by trainer.
    """
    param_states = {}
    critical_errors = []
    warnings = []
    
    try:
        # Normalize device specification
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda:0')
            
        # Verify required parameters exist
        required_params = [
            'scheduler', 'alphas', 'betas', 'alphas_cumprod',
            'sigmas', 'snr_values', 'snr_weights', 
            'c_skip', 'c_out', 'c_in'
        ]
        
        missing_params = [p for p in required_params if p not in scheduler_params]
        if missing_params:
            critical_errors.append(f"Missing scheduler parameters: {missing_params}")
            param_states['missing_params'] = missing_params
            
        # Verify parameter properties
        for name, param in scheduler_params.items():
            param_states[name] = True  # Start optimistic
            
            if name == 'scheduler':
                if not isinstance(param, DDPMScheduler):
                    critical_errors.append("Invalid scheduler type")
                    param_states['scheduler_type'] = False
                continue
                
            if not isinstance(param, torch.Tensor):
                critical_errors.append(f"Scheduler parameter {name} is not a tensor")
                param_states[name] = False
                continue
                
            # Device check - compare device strings to handle cuda vs cuda:0
            if str(param.device) != str(device):
                warnings.append(f"Scheduler parameter {name} is on wrong device: {param.device} vs {device}")
                try:
                    # Try to move parameter to correct device
                    scheduler_params[name] = param.to(device=device)
                    logger.info(f"Moved scheduler parameter {name} to device {device}")
                except Exception as e:
                    logger.error(f"Failed to move parameter {name} to device {device}: {e}")
                    critical_errors.append(f"Failed to move parameter {name} to correct device")
                    param_states[name] = False
                
        # Verify scheduler configuration
        if scheduler_params['scheduler'].num_train_timesteps != len(scheduler_params['sigmas']):
            critical_errors.append("Scheduler timesteps mismatch")
            param_states['timesteps_match'] = False
            
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
            
        # Log critical errors
        if critical_errors:
            for error in critical_errors:
                logger.error(error)
            param_states['critical_errors'] = critical_errors
            
        # Only consider parameters invalid if there are critical errors
        param_states['valid'] = len(critical_errors) == 0
            
    except Exception as e:
        logger.error(f"Error verifying scheduler parameters: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        param_states['error'] = str(e)
        param_states['valid'] = False
        
    return param_states

def check_memory_status(
    initial_memory: float,
    device: torch.device,
    logger: logging.Logger,
    threshold: float = 1.1
) -> Dict[str, float]:
    """Check current memory status and detect leaks.
    Returns dict with memory metrics.
    """
    memory_stats = {}
    
    try:
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(device) / (1024**3)
            memory_stats['current'] = current_memory
            
            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            cleaned_memory = torch.cuda.memory_allocated(device) / (1024**3)
            memory_stats['after_cleanup'] = cleaned_memory
            
            # Check for leaks
            if cleaned_memory > initial_memory * threshold:
                leak_size = cleaned_memory - initial_memory
                logger.warning(f"Possible memory leak detected: {leak_size:.2f}GB unreleased")
                memory_stats['leak'] = leak_size
                
    except Exception as e:
        logger.warning(f"Error checking memory status: {e}")
        memory_stats['error'] = str(e)
        
    return memory_stats