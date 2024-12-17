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
    """Optimize memory usage with SDXL-specific techniques.
    Each optimization has error checking and fallbacks.
    """
    try:
        # Force garbage collection before setup
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                # Verify cleanup worked
                initial_memory = torch.cuda.memory_allocated()
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.memory_allocated() >= initial_memory:
                    logger.warning("Memory cleanup may not be working effectively")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
        
        # Configure model memory format with verification
        try:
            if hasattr(model, 'conv_in') and hasattr(model.conv_in, 'weight'):
                original_format = model.conv_in.weight.data.memory_format
                model = configure_model_memory_format(model, config)
                # Verify memory format changed if it should have
                if config.system.channels_last:
                    if model.conv_in.weight.data.memory_format == original_format:
                        logger.warning("channels_last memory format may not have been applied")
        except Exception as e:
            logger.warning(f"Memory format optimization failed: {e}")
        
        # Enable memory efficient attention with verification
        if config.system.enable_xformers and is_xformers_installed():
            try:
                # Store original forward function
                original_forward = model.forward
                model.enable_xformers_memory_efficient_attention()
                # Verify the forward function changed
                if model.forward == original_forward:
                    logger.warning("xformers attention may not have been applied")
                else:
                    logger.info("Verified xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Failed to enable xformers: {e}, falling back to default attention")
                try:
                    # Try to enable memory efficient attention without xformers
                    model.set_use_memory_efficient_attention(True)
                except Exception as e2:
                    logger.warning(f"Failed to enable default memory efficient attention: {e2}")
        
        # Enable gradient checkpointing with verification
        if config.system.gradient_checkpointing:
            try:
                model.enable_gradient_checkpointing()
                # Verify gradient checkpointing is enabled
                checkpointing_enabled = False
                for module in model.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        if module.gradient_checkpointing:
                            checkpointing_enabled = True
                            break
                if not checkpointing_enabled:
                    logger.warning("Gradient checkpointing may not be enabled")
                else:
                    logger.info("Verified gradient checkpointing is enabled")
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")
        
        # Configure CUDA optimizations with verification
        if torch.cuda.is_available():
            try:
                # Store original settings
                original_tf32 = torch.backends.cuda.matmul.allow_tf32
                original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                original_benchmark = torch.backends.cudnn.benchmark
                
                # Apply optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if config.system.cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                
                # Verify settings changed
                if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
                    logger.warning("TF32 optimizations may not be enabled")
                if config.system.cudnn_benchmark and not torch.backends.cudnn.benchmark:
                    logger.warning("cuDNN benchmark may not be enabled")
                
                # Try to set memory fraction
                try:
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    # Verify it worked
                    allocated = torch.cuda.memory_allocated()
                    max_allowed = torch.cuda.get_device_properties(0).total_memory * 0.95
                    if allocated > max_allowed:
                        logger.warning("Memory fraction limit may not be working")
                except Exception as e:
                    logger.warning(f"Failed to set memory fraction: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to configure CUDA optimizations: {e}")
                # Try to restore original settings
                try:
                    torch.backends.cuda.matmul.allow_tf32 = original_tf32
                    torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
                    torch.backends.cudnn.benchmark = original_benchmark
                except:
                    pass
        
        # Get memory format for buffers
        memory_format = (torch.channels_last 
                        if config.system.channels_last 
                        else torch.contiguous_format)
        
        # Pre-allocate buffers with verification
        buffers = {}
        try:
            # Helper to verify tensor properties
            def verify_tensor(name: str, tensor: torch.Tensor, expected_size: tuple, 
                            expected_dtype: torch.dtype, expected_device: torch.device):
                if tensor.size() != expected_size:
                    logger.warning(f"{name} size mismatch: got {tensor.size()}, expected {expected_size}")
                if tensor.dtype != expected_dtype:
                    logger.warning(f"{name} dtype mismatch: got {tensor.dtype}, expected {expected_dtype}")
                if tensor.device.type != expected_device.type or (
                    expected_device.type == 'cuda' and 
                    tensor.device.index != (expected_device.index if expected_device.index is not None else 0)
                ):
                    logger.warning(f"{name} device mismatch: got {tensor.device}, expected {expected_device}")
                return tensor
            
            # Allocate and verify each buffer
            try:
                noise_template = verify_tensor(
                    "noise_template",
                    torch.empty(
                        (batch_size, 4, 128, 128),  # SDXL latent size
                        device=device,
                        dtype=torch.bfloat16,
                        memory_format=memory_format
                    ),
                    (batch_size, 4, 128, 128),
                    torch.bfloat16,
                    device
                )
                buffers['noise_template'] = noise_template
            except Exception as e:
                logger.error(f"Failed to allocate noise template: {e}")
                raise
            
            try:
                grad_norm_buffer = verify_tensor(
                    "grad_norm_buffer",
                    torch.zeros(len(list(model.parameters())), device=device),
                    (len(list(model.parameters())),),
                    torch.float32,
                    device
                )
                buffers['grad_norm_buffer'] = grad_norm_buffer
            except Exception as e:
                logger.error(f"Failed to allocate grad norm buffer: {e}")
                raise
            
            # Calculate latent dimensions
            max_height, max_width = config.data.image_size
            vae_scale_factor = 8
            max_latent_height = max(max_height // vae_scale_factor, 128)
            max_latent_width = max(max_width // vae_scale_factor, 128)
            
            try:
                noise_buffer = verify_tensor(
                    "noise_buffer",
                    torch.empty(
                        (micro_batch_size, 4, max_latent_height, max_latent_width),
                        device=device,
                        dtype=torch.bfloat16,
                        memory_format=memory_format
                    ),
                    (micro_batch_size, 4, max_latent_height, max_latent_width),
                    torch.bfloat16,
                    device
                )
                buffers['noise_buffer'] = noise_buffer
            except Exception as e:
                logger.error(f"Failed to allocate noise buffer: {e}")
                raise
            
            try:
                latent_buffer = verify_tensor(
                    "latent_buffer",
                    torch.empty_like(noise_buffer, memory_format=memory_format),
                    noise_buffer.size(),
                    noise_buffer.dtype,
                    device
                )
                buffers['latent_buffer'] = latent_buffer
            except Exception as e:
                logger.error(f"Failed to allocate latent buffer: {e}")
                raise
            
            try:
                timestep_buffer = verify_tensor(
                    "timestep_buffer",
                    torch.empty((micro_batch_size,), device=device, dtype=torch.long),
                    (micro_batch_size,),
                    torch.long,
                    device
                )
                buffers['timestep_buffer'] = timestep_buffer
            except Exception as e:
                logger.error(f"Failed to allocate timestep buffer: {e}")
                raise
            
            if config.training.snr_gamma is not None:
                try:
                    snr_weight_buffer = verify_tensor(
                        "snr_weight_buffer",
                        torch.empty((micro_batch_size,), device=device, dtype=torch.bfloat16),
                        (micro_batch_size,),
                        torch.bfloat16,
                        device
                    )
                    buffers['snr_weight_buffer'] = snr_weight_buffer
                except Exception as e:
                    logger.error(f"Failed to allocate SNR weight buffer: {e}")
                    raise
            
        except Exception as e:
            logger.error(f"Failed to allocate buffers: {e}")
            # Try to free any partially allocated buffers
            for buffer in buffers.values():
                try:
                    del buffer
                except:
                    pass
            torch.cuda.empty_cache()
            raise
        
        # Final verification
        try:
            # Test basic operations on buffers
            for name, buffer in buffers.items():
                if buffer is not None:
                    try:
                        # Try basic operations
                        buffer.zero_()
                        if buffer.dtype.is_floating_point:
                            buffer.uniform_()
                    except Exception as e:
                        logger.warning(f"Buffer {name} may not be usable: {e}")
            
            # Verify model can still do a basic forward pass
            try:
                dummy_input = torch.randn(1, model.config.in_channels, 64, 64, device=device)
                model(dummy_input)
            except Exception as e:
                logger.warning(f"Model forward pass verification failed: {e}")
                
        except Exception as e:
            logger.warning(f"Final verification failed: {e}")
        
        return {
            'model': model,
            **buffers
        }

    except Exception as e:
        logger.error(f"Fatal error in memory optimizations: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # Emergency cleanup
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
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
    """Verify memory optimizations are still active.
    Returns dict of optimization states that can be checked by trainer.
    """
    optimization_states = {}
    
    try:
        # Verify memory format
        if config.system.channels_last:
            format_ok = model.conv_in.weight.data.memory_format == torch.channels_last
            if not format_ok:
                logger.warning("channels_last memory format was lost")
            optimization_states['channels_last'] = format_ok
            
        # Verify xformers if enabled
        if config.system.enable_xformers and is_xformers_installed():
            has_xformers = False
            for module in model.modules():
                if hasattr(module, '_use_memory_efficient_attention_xformers'):
                    if module._use_memory_efficient_attention_xformers:
                        has_xformers = True
                        break
            if not has_xformers:
                logger.warning("xformers attention was disabled")
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
                logger.warning("Gradient checkpointing was disabled")
            optimization_states['gradient_checkpointing'] = checkpointing_enabled
            
        # Verify CUDA optimizations
        if torch.cuda.is_available():
            tf32_ok = torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
            if not tf32_ok:
                logger.warning("TF32 optimizations were disabled")
            optimization_states['tf32'] = tf32_ok
            
            if config.system.cudnn_benchmark:
                benchmark_ok = torch.backends.cudnn.benchmark
                if not benchmark_ok:
                    logger.warning("cuDNN benchmark was disabled")
                optimization_states['cudnn_benchmark'] = benchmark_ok
                
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