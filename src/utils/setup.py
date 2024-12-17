import torch
import os
import logging
from accelerate import Accelerator
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from typing import Tuple, Dict, Any
from src.config.config import Config
import traceback
from src.utils.model import configure_model_memory_format, is_xformers_installed

logger = logging.getLogger(__name__)

def setup_memory_optimizations(
    model: torch.nn.Module,
    config: Config,
    device: torch.device,
    batch_size: int,
    micro_batch_size: int
) -> Dict[str, Any]:
    """Optimize memory usage with NovelAI's techniques.
    
    Implements several memory optimization techniques:
    1. Channels last memory format for tensors
    2. Gradient checkpointing
    3. Efficient attention implementation
    4. Pre-allocated buffers
    """
    try:
        # Configure model memory format
        model = configure_model_memory_format(model, config)
        
        # Get memory format for buffers
        memory_format = (torch.channels_last 
                        if config.system.channels_last 
                        else torch.contiguous_format)
        
        # Pre-allocate reusable buffers with appropriate memory format
        noise_template = torch.empty(
            (batch_size, 4, 64, 64),  # Standard latent size
            device=device,
            dtype=torch.bfloat16,
            memory_format=memory_format
        )
        
        # Pre-allocate gradient norm buffer (1D tensor, memory format doesn't matter)
        grad_norm_buffer = torch.zeros(
            len(list(model.parameters())), 
            device=device
        )
        
        # Enable TF32 for better performance if CUDA is available
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Optimize CUDA operations
                if config.system.cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
            except Exception as e:
                logger.warning(f"Failed to configure CUDA optimizations: {e}")

        # Get maximum bucket dimensions
        max_height, max_width = config.data.image_size
        vae_scale_factor = 8  # SDXL VAE downscales by 8
        max_latent_height = max_height // vae_scale_factor
        max_latent_width = max_width // vae_scale_factor
        
        # Pre-allocate noise buffer for maximum possible size
        noise_buffer = torch.empty(
            (micro_batch_size, 4, max_latent_height, max_latent_width),
            device=device,
            dtype=torch.bfloat16,
            memory_format=memory_format
        )
        
        # Pre-allocate latent buffer
        latent_buffer = torch.empty_like(
            noise_buffer,
            memory_format=memory_format
        )
        
        # Pre-allocate timestep buffer
        timestep_buffer = torch.empty(
            (micro_batch_size,),
            device=device,
            dtype=torch.long
        )
        
        # Pre-allocate SNR weight buffer if using SNR weighting
        snr_weight_buffer = None
        if config.training.snr_gamma is not None:
            snr_weight_buffer = torch.empty(
                (micro_batch_size,),
                device=device,
                dtype=torch.bfloat16
            )
        
        return {
            'model': model,
            'noise_template': noise_template,
            'grad_norm_buffer': grad_norm_buffer,
            'noise_buffer': noise_buffer,
            'latent_buffer': latent_buffer,
            'timestep_buffer': timestep_buffer,
            'snr_weight_buffer': snr_weight_buffer
        }

    except Exception as e:
        logger.error(f"Failed to setup memory optimizations: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def setup_accelerator(config: Config) -> Accelerator:
    """Setup accelerator with proper configuration and error handling."""
    try:
        # Create FSDP plugin if enabled
        if config.system.use_fsdp:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy="FULL_SHARD" if config.system.full_shard else "SHARD_GRAD_OP",
                min_num_params=config.system.min_num_params_per_shard,
                cpu_offload=config.system.cpu_offload,
                forward_prefetch=config.system.forward_prefetch,
                backward_prefetch=config.system.backward_prefetch,
                limit_all_gathers=config.system.limit_all_gathers,
                state_dict_type="FULL_STATE_DICT",
                use_orig_params=True,
                sync_module_states=True,
            )
        else:
            fsdp_plugin = None

        accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=config.system.mixed_precision,
            log_with="wandb",
            project_dir=config.paths.logs_dir,
            device_placement=True,
            fsdp_plugin=fsdp_plugin
        )

        if config.system.use_fsdp and config.system.sync_batch_norm:
            import torch.nn as nn
            def convert_sync_batchnorm(module):
                module_output = module
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module_output = nn.SyncBatchNorm.convert_sync_batchnorm(module)
                for name, child in module.named_children():
                    module_output.add_module(name, convert_sync_batchnorm(child))
                return module_output
            
            accelerator.sync_batchnorm = convert_sync_batchnorm

        return accelerator

    except Exception as e:
        logger.error(f"Failed to setup accelerator: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def setup_distributed(config: Config) -> Tuple[int, int]:
    """Initialize distributed training environment with error handling."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, running in CPU mode")
            return 1, 0

        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size <= 1:
            logger.info("Running in single process mode")
            return 1, 0

        torch.distributed.init_process_group(
            backend=config.system.backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)

        if config.system.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.distributed.barrier()
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
        return world_size, rank

    except Exception as e:
        logger.error(f"Failed to setup distributed training: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 