import argparse
import torch
import os
import wandb
import signal
import sys
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel, 
    AutoencoderKL,
    DDPMScheduler
)
from safetensors.torch import load_file
import torch.distributed as dist
from typing import Optional, Tuple
from pathlib import Path
import traceback
import logging
from src.training.vae_trainer import VAETrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project components
from data.dataset import NovelAIDataset
from src.training.trainer import NovelAIDiffusionV3Trainer
from utils.transforms import get_transform
from adamw_bf16 import AdamWBF16
from src.config.config import Config

def setup_accelerator(config: Config) -> Accelerator:
    """Setup accelerator with proper configuration"""
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

    # Create accelerator without sync_batch_norm parameter
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.system.mixed_precision,
        log_with="wandb",
        project_dir=config.paths.logs_dir,
        device_placement=True,
        fsdp_plugin=fsdp_plugin
    )

    # Handle batch norm synchronization separately if needed
    if config.system.use_fsdp and config.system.sync_batch_norm:
        import torch.nn as nn
        def convert_sync_batchnorm(module):
            module_output = module
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module_output = nn.SyncBatchNorm.convert_sync_batchnorm(module)
            for name, child in module.named_children():
                module_output.add_module(name, convert_sync_batchnorm(child))
            return module_output
        
        # This will be applied to models before training
        accelerator.sync_batchnorm = convert_sync_batchnorm

    return accelerator

def setup_distributed(config: Config) -> Tuple[int, int]:
    """Initialize distributed training environment"""
    if not torch.cuda.is_available():
        return 1, 0

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
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
    return world_size, rank

def parse_args():
    parser = argparse.ArgumentParser(description='Train SDXL with optimizations')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        help='Path to checkpoint directory to resume from'
    )
    parser.add_argument(
        '--unet_path',
        type=str,
        help='Path to UNet safetensors file to start from'
    )
    parser.add_argument(
        '--train_vae',
        action='store_true',
        help='Enable VAE training mode'
    )
    parser.add_argument(
        '--vae_path',
        type=str,
        help='Path to VAE safetensors file to start from'
    )
    return parser.parse_args()

def setup_model(
    args,
    device: torch.device,
    config: Config
) -> tuple[UNet2DConditionModel, AutoencoderKL]:
    """Setup UNet and VAE models"""
    pretrained_model_name = config.model.pretrained_model_name
    
    print("Loading VAE...")
    if args.train_vae and args.vae_path:
        print(f"Loading VAE from: {args.vae_path}")
        vae = AutoencoderKL.from_pretrained(
            args.vae_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(device)
        
    if not args.train_vae:
        vae.eval()
        vae.requires_grad_(False)
    
    print("Loading UNet...")
    if args.resume_from_checkpoint:
        print(f"Loading UNet from checkpoint: {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            args.resume_from_checkpoint,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    elif args.unet_path:
        print(f"Loading UNet weights from: {args.unet_path}")
        unet_dir = os.path.dirname(args.unet_path)
        config_path = os.path.join(unet_dir, "config.json")
        
        if os.path.exists(config_path):
            unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(device)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            ).to(device)
            state_dict = load_file(args.unet_path)
            unet.load_state_dict(state_dict)
    else:
        print("Loading fresh UNet from pretrained model")
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    
    return unet, vae

def main():
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Create directories from config
    if dist.is_initialized() and dist.get_rank() == 0:
        os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
        os.makedirs(config.paths.logs_dir, exist_ok=True)
        os.makedirs(config.paths.output_dir, exist_ok=True)
        os.makedirs(config.data.cache_dir, exist_ok=True)
        os.makedirs(config.data.text_cache_dir, exist_ok=True)
    
    # Setup distributed training
    world_size, rank = setup_distributed(config)
    is_main_process = rank == 0
    
    # Setup accelerator
    accelerator = setup_accelerator(config)
    
    device = accelerator.device
    
    # Initialize wandb if main process
    if is_main_process:
        wandb.init(
            project="sdxl-finetune",
            config={
                "batch_size": config.training.batch_size,
                "grad_accum_steps": config.training.gradient_accumulation_steps,
                "effective_batch": config.training.batch_size * config.training.gradient_accumulation_steps * world_size,
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "world_size": world_size,
                "rank": rank,
                "backend": config.system.backend,
                "timestep_bias_strategy": config.training.timestep_bias_strategy,
                "snr_gamma": config.training.snr_gamma
            }
        )
    
    # Setup models with config
    unet, vae = setup_model(args, device, config)
    
    if config.system.use_fsdp and config.system.sync_batch_norm:
        unet = accelerator.sync_batchnorm(unet)
        if args.train_vae:
            vae = accelerator.sync_batchnorm(vae)
    
    if args.train_vae:
        # Setup VAE trainer
        trainer = VAETrainer(
            config=config,
            accelerator=accelerator,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    else:
        # Initialize trainer with new signature
        trainer = NovelAIDiffusionV3Trainer(
            config_path=args.config,
            model=unet,
            vae=vae,
            accelerator=accelerator
        )
    
    # Validate image directories before dataset creation
    if not config.data.image_dirs:
        raise ValueError(
            "No image directories specified in config. Please add valid image directories to config.data.image_dirs"
        )
        
    total_images = 0
    valid_dirs = []
    
    # Check each directory
    for img_dir in config.data.image_dirs:
        if not os.path.exists(img_dir):
            logger.warning(f"Directory not found: {img_dir}")
            continue
            
        num_images = len([f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        if num_images == 0:
            logger.warning(f"No valid images found in directory: {img_dir}")
            continue
            
        total_images += num_images
        valid_dirs.append(img_dir)
        
    if total_images == 0:
        raise ValueError(
            "No valid images found in any of the specified directories. "
            "Please ensure the directories contain supported image files "
            "(.png, .jpg, .jpeg, .webp)"
        )
        
    # Update config with validated directories
    config.data.image_dirs = valid_dirs
    
    if is_main_process:
        logger.info(f"Found {total_images} valid images across {len(valid_dirs)} directories")

    # Setup dataset using validated directories
    dataset = NovelAIDataset(
        image_dirs=valid_dirs,
        transform=get_transform(),
        device=device,
        vae=vae,
        cache_dir=config.data.cache_dir,
        text_cache_dir=config.data.text_cache_dir,
        config=config
    )
    
    # Prepare for distributed training - optimizer removed since it's handled by trainer
    trainer, dataset = accelerator.prepare(trainer, dataset)
    
    # Create dataloader with distributed sampler
    train_dataloader = trainer.create_dataloader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )
    
    # Training loop
    start_epoch = trainer.current_epoch
    save_dir = config.paths.checkpoints_dir
    
    try:
        if is_main_process:
            logger.info(f"Starting {'VAE' if args.train_vae else 'UNet'} training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, config.training.num_epochs + 1):
            if is_main_process:
                logger.info(f"\nStarting epoch {epoch}")
            
            # Set epoch for distributed sampler
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = trainer.train_epoch(
                epoch=epoch,
                train_dataloader=train_dataloader
            )
            
            if is_main_process:
                logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
                
                # Save checkpoint
                trainer.save_checkpoint(save_dir, epoch)
                
                # Log metrics with appropriate prefix
                prefix = "vae_" if args.train_vae else "unet_"
                wandb.log({
                    f"{prefix}epoch": epoch,
                    f"{prefix}loss": epoch_loss,
                }, step=trainer.global_step)
            
            # Sync processes after each epoch
            if world_size > 1:
                torch.distributed.barrier()
            
    except KeyboardInterrupt:
        if is_main_process:
            logger.info("\nTraining interrupted. Saving final checkpoint...")
            trainer.save_checkpoint(save_dir, epoch)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
        
    finally:
        # Cleanup
        if is_main_process:
            wandb.finish()
        accelerator.end_training()
        
        # Cleanup distributed
        if world_size > 1:
            torch.distributed.destroy_process_group()
            
        if is_main_process:
            logger.info("Training completed!")

if __name__ == "__main__":
    main() 