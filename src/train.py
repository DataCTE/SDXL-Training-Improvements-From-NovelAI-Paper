import argparse
import torch
import os
import wandb
import signal
import sys
from accelerate import Accelerator
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel, 
    AutoencoderKL,
    DDPMScheduler
)
from safetensors.torch import load_file
import torch.distributed as dist
from typing import Optional
from pathlib import Path
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project components
from data.dataset import NovelAIDataset
from training.trainer import NovelAIDiffusionV3Trainer
from utils.transforms import get_transform
from adamw_bf16 import AdamWBF16
from src.config.config import Config


def setup_distributed():
    """Initialize distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0

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
    return parser.parse_args()

def setup_model(
    args,
    device: torch.device,
    config: Config
) -> tuple[UNet2DConditionModel, AutoencoderKL]:
    """Setup UNet and VAE models"""
    pretrained_model_name = config.model.pretrained_model_name
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to(device)
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
    os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.data.cache_dir, exist_ok=True)
    os.makedirs(config.data.text_cache_dir, exist_ok=True)
    
    # Setup distributed training
    world_size, rank = setup_distributed()
    is_main_process = rank == 0
    
    # Setup accelerator with config
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="tensorboard",
        project_dir="logs",
        device_placement=True,
    )
    
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
            }
        )
    
    # Setup models with config
    unet, vae = setup_model(args, device, config)
    
    # Setup optimizer
    optimizer = AdamWBF16(
        unet.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.optimizer_betas,
        weight_decay=config.training.weight_decay,
        eps=config.training.optimizer_eps
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        torch_dtype=torch.bfloat16
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
    
    # Create trainer with config
    trainer = NovelAIDiffusionV3Trainer(
        model=unet,
        vae=vae,
        optimizer=optimizer,
        scheduler=noise_scheduler,
        device=device,
        config=config,
        accelerator=accelerator,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Prepare for distributed training
    trainer, optimizer, dataset = accelerator.prepare(
        trainer, optimizer, dataset
    )
    
    # Create dataloader
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
        logger.info(f"Starting training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, config.training.num_epochs + 1):
            logger.info(f"\nStarting epoch {epoch}")
            
            epoch_loss = trainer.train_epoch(
                epoch=epoch,
                train_dataloader=train_dataloader
            )
            
            logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
            
            if is_main_process:
                # Save checkpoint
                trainer.save_checkpoint(save_dir, epoch)
                
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "loss": epoch_loss,
                }, step=trainer.global_step)
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving final checkpoint...")
        if is_main_process:
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
        logger.info("Training completed!")

if __name__ == "__main__":
    main() 