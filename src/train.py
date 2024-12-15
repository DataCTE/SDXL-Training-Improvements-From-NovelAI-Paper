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
    
    # Setup dataset using config paths
    dataset = NovelAIDataset(
        image_dirs=config.data.image_dirs,
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
    dataloader = trainer.create_dataloader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name} signal. Saving checkpoint...")
        
        if trainer is not None:
            try:
                emergency_save_path = os.path.join("checkpoints", "emergency_checkpoint")
                trainer.save_checkpoint(emergency_save_path)
                print("Emergency checkpoint saved successfully.")
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")
        
        print("Exiting...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Training loop
    start_epoch = trainer.current_epoch
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training from epoch {start_epoch + 1}")
    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
            
            epoch_loss = trainer.train_epoch(
                dataloader=dataloader,
                epoch=epoch,
                log_interval=config.training.log_steps
            )
            
            if is_main_process:
                # Save epoch checkpoint
                epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                trainer.save_checkpoint(epoch_checkpoint_path)
                
                # Save step checkpoint if interval is reached
                if trainer.global_step % config.training.save_steps == 0:
                    step_checkpoint_path = os.path.join(
                        save_dir,
                        f"checkpoint_step_{trainer.global_step}"
                    )
                    trainer.save_checkpoint(step_checkpoint_path)
            
            print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        if is_main_process:
            trainer.save_checkpoint(os.path.join(save_dir, "final_checkpoint"))
    
    # Cleanup
    if is_main_process:
        wandb.finish()
    accelerator.end_training()
    print("Training completed!")

if __name__ == "__main__":
    main() 