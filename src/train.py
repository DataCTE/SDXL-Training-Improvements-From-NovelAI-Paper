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
from utils.optimizer import AdamWBF16

def setup_distributed():
    """Initialize distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0

def parse_args():
    parser = argparse.ArgumentParser(description='Train SDXL with optimizations')
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
        '--batch_size',
        type=int,
        default=8,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=4,
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=4e-7,
        help='Learning rate'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=1000,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--log_steps',
        type=int,
        default=10,
        help='Log metrics every N steps'
    )
    return parser.parse_args()

def setup_model(
    args,
    device: torch.device,
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
) -> tuple[UNet2DConditionModel, AutoencoderKL]:
    """Setup UNet and VAE models"""
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
    
    # Setup distributed training
    world_size, rank = setup_distributed()
    is_main_process = rank == 0
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
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
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch": args.batch_size * args.grad_accum_steps * world_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
            }
        )
    
    # Setup models
    unet, vae = setup_model(args, device)
    
    # Setup optimizer
    optimizer = AdamWBF16(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        torch_dtype=torch.bfloat16
    )
    
    # Setup dataset and dataloader
    image_dirs = [
        "path/to/image/dir1",
        "path/to/image/dir2",
        # Add your image directories here
    ]
    
    dataset = NovelAIDataset(
        image_dirs=image_dirs,
        transform=get_transform(),
        device=device,
        vae=vae,
        cache_dir="latent_cache",
        text_cache_dir="text_cache"
    )
    
    # Create trainer instance
    trainer = NovelAIDiffusionV3Trainer(
        model=unet,
        vae=vae,
        optimizer=optimizer,
        scheduler=noise_scheduler,
        device=device,
        accelerator=accelerator,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=args.grad_accum_steps
    )
    
    # Prepare for distributed training
    trainer, optimizer, dataset = accelerator.prepare(
        trainer, optimizer, dataset
    )
    
    # Create dataloader
    dataloader = trainer.create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
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
        for epoch in range(start_epoch, args.num_epochs):
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            
            epoch_loss = trainer.train_epoch(
                dataloader=dataloader,
                epoch=epoch,
                log_interval=args.log_steps
            )
            
            if is_main_process:
                # Save epoch checkpoint
                epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                trainer.save_checkpoint(epoch_checkpoint_path)
                
                # Save step checkpoint if interval is reached
                if trainer.global_step % args.save_steps == 0:
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