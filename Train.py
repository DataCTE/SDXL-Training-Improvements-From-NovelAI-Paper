import argparse
import os
import signal
import sys
import torch
import wandb
from pathlib import Path
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from safetensors.torch import load_file
from adamw_bf16 import AdamWBF16

from configs.training_config import TrainingConfig
from trainers.sdxl_trainer import SDXLTrainer
from data.dataset import NovelAIDataset
from data.sampler import AspectBatchSampler
from utils.checkpoints import CheckpointManager

def setup_wandb(config: TrainingConfig):
    wandb.init(
        project="sdxl-finetune",
        config={
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "effective_batch": config.batch_size * config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "model": "SDXL-base-1.0",
            "optimizer": "AdamW-BF16",
            "scheduler": "DDPM",
            "min_snr_gamma": config.min_snr_gamma,
        }
    )

def load_unet(args, config: TrainingConfig, device: torch.device):
    """Load UNet model based on arguments"""
    if args.resume_from_checkpoint:
        print(f"Loading UNet from checkpoint directory: {args.resume_from_checkpoint}")
        return UNet2DConditionModel.from_pretrained(
            args.resume_from_checkpoint,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    elif args.unet_path:
        print(f"Loading UNet from safetensors file: {args.unet_path}")
        unet_dir = os.path.dirname(args.unet_path)
        config_path = os.path.join(unet_dir, "config.json")
        
        if os.path.exists(config_path):
            print(f"Found config.json in same directory, loading from: {unet_dir}")
            unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(device)
        else:
            print("No config.json found, loading architecture from base model")
            unet = UNet2DConditionModel.from_pretrained(
                config.pretrained_model_name,
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            ).to(device)
            state_dict = load_file(args.unet_path)
            unet.load_state_dict(state_dict)
        return unet
    else:
        print("Loading fresh UNet from pretrained model")
        return UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, 
                       help="Path to checkpoint directory to resume from")
    parser.add_argument("--unet_path", type=str, 
                       help="Path to UNet safetensors file to start from")
    args = parser.parse_args()

    # Initialize config and components
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    
    # Setup wandb
    setup_wandb(config)
    
    # Load models
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("Loading UNet...")
    unet = load_unet(args, config, device)
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()
    
    # Setup dataset and dataloader
    print("Creating dataset...")
    dataset = NovelAIDataset(
        image_dirs=config.image_dirs,
        transform=config.get_transform(),
        device=device,
        vae=vae,
        cache_dir=config.cache_dir,
        text_cache_dir=config.text_cache_dir
    )
    
    print("Creating dataloader...")
    dataloader = AspectBatchSampler(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Setup optimizer and scheduler
    print("Setting up optimizer...")
    optimizer = AdamWBF16(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    print("Setting up noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name,
        subfolder="scheduler",
        torch_dtype=torch.bfloat16
    )
    
    # Setup accelerator
    print("Setting up accelerator...")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accum_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="logs",
        device_placement=True,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("sdxl_finetune")
    
    # Create trainer
    trainer = SDXLTrainer(
        model=unet,
        vae=vae,
        optimizer=optimizer,
        scheduler=noise_scheduler,
        device=device,
        config=config,
        accelerator=accelerator,
        checkpoint_manager=checkpoint_manager
    )
    
    # Training loop
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint")
    finally:
        wandb.finish()
        accelerator.end_training()

if __name__ == "__main__":
    main()

