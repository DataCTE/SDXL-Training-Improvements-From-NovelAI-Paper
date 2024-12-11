import argparse
import os
import signal
import sys
import torch
import wandb
import torch.multiprocessing as mp
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from safetensors.torch import load_file
from adamw_bf16 import AdamWBF16

from configs.training_config import TrainingConfig
from trainers.sdxl_trainer import SDXLTrainer
from data.dataset import NovelAIDataset
from data.sampler import AspectBatchSampler
from utils.checkpoints import CheckpointManager
from utils.distributed import setup_distributed, cleanup_distributed

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--unet_path", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str)
    args = parser.parse_args()
    
    # Setup distributed training if needed
    if args.local_rank != -1:
        device = setup_distributed(args.local_rank, args.world_size)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        config = TrainingConfig()
        
        # Initialize wandb only on main process
        if args.local_rank == -1 or args.local_rank == 0:
            setup_wandb(config)
        
        # Load models
        unet = UNet2DConditionModel.from_pretrained(args.unet_path)
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        
        # Initialize optimizer
        optimizer = AdamWBF16(
            unet.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm
        )
        
        # Initialize dataset with distributed info
        dataset = NovelAIDataset(
            config.data_path,
            device=device,
            local_rank=args.local_rank,
            world_size=args.world_size if args.local_rank != -1 else 1
        )
        sampler = AspectBatchSampler(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            world_size=args.world_size,
            rank=args.local_rank
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize trainer
        trainer = SDXLTrainer(
            model=unet,
            vae=vae,
            optimizer=optimizer,
            scheduler=DDPMScheduler(),
            device=device,
            config=config,
            local_rank=args.local_rank
        )
        
        if args.resume_from_checkpoint:
            trainer.load_checkpoint(args.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(trainer.current_epoch, config.num_epochs):
            trainer.current_epoch = epoch
            avg_loss = trainer.train_epoch(dataloader)
            
            if args.local_rank == -1 or args.local_rank == 0:
                print(f"Epoch {epoch} completed with average loss: {avg_loss:.4f}")
                trainer.save_checkpoint()
                
    finally:
        cleanup_distributed()
        if args.local_rank == -1 or args.local_rank == 0:
            wandb.finish()

if __name__ == "__main__":
    main()

