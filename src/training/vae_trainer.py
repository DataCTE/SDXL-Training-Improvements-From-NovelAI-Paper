# src/training/vae_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers import AutoencoderKL
import wandb
import os
from typing import Dict, Optional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
from src.config.config import Config
from src.utils.model import configure_model_memory_format
import gc

logger = logging.getLogger(__name__)

class VAETrainer:
    def __init__(
        self,
        config: Config,
        model: Optional[AutoencoderKL] = None,
        device: Optional[torch.device] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """Initialize VAE trainer."""
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = model or AutoencoderKL.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Configure model memory format
        self.model = configure_model_memory_format(self.model, config)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.optimizer_betas,
            weight_decay=config.training.weight_decay,
            eps=config.training.optimizer_eps
        )
        
        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        try:
            # Move batch to device
            images = batch["pixel_values"].to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            loss = self.model(images, return_dict=False)[0]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in train_step: {str(e)}")
            raise
            
    def train_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Setup progress bar
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not (dist.get_rank() == 0 if dist.is_initialized() else True)
        )
        
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log metrics
            if self.global_step % self.config.training.log_steps == 0:
                self.log_metrics({
                    "train/loss": loss,
                    "train/epoch": epoch,
                    "train/step": self.global_step,
                })
                
            self.global_step += 1
            
        return total_loss / num_batches if num_batches > 0 else float('inf')
        
    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save model checkpoint."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f"vae_checkpoint_epoch_{epoch:04d}.pt")
            
            # Save checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'global_step': self.global_step,
                'config': self.config.to_dict()
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
            
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console."""
        if self.config.training.use_wandb and (dist.get_rank() == 0 if dist.is_initialized() else True):
            wandb.log(metrics, step=self.global_step)
            
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step}: {metrics_str}")