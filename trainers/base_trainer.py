import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from typing import Optional, Dict, Any

from utils.checkpoints import CheckpointManager

class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        accelerator = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.accelerator = accelerator
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def save_checkpoint(self, name: str = None):
        """Save training checkpoint"""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            name=name
        )
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        training_state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        self.current_epoch = training_state["epoch"]
        self.global_step = training_state["global_step"]
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        raise NotImplementedError
        
    def training_step(self, batch: Any) -> torch.Tensor:
        """Single training step"""
        raise NotImplementedError
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all trackers"""
        if self.accelerator and self.accelerator.is_main_process:
            # Log to wandb
            if wandb.run:
                wandb.log(metrics, step=self.global_step)
                
            # Log to tensorboard via accelerator
            self.accelerator.log(metrics, step=self.global_step)
            
    def compute_grad_norm(self) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 