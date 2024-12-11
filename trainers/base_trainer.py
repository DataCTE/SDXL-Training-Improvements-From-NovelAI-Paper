import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from typing import Optional, Dict, Any
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.checkpoints import CheckpointManager
from utils.error_handling import error_handler

class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        accelerator = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        local_rank: int = -1
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.accelerator = accelerator
        self.local_rank = local_rank
        
        # Wrap model in DDP if using distributed training
        if self.local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    @error_handler
    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint"""
        if self.local_rank == 0:
            state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            self.checkpoint_manager.save_checkpoint(
                model=state_dict,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                global_step=self.global_step,
                name=name
            )
        
    @error_handler
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        if self.local_rank == 0:
            state_dict = torch.load(checkpoint_path)
            if self.local_rank != -1:
                state_dict = self.broadcast_object(state_dict)
            self.current_epoch = state_dict["epoch"]
            self.global_step = state_dict["global_step"]
            self.optimizer.load_state_dict(state_dict["optimizer_state"])
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        raise NotImplementedError
        
    def training_step(self, batch: Any) -> torch.Tensor:
        """Single training step"""
        raise NotImplementedError
        
    @error_handler
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all trackers (only on main process)"""
        if self.local_rank == -1 or self.local_rank == 0:
            if wandb.run:
                wandb.log(metrics, step=self.global_step)
            if self.accelerator:
                self.accelerator.log(metrics, step=self.global_step)
                
    @error_handler
    def compute_grad_norm(self) -> float:
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 