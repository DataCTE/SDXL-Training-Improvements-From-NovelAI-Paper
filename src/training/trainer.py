import torch
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.training.training_steps import train_step
from src.training.training_utils import initialize_training_components
from src.training.validation import generate_validation_images
from src.training.metrics import MetricsManager

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 1e-5
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    use_8bit_adam: bool = False
    use_ema: bool = True
    ema_decay: float = 0.9999
    train_text_encoder: bool = False
    validation_epochs: int = 5
    save_epochs: int = 10

class SDXLTrainer:
    """Main trainer class for SDXL model improvements."""
    
    def __init__(
        self,
        config: TrainingConfig,
        models: Dict[str, Any],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.models = models
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
        
        # Initialize components (optimizer, scheduler, etc.)
        self.components = initialize_training_components(config, models)
        self.metrics_manager = MetricsManager()
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
            
    def train(self, save_dir: str):
        """Execute training loop with validation."""
        for epoch in range(self.config.num_epochs):
            # Training epoch
            self.train_epoch(epoch)
            
            # Validation
            if self.val_dataloader and epoch % self.config.validation_epochs == 0:
                self.validate(epoch)
                
            # Save checkpoint
            if epoch % self.config.save_epochs == 0:
                self.save_checkpoint(save_dir, epoch)
                
    def train_epoch(self, epoch: int):
        """Execute single training epoch."""
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss_dict = train_step(
                self.config,
                self.models,
                self.components,
                batch,
                batch_idx,
                self.device,
                self.dtype,
                self.metrics_manager
            )
            
            total_loss += loss_dict["total_loss"]
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{num_batches}]: "
                          f"Loss: {loss_dict['total_loss']:.4f}")
                
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
    def validate(self, epoch: int):
        """Run validation loop."""
        generate_validation_images(
            self.config,
            self.models,
            self.val_dataloader,
            self.device,
            self.dtype
        )
        logger.info(f"Validation images saved for epoch {epoch}")
        
    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': {name: model.state_dict() 
                          for name, model in self.models.items()},
            'optimizer_state': self.components['optimizer'].state_dict(),
            'config': self.config,
        }
        
        if 'ema_model' in self.components:
            checkpoint['ema_state'] = self.components['ema_model'].state_dict()
            
        save_path = f"{save_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, train_dataloader, val_dataloader=None):
        """Load training checkpoint and reconstruct trainer."""
        checkpoint = torch.load(checkpoint_path)
        
        # Reconstruct trainer
        trainer = cls(
            config=checkpoint['config'],
            models={name: type(model)() for name, model in checkpoint['model_state'].items()},
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        # Load states
        for name, state in checkpoint['model_state'].items():
            trainer.models[name].load_state_dict(state)
        trainer.components['optimizer'].load_state_dict(checkpoint['optimizer_state'])
        
        if 'ema_state' in checkpoint and 'ema_model' in trainer.components:
            trainer.components['ema_model'].load_state_dict(checkpoint['ema_state'])
            
        return trainer