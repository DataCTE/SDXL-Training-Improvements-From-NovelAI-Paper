import torch
import logging
from typing import Dict, Any, Optional
from src.config.args import TrainingConfig
from src.training.training_steps import train_step
from src.training.training_utils import initialize_training_components
from src.training.validation import run_validation
from src.training.metrics import MetricsManager
from src.training.ema import setup_ema_model
from src.utils.progress import ProgressTracker
import wandb

logger = logging.getLogger(__name__)


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
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        self.val_dataloader = self._setup_dataloader(val_dataloader) if val_dataloader else None
        self.device = torch.device(device)
        self.dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
        
        # Initialize components (optimizer, scheduler, etc.)
        self.components = initialize_training_components(config, models)
        self.metrics_manager = MetricsManager()
        
        # Move only torch.nn.Module models to device
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
                model.train()  # Ensure training mode
            
        # Setup EMA if enabled in config
        if hasattr(config, 'use_ema') and config.use_ema:
            self.ema_model = setup_ema_model(
                model=self.models['unet'],  # Usually EMA is applied to UNet
                device=self.device,
                power=getattr(config, 'ema_power', 0.75),
                max_value=getattr(config, 'ema_max_value', 0.9999),
                update_after_step=getattr(config, 'ema_update_after_step', 0)
            )
        else:
            self.ema_model = None
        
        # Initialize wandb if enabled
        self.wandb_run = None
        if hasattr(config, 'use_wandb') and config.use_wandb:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
    def _setup_dataloader(self, dataloader):
        """Setup and validate the dataloader."""
        if not dataloader:
            return None
            
        if torch.cuda.is_available():
            # Get shuffle parameter from dataloader config instead of attribute
            shuffle = False
            if isinstance(dataloader, torch.utils.data.DataLoader):
                # Access the dataset's sampler to determine if shuffling is enabled
                shuffle = dataloader.sampler is None or isinstance(
                    dataloader.sampler, 
                    torch.utils.data.RandomSampler
                )
            
            return torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=shuffle,
                num_workers=dataloader.num_workers,
                multiprocessing_context='spawn',
                persistent_workers=True,  # Keep workers alive between iterations
                pin_memory=True  # Enable pinned memory for faster GPU transfer
            )
        return dataloader
        
    def train(self, save_dir: str):
        """Execute training loop with validation."""
        with ProgressTracker(
            "SDXL Training",
            total=self.config.num_epochs,
            wandb_run=self.wandb_run
        ) as progress:
            for epoch in range(self.config.num_epochs):
                # Training epoch
                epoch_metrics = self.train_epoch(epoch)
                
                # Update progress with epoch metrics
                progress.update(1, epoch_metrics)
                
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
        
        # Set models to training mode
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.train()
        
        with ProgressTracker(
            f"Epoch {epoch}",
            total=num_batches,
            wandb_run=self.wandb_run,
            log_interval=0.1  # Log every 10% of batches
        ) as progress:
            for batch in self.train_dataloader:
                # Move batch to device and handle text embeddings
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Training step
                loss, metrics_dict = train_step(
                    self.config,
                    self.models,
                    self.components['optimizer'],
                    self.components['scheduler'],
                    batch,
                    self.device,
                    self.dtype,
                    self.components['grad_accumulator'],
                    self.components['scaler']
                )
                
                total_loss += loss.item()
                
                # Update metrics and progress
                metrics_dict['loss'] = loss.item()
                metrics_dict['learning_rate'] = self.components['scheduler'].get_last_lr()[0]
                progress.update(1, metrics_dict)
        
        avg_loss = total_loss / num_batches
        return {'epoch': epoch, 'avg_loss': avg_loss, **metrics_dict}
        
    def validate(self, epoch: int):
        """Run validation loop."""
        # Set models to eval mode
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.eval()

        with ProgressTracker(
            f"Validation Epoch {epoch}",
            total=len(self.val_dataloader) if self.val_dataloader else 0,
            wandb_run=self.wandb_run
        ) as progress:
            metrics = run_validation(
                self.config,
                self.models,
                self.components,
                self.device,
                self.dtype,
                epoch,
                self.metrics_manager,
                progress_callback=progress.update
            )
            
            if self.wandb_run:
                # Log validation metrics to wandb and images
                self.wandb_run.log(metrics)
                self.wandb_run.log({
                    'validation_images': [wandb.Image(image_path) for image_path in metrics['validation_images']]
                })
        
    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save training checkpoint."""
        with ProgressTracker(
            f"Saving checkpoint for epoch {epoch}",
            total=1,
            wandb_run=self.wandb_run
        ) as progress:
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
            
            if self.wandb_run:
                artifact = wandb.Artifact(
                    name=f"checkpoint-epoch-{epoch}",
                    type="model"
                )
                artifact.add_file(save_path)
                self.wandb_run.log_artifact(artifact)
            
            progress.update(1, {
                'status': 'Checkpoint saved',
                'path': save_path
            })
        
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
        
    def _setup_ema_model(self, model):
        """Properly set up EMA model."""
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model