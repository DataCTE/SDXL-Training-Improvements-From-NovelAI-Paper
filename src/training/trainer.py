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
        self.device = torch.device(device)
        self.dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
        
        # Initialize with single process dataloader
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        self.val_dataloader = self._setup_dataloader(val_dataloader) if val_dataloader else None
        
        # Initialize components
        self.components = initialize_training_components(config, models)
        self.metrics_manager = MetricsManager()
        
        # Move models to device
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
                model.train()
    
        # Initialize wandb
        self.wandb_run = None
        if hasattr(config, 'use_wandb') and config.use_wandb:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
    
        # Initialize cleanup handlers
        self._cleanup_handlers = []
        self._setup_cleanup_handlers()
                
    def __del__(self):
        """Cleanup resources when trainer is destroyed."""
        try:
            # Run all cleanup handlers
            for cleanup_handler in self._cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    logger.error(f"Error during cleanup: {str(e)}")
            
            # Clear components
            self.components.clear()
            
            # Clear models
            self.models.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during trainer cleanup: {str(e)}")
            
    def _setup_dataloader(self, dataloader):
        """Setup and validate the dataloader."""
        if not dataloader:
            return None
            
        try:
            if torch.cuda.is_available():
                # Get existing settings
                batch_size = dataloader.batch_size
                dataset = dataloader.dataset
                
                
                # Create new dataloader with single process if there are pickling issues
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
                    num_workers=0,  # Force single process to avoid pickling issues
                    pin_memory=True,
                    collate_fn=getattr(dataloader, 'collate_fn', None)
                )
            return dataloader
        except Exception as e:
            logger.error(f"Error setting up dataloader: {e}")
            raise
        
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
        try:
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
                log_interval=0.1
            ) as progress:
                for batch_idx, batch in enumerate(self.train_dataloader):
                    try:
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
                        
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                        continue  # Skip to next batch on error
            
            avg_loss = total_loss / num_batches
            return {'epoch': epoch, 'avg_loss': avg_loss, **metrics_dict}
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            raise
        
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
        try:
            if not isinstance(model, torch.nn.Module):
                raise ValueError("Model must be a torch.nn.Module instance")
            
            # Create a deep copy on the same device
            ema_model = type(model)().to(model.device)
            ema_model.load_state_dict(model.state_dict())
            
            # Ensure no gradients are computed for EMA model
            for param in ema_model.parameters():
                param.requires_grad_(False)
            
            return ema_model
        except Exception as e:
            logger.error(f"Failed to set up EMA model: {str(e)}")
            return None