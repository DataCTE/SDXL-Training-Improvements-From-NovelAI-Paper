"""Ultra-optimized SDXL trainer implementing NAI's improvements."""

import torch
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
from torch.cuda.amp import GradScaler
from src.config.args import TrainingConfig
from src.training.training_steps import train_step
from src.training.training_utils import initialize_training_components
from src.training.metrics import MetricsManager
from src.training.ema import setup_ema_model
from src.utils.progress import ProgressTracker
from src.models.StateTracker import StateTracker
from src.utils.logging import SDXLTrainingLogger
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
from src.models.SDXL.scheduler import EDMEulerScheduler
from src.training.optimizers.lion import Lion
import os

logger = logging.getLogger(__name__)

class SDXLTrainer:
    """SDXL trainer with NAI improvements."""
    
    def __init__(
        self,
        config: TrainingConfig,
        models: Dict[str, Any],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda",
    ):
        """
        Initialize SDXL trainer with NAI improvements:
        1. v-prediction parameterization (Section 2.1)
        2. ZTSNR noise schedule (Section 2.2)
        3. Resolution-based sigma scaling (Section 2.3)
        4. MinSNR loss weighting (Section 2.4)
        """
        self.config = config
        self.models = models
        self.device = torch.device(device)
        self.dtype = torch.float16  # NAI: trained in float16 with tf32
        
        # Initialize dataloaders with pin memory
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Initialize components
        self.components = initialize_training_components(config, models)
        
        # Setup Lion optimizer per NAI
        self.optimizers = {
            "unet": Lion(
                self.models["unet"].parameters(),
                lr=config.optimizer.learning_rate,
                weight_decay=config.optimizer.weight_decay,
                betas=(0.95, 0.98)  # NAI recommended
            )
        }
        
        # Linear LR schedule
        self.schedulers = {
            "unet": torch.optim.lr_scheduler.LinearLR(
                self.optimizers["unet"],
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.num_epochs * len(train_dataloader)
            )
        }
        
        # Setup mixed precision with tf32
        torch.backends.cuda.matmul.allow_tf32 = True  # NAI: used tf32
        torch.backends.cudnn.allow_tf32 = True
        self.scaler = GradScaler()
        
        # Setup metrics and logging
        self.metrics_manager = MetricsManager()
        self.state_tracker = StateTracker()
        
        # Initialize logger
        wandb_config = {
            "project": config.wandb.project,
            "run_name": config.wandb.run_name,
            "config": asdict(config),
            "log_model": config.wandb.log_model
        }
        
        self.logger = SDXLTrainingLogger(
            log_dir=config.output_dir,
            use_wandb=config.wandb.use_wandb,
            log_frequency=config.wandb.logging_steps,
            window_size=config.wandb.window_size,
            wandb_config=wandb_config
        )
        
        self.wandb_run = self.logger.wandb_run if config.wandb.use_wandb else None
        
        # Setup progress tracking
        self.progress = ProgressTracker(
            description="SDXL Training",
            total_steps=config.num_epochs * len(train_dataloader),
            log_steps=config.wandb.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps
        )
        
        # Initialize EMA
        self.ema_models = {}
        if config.use_ema:
            ema_model = setup_ema_model(
                model=models["unet"],
                device=self.device,
                power=0.75,
                max_value=config.ema_decay,
                update_after_step=config.ema_update_after_step
            )
            if ema_model is not None:
                self.ema_models["unet"] = ema_model

    def train(self):
        """Execute training loop with NAI improvements."""
        try:
            # Enable training mode
            self.models["unet"].train()
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                self.models["unet"].enable_gradient_checkpointing()
            
            for epoch in range(self.config.num_epochs):
                # Training epoch
                epoch_metrics = self.train_epoch(epoch)
                
                # Update EMA models
                if self.ema_models:
                    for ema_model in self.ema_models.values():
                        ema_model.step(self.state_tracker.global_step)
                
                # Log epoch metrics
                self.logger.log_epoch_summary(epoch, epoch_metrics)
                
                
                    
                # Save checkpoint
                if epoch % self.config.save_epochs == 0:
                    self.save_checkpoint(epoch)
                    
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute single training epoch with NAI improvements."""
        self.state_tracker.reset()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Process batch with NAI improvements
            loss, metrics = train_step(
                model_dict=self.models,
                batch=batch,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                scaler=self.scaler,
                state_tracker=self.state_tracker,
                num_inference_steps=1000,  # NAI's full schedule length
                min_snr_gamma=5.0,  # NAI's MinSNR value
                device=self.device,
                dtype=self.dtype
            )
            
            # Log step metrics
            self.logger.log_training_step(
                loss=loss,
                metrics=metrics,
                learning_rate=self.schedulers["unet"].get_last_lr()[0],
                step=self.state_tracker.global_step,
                epoch=epoch,
                batch_idx=batch_idx,
                step_time=self.state_tracker.get_step_time(),
                v_pred_loss=metrics.get("v_pred_loss"),
                v_pred_values=None,  # Optional detailed logging
                v_pred_targets=None  # Optional detailed logging
            )
            
            # Log model state periodically
            if self.state_tracker.global_step % self.config.wandb.logging_steps == 0:
                self.logger.log_model_state(
                    model=self.models["unet"],
                    epoch=epoch,
                    step=self.state_tracker.global_step
                )
        
        return self.state_tracker.get_epoch_metrics()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        pipeline = StableDiffusionXLPipeline(
            vae=self.models["vae"],
            text_encoder=self.models["text_encoder"],
            text_encoder_2=self.models["text_encoder_2"],
            tokenizer=self.models["tokenizer"],
            tokenizer_2=self.models["tokenizer_2"],
            unet=self.models["unet"],
            scheduler=EDMEulerScheduler()  # NAI: EDM scheduler
        )
        
        save_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}")
        pipeline.save_pretrained(save_path)

    def __del__(self):
        """Cleanup resources."""
        self.state_tracker.reset()
        self.ema_models.clear()
        self.components.clear()
        self.models.clear()
        torch.cuda.empty_cache()