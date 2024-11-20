import torch
import logging
from typing import Dict, Any, Optional, Tuple
from torch.cuda.amp import GradScaler
from src.config.args import TrainingConfig
from src.training.training_steps import GradientAccumulator  # Add this import
from src.training.training_steps import train_step
from src.training.training_utils import initialize_training_components
from src.training.validation import generate_validation_images
from src.training.metrics import MetricsManager
from src.training.ema import setup_ema_model
from src.utils.progress import ProgressTracker
from src.models.StateTracker import StateTracker
import wandb
import os
from src.models.model_loader import save_diffusers_format, save_checkpoint
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
import warnings
from src.utils.logging import SDXLTrainingLogger
from dataclasses import asdict
#D:\SDXL-Training-Improvements-From-NovelAI-Paper\src\models\SDXL\scheduler.py
from src.models.SDXL.scheduler import EDMEulerScheduler
from src.training.optimizers.lion.__init__ import Lion
# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Setuptools is replacing distutils.*")

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
        """Initialize SDXL trainer."""
        self._cleanup_handlers = []
        
        self.config = config
        self.models = models
        self.device = torch.device(device)
        self.dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
        
        # Initialize dataloaders
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        self.val_dataloader = self._setup_dataloader(val_dataloader) if val_dataloader else None
        
        # Initialize components
        self.components = initialize_training_components(config, models)
        
        # Setup optimizers based on config type
        if config.optimizer.optimizer_type == "lion":
            self.optimizers = {
                "unet": Lion(
                    self.models["unet"].parameters(),
                    lr=config.optimizer.learning_rate,
                    weight_decay=config.optimizer.weight_decay,
                    betas=config.optimizer.lion_betas  # NAI recommended (0.95, 0.98)
                )
            }
        else:
            self.optimizers = {
                "unet": torch.optim.AdamW(
                    self.models["unet"].parameters(),
                    lr=config.optimizer.learning_rate,
                    betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
                    weight_decay=config.optimizer.weight_decay,
                    eps=config.optimizer.adam_epsilon
                )
            }
        
        if isinstance(self.optimizers["unet"], Lion):
            self.schedulers = {
                "unet": torch.optim.lr_scheduler.LinearLR(
                    self.optimizers["unet"],
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=config.num_epochs * len(train_dataloader)
                )
            }
        else:
            self.schedulers = {
                "unet": torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizers["unet"],
                    T_max=config.num_epochs * len(self.train_dataloader),
                    eta_min=config.min_learning_rate
                )
            }
        
        # Setup gradient accumulation
        self.grad_accumulator = GradientAccumulator(
            accumulation_steps=config.gradient_accumulation_steps
        ) if config.gradient_accumulation_steps > 1 else None
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None
        
        # Setup metrics and logging
        self.metrics_manager = MetricsManager()
        self.state_tracker = StateTracker()
        
        # Create wandb config dictionary
        wandb_config = {
            "project": config.wandb.project,
            "run_name": config.wandb.run_name,
            "config": asdict(config),  # Convert config to dict
            "log_model": config.wandb.log_model
        }
        
        # Update logger initialization with wandb config
        self.logger = SDXLTrainingLogger(
            log_dir=config.output_dir,
            use_wandb=config.wandb.use_wandb,
            log_frequency=config.wandb.logging_steps,
            window_size=config.wandb.window_size,
            wandb_config=wandb_config
        )
        
        # Remove duplicate WandB setup since it's handled by SDXLTrainingLogger
        self.wandb_run = self.logger.wandb_run if config.wandb.use_wandb else None
        
        # Setup progress tracking
        self.progress = ProgressTracker(
            description="SDXL Training",
            total_steps=config.num_epochs * len(train_dataloader),
            log_steps=10,
            save_steps=1000,
            eval_steps=1000
        )
        
        # Initialize EMA models if enabled
        self.ema_models = {}
        if getattr(config, "use_ema", False):
            ema_model = setup_ema_model(
                model=models["unet"],
                device=self.device,
                power=0.75,  # Default power value
                max_value=getattr(config, "ema_decay", 0.9999),
                update_after_step=getattr(config, "ema_update_after_step", 0),
                inv_gamma=1.0
            )
            if ema_model is not None:
                self.ema_models["unet"] = ema_model
        
        # Register cleanup handlers
        self._setup_cleanup_handlers()
    
    def _setup_dataloader(self, dataloader):
        """Setup and validate the dataloader."""
        if not dataloader:
            return None
            
        try:
            # Get existing settings
            batch_size = dataloader.batch_size
            dataset = dataloader.dataset
            shuffle = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
            
            # Create new dataloader with optimized settings
            # Disable pin_memory if data is already on CUDA
            pin_memory = not any(
                isinstance(item, torch.cuda.FloatTensor) 
                for item in next(iter(dataloader)).values()
                if isinstance(item, torch.Tensor)
            )
            
            # Reset iterator
            dataloader._iterator = None
            
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Disable multiprocessing to avoid pickling issues
                pin_memory=pin_memory,  # Only pin memory for CPU tensors
                collate_fn=getattr(dataloader, 'collate_fn', None),
                persistent_workers=False  # Disable persistent workers
            )
        except Exception as e:
            logger.error(f"Error setting up dataloader: {e}")
            raise
    
    def train(self):
        """Execute training loop with validation."""
        try:
            # Enable full model training
            for model in self.models.values():
                if isinstance(model, torch.nn.Module):
                    model.requires_grad_(True)
                    
            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                self.models["unet"].enable_gradient_checkpointing()
                if self.config.train_text_encoder:
                    self.models["text_encoder"].gradient_checkpointing_enable()
                    self.models["text_encoder_2"].gradient_checkpointing_enable()
            
            progress = ProgressTracker(
                "SDXL Training",
                total=self.config.num_epochs,
                wandb_run=self.wandb_run
            )
            
            for epoch in range(self.config.num_epochs):
                # Training epoch
                epoch_metrics = self.train_epoch(epoch)
                
                # Update EMA models
                if self.ema_models:
                    for ema_model in self.ema_models.values():
                        ema_model.step(epoch)
                
                # Update progress with epoch metrics
                progress.update(1, epoch_metrics)
                
                # Generate validation images
                if epoch % self.config.validation_epochs == 0:
                    self.validate(epoch)
                    
                # Save checkpoint
                if epoch % self.config.save_epochs == 0:
                    # Create pipeline instance for saving
                    pipeline = StableDiffusionXLPipeline(
                        vae=self.models["vae"],
                        text_encoder=self.models["text_encoder"],
                        text_encoder_2=self.models["text_encoder_2"],
                        tokenizer=self.models["tokenizer"],
                        tokenizer_2=self.models["tokenizer_2"],
                        unet=self.models["unet"],
                        scheduler=self.models["scheduler"]
                    )
                    
                    # Save in diffusers format
                    save_diffusers_format(
                        pipeline=pipeline,
                        output_dir=os.path.join(self.config.output_dir, f"epoch_{epoch}"),
                        save_vae=getattr(self.config, "save_vae", True),
                        use_safetensors=getattr(self.config, "use_safetensors", True)
                    )
                    
                    # Optionally save checkpoint
                    if getattr(self.config, "save_checkpoint", False):
                        save_checkpoint(
                            pipeline=pipeline,
                            checkpoint_path=os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.safetensors"),
                            save_vae=getattr(self.config, "save_vae", True),
                            use_safetensors=getattr(self.config, "use_safetensors", True)
                        )
                    
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
    
    def train_epoch(self, epoch: int):
        """Execute single training epoch with enhanced logging."""
        try:
            total_loss = 0
            self.state_tracker.reset()
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Process batch and get metrics
                loss, batch_metrics = train_step(
                    self.config,
                    self.models,
                    self.optimizers,
                    self.schedulers,
                    batch,
                    self.device,
                    dtype=self.dtype,
                    grad_accumulator=self.grad_accumulator,
                    scaler=self.scaler,
                    state_tracker=self.state_tracker
                )
                
                # Update metrics
                total_loss += loss.item()
                
                # Enhanced logging with detailed metrics
                self.logger.log_training_step(
                    loss=loss,
                    metrics=batch_metrics,
                    learning_rate=self.schedulers["unet"].get_last_lr()[0],
                    step=self.state_tracker.global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    step_time=batch_metrics.get("step_time", 0.0),
                    v_pred_loss=batch_metrics.get("v_pred_loss"),
                    v_pred_values=batch_metrics.get("v_pred_values"),
                    v_pred_targets=batch_metrics.get("v_pred_targets"),
                    grad_norm=batch_metrics.get("grad_norm/total"),
                    parameter_metrics=batch_metrics.get("parameter_stats", {})
                )
                
                # Log model state periodically
                if self.state_tracker.global_step % self.config.wandb.logging_steps == 0:
                    self.logger.log_model_state(
                        model=self.models["unet"],
                        epoch=epoch,
                        step=self.state_tracker.global_step
                    )
            
            # Log epoch summary
            epoch_metrics = self.state_tracker.get_epoch_metrics()
            self.logger.log_epoch_summary(epoch, epoch_metrics)
            
            return epoch_metrics
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup resources when trainer is destroyed."""
        try:
            # Reset state tracker
            if hasattr(self, 'state_tracker'):
                self.state_tracker.reset()
            
            # Clear EMA models
            self.ema_models.clear()
            
            # Run cleanup handlers
            for cleanup_handler in self._cleanup_handlers:
                try:
                    cleanup_handler()
                except Exception as e:
                    logger.error(f"Error during cleanup: {str(e)}")
            
            # Clear components and models
            self.components.clear()
            self.models.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during trainer cleanup: {str(e)}")
        
    def _setup_cleanup_handlers(self):
        """Setup cleanup handlers for various components."""
        try:
            # Add model cleanup handlers
            for name, model in self.models.items():
                if hasattr(model, 'cleanup'):
                    self._cleanup_handlers.append(model.cleanup)
        
            # Add EMA cleanup handlers if EMA is enabled
            if hasattr(self, 'ema_models'):
                for ema_model in self.ema_models.values():
                    if hasattr(ema_model, 'cleanup'):
                        self._cleanup_handlers.append(ema_model.cleanup)
            
            # Add component cleanup handlers
            for component in self.components.values():
                if hasattr(component, 'cleanup'):
                    self._cleanup_handlers.append(component.cleanup)
                
            # Add dataloader cleanup if needed
            if hasattr(self.train_dataloader, 'cleanup'):
                self._cleanup_handlers.append(self.train_dataloader.cleanup)
                
        except Exception as e:
            logger.error(f"Error setting up cleanup handlers: {str(e)}")
            raise
    
    def validate(self, epoch: int):
        """Generate sample images using current model state."""
        try:
            # Create pipeline instance
            validation_scheduler = EDMEulerScheduler(
                sigma_min=0.002,
                sigma_max=20000.0,  # NAI's practical infinity
                s_churn=0,
                s_tmin=0,
                s_tmax=float('inf'),
                s_noise=1.0
            )
            
            pipeline = StableDiffusionXLPipeline(
                vae=self.models["vae"],
                text_encoder=self.models["text_encoder"],
                text_encoder_2=self.models["text_encoder_2"],
                tokenizer=self.models["tokenizer"],
                tokenizer_2=self.models["tokenizer_2"],
                unet=self.models["unet"],
                scheduler=validation_scheduler
            )
            
            # Set models to eval mode
            for model in self.models.values():
                if isinstance(model, torch.nn.Module):
                    model.eval()
            
            # Setup validation directory
            val_dir = os.path.join(self.config.output_dir, f"validation_images/epoch_{epoch}")
            os.makedirs(val_dir, exist_ok=True)
            
            # Get validation prompts
            validation_prompts = getattr(self.config, "validation_prompts", [
                "a beautiful sunset over mountains",
                "a cute cat playing with yarn",
                "an astronaut riding a horse on mars"
            ])
            
            progress = ProgressTracker(
                f"Generating Validation Images - Epoch {epoch}",
                total=len(validation_prompts),
                wandb_run=self.wandb_run
            )
            
            generated_paths = generate_validation_images(
                pipeline=pipeline,
                prompts=validation_prompts,
                save_dir=val_dir,
                device=self.device,
                num_inference_steps=getattr(self.config, "validation_num_inference_steps", 28),
                guidance_scale=getattr(self.config, "validation_guidance_scale", 5.0),
                height=getattr(self.config, "validation_image_height", 1024),
                width=getattr(self.config, "validation_image_width", 1024),
                num_images_per_prompt=1,
                progress_callback=progress.update
            )
            
            # Log images to wandb
            if self.wandb_run is not None:
                for path in generated_paths:
                    self.wandb_run.log({
                        f"validation/image_{os.path.basename(path)}": wandb.Image(path)
                    }, step=epoch)
            
            logger.info(f"Generated {len(generated_paths)} validation images for epoch {epoch}")
            
            # Set models back to train mode
            for model in self.models.values():
                if isinstance(model, torch.nn.Module):
                    model.train()
                    
        except Exception as e:
            logger.error(f"Failed to generate validation images at epoch {epoch}: {str(e)}")
            raise
    
    