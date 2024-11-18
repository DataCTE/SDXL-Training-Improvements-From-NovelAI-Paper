import torch
import logging
from typing import Dict, Any, Optional
from src.config.args import TrainingConfig
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
        """Initialize SDXL trainer.
        
        Args:
            config: Training configuration
            models: Dictionary of model components
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            device: Target device for training
        """
        # Initialize cleanup handlers first
        self._cleanup_handlers = []
        
        self.config = config
        self.models = models
        self.device = torch.device(device)
        self.dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
        
        # Initialize with single process dataloader
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        self.val_dataloader = self._setup_dataloader(val_dataloader) if val_dataloader else None
        
        # Initialize components with updated config
        self.components = initialize_training_components(config, models)
        self.metrics_manager = MetricsManager()
        
        # Setup progress tracking
        self.progress = ProgressTracker(
            description="SDXL Training",
            total_steps=config.num_epochs * len(train_dataloader),
            log_steps=10,
            save_steps=1000,
            eval_steps=1000
        )
        
        # Move models to device
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
                if config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
                    model.enable_gradient_checkpointing()
                model.train()
        
        # Setup EMA models if enabled
        self.ema_models = {}
        if config.use_ema:
            for name, model in self.models.items():
                if isinstance(model, torch.nn.Module) and name in ['unet', 'text_encoder', 'text_encoder_2']:
                    ema_model = setup_ema_model(
                        model=model,
                        device=self.device,
                        power=0.75,
                        update_after_step=config.ema.update_after_step,
                        max_value=0.9999,
                        min_value=0,
                        inv_gamma=1
                    )
                    if ema_model is not None:
                        self.ema_models[name] = ema_model
        
        # Initialize wandb
        self.wandb_run = None
        if config.wandb.use_wandb:
            self.wandb_run = wandb.init(
                project=config.wandb.wandb_project,
                name=config.wandb.wandb_run_name,
                config=vars(config)
            )
        
        # Initialize state tracker without adding callback
        self.state_tracker = StateTracker()
        
        # Setup cleanup handlers
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
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Disable multiprocessing to avoid pickling issues
                pin_memory=True,
                collate_fn=getattr(dataloader, 'collate_fn', None)
            )
        except Exception as e:
            logger.error(f"Error setting up dataloader: {e}")
            raise
    
    def train(self, save_dir: str):
        """Execute training loop with validation."""
        try:
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
                        output_dir=os.path.join(save_dir, f"epoch_{epoch}"),
                        save_vae=getattr(self.config, "save_vae", True),
                        use_safetensors=getattr(self.config, "use_safetensors", True)
                    )
                    
                    # Optionally save checkpoint
                    if getattr(self.config, "save_checkpoint", False):
                        save_checkpoint(
                            pipeline=pipeline,
                            checkpoint_path=os.path.join(save_dir, f"checkpoint_epoch_{epoch}.safetensors"),
                            save_vae=getattr(self.config, "save_vae", True),
                            use_safetensors=getattr(self.config, "use_safetensors", True)
                        )
                    
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
    
    def train_epoch(self, epoch: int):
        """Execute single training epoch."""
        try:
            total_loss = 0
            num_batches = len(self.train_dataloader)
            
            # Reset state tracker at start of epoch
            self.state_tracker.reset()
            
            # Set models to training mode
            for model in self.models.values():
                if isinstance(model, torch.nn.Module):
                    model.train()
            
            progress = ProgressTracker(
                f"Epoch {epoch}",
                total=num_batches,
                wandb_run=self.wandb_run,
                log_interval=0.1
            )
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Store text encoder outputs in state tracker
                    if 'text_encoder_hidden_states' in batch:
                        self.state_tracker.store_text_encoder_outputs(
                            text_encoder_hidden_states=batch.get('text_encoder_hidden_states'),
                            text_encoder_2_hidden_states=batch.get('text_encoder_2_hidden_states'),
                            prompt_embeds=batch.get('prompt_embeds'),
                            pooled_prompt_embeds=batch.get('pooled_prompt_embeds')
                        )
                    
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
                        self.components['scaler'],
                        state_tracker=self.state_tracker
                    )
                    
                    # Get state from tracker and add to metrics
                    state = self.state_tracker.get_state()
                    if state['latents'] is not None:
                        metrics_dict.update({
                            'current_timestep': state['current_timestep'],
                            'latents_mean': state['latents'].mean().item(),
                            'latents_std': state['latents'].std().item(),
                        })
                    
                    total_loss += loss.item()
                    
                    # Update metrics and progress
                    metrics_dict['loss'] = loss.item()
                    metrics_dict['learning_rate'] = self.components['scheduler'].get_last_lr()[0]
                    progress.update(1, metrics_dict)
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                    continue
            
            avg_loss = total_loss / num_batches
            return {'epoch': epoch, 'avg_loss': avg_loss, **metrics_dict}
            
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
            
            # Add EMA cleanup handlers
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
            if hasattr(self.val_dataloader, 'cleanup'):
                self._cleanup_handlers.append(self.val_dataloader.cleanup)
            
            # Add metrics manager cleanup
            if hasattr(self.metrics_manager, 'cleanup'):
                self._cleanup_handlers.append(self.metrics_manager.cleanup)
            
            # Add wandb cleanup if active
            if self.wandb_run is not None:
                self._cleanup_handlers.append(self.wandb_run.finish)
            
        except Exception as e:
            logger.error(f"Error setting up cleanup handlers: {str(e)}")
            raise
    
    def validate(self, epoch: int):
        """Generate sample images using current model state."""
        try:
            # Create pipeline instance
            pipeline = StableDiffusionXLPipeline(
                vae=self.models["vae"],
                text_encoder=self.models["text_encoder"],
                text_encoder_2=self.models["text_encoder_2"],
                tokenizer=self.models["tokenizer"],
                tokenizer_2=self.models["tokenizer_2"],
                unet=self.models["unet"],
                scheduler=self.models["scheduler"]
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
    
    