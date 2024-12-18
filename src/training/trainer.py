"""NovelAI Diffusion V3 Trainer with optimized memory usage."""
import torch
import os
import logging
import time
import gc
from typing import Dict, Optional, Union, Tuple, Any
from diffusers import UNet2DConditionModel, AutoencoderKL
from torch.utils.data import DataLoader
import wandb
import traceback
from src.config.config import Config
from src.data.dataset import NovelAIDataset
from src.utils.system.setup import setup_memory_optimizations, verify_memory_optimizations
from src.utils.model.model import configure_model_memory_format
from src.training.scheduler import configure_noise_scheduler, get_karras_scalings
from src.utils.model.noise import generate_noise
from src.utils.model.embeddings import get_add_time_ids
from src.utils.logging.metrics import log_metrics as utils_log_metrics

logger = logging.getLogger(__name__)

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    """Trainer for NovelAI Diffusion V3 with optimized memory usage."""
    
    def __init__(
        self,
        config: Config,
        model: Optional[UNet2DConditionModel] = None,
        dataset: Optional[NovelAIDataset] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer with optimized memory usage."""
        super().__init__()
        
        # Store config
        self.config = config
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and move to device
        self.model = model
        if self.model is None:
            self.model = UNet2DConditionModel.from_pretrained(
                self.config.model.pretrained_model_name,
                subfolder="unet"
            )
        self.model.to(self.device)
        
        # Configure noise scheduler and get parameters
        try:
            scheduler_params = configure_noise_scheduler(self.config, self.device)
            self.scheduler = scheduler_params['scheduler']
            self.sigmas = scheduler_params['sigmas']
            self.snr_weights = scheduler_params['snr_weights']
            self.c_skip = scheduler_params['c_skip']
            self.c_out = scheduler_params['c_out']
            self.c_in = scheduler_params['c_in']
            self.num_train_timesteps = self.config.model.num_timesteps
            logger.info("Successfully configured noise scheduler")
        except Exception as e:
            logger.error(f"Failed to configure noise scheduler: {str(e)}")
            raise
        
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        
        # Get model dtype for consistent tensor operations
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Configure model format and optimizations
        if not self._setup_model_optimizations():
            logger.warning("Training will proceed without memory optimizations")
            
        # Setup dataset and dataloader
        self.dataset = dataset
        if self.dataset is not None:
            # Use dataset's built-in sampler and dataloader creation
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=self.dataset.get_sampler(),
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                persistent_workers=config.data.persistent_workers
            )
            
            # Calculate max_train_steps based on dataset size
            steps_per_epoch = len(self.dataset) // self.config.training.batch_size
            self.max_train_steps = steps_per_epoch * self.config.training.num_epochs
            logger.info(f"Training will run for {self.max_train_steps} steps")
        else:
            self.dataloader = None
            self.max_train_steps = None
            logger.warning("No dataset provided, trainer initialized without data")

    def _setup_model_optimizations(self):
        """Setup model memory optimizations."""
        try:
            # Configure memory format
            configure_model_memory_format(
                model=self.model,
                config=self.config
            )
            
            # Setup memory optimizations
            batch_size = self.config.training.batch_size
            micro_batch_size = batch_size // self.config.training.gradient_accumulation_steps
            
            memory_setup = setup_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size
            )
            
            # Verify memory setup was successful
            if not memory_setup:
                logger.warning("Memory optimizations setup failed")
                return False
            
            # Verify optimizations
            optimization_states = verify_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=logger
            )
            
            if not all(optimization_states.values()):
                logger.warning("Some memory optimizations failed to initialize")
                return False
                
            logger.info("Memory optimizations setup completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error setting up memory optimizations: {e}")
            raise

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        try:
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.optimizer_betas,
                weight_decay=self.config.training.weight_decay,
                eps=self.config.training.optimizer_eps
            )
            
            # Calculate max_train_steps if not set
            if self.config.training.max_train_steps is None:
                logger.info("max_train_steps not set, will be calculated when dataloader is provided")
                self.max_train_steps = None
            else:
                self.max_train_steps = self.config.training.max_train_steps
            
            # Create scheduler if enabled
            if self.config.training.lr_scheduler != "none":
                if self.max_train_steps is None:
                    logger.warning("Cannot create scheduler yet as max_train_steps is not set")
                    self.lr_scheduler = None
                    return
                
                if self.config.training.lr_scheduler == "cosine":
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.max_train_steps - self.config.training.warmup_steps
                    )
                elif self.config.training.lr_scheduler == "linear":
                    self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1.0,
                        end_factor=0.1,
                        total_iters=self.max_train_steps - self.config.training.warmup_steps
                    )
            else:
                self.lr_scheduler = None
                
        except Exception as e:
            logger.error(f"Error setting up optimizer: {e}")
            raise

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted loss with SNR scaling."""
        # Basic MSE in float32 for stability
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        
        # Average over non-batch dimensions
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        # Apply SNR weights if enabled
        if self.config.training.snr_gamma is not None and self.snr_weights is not None:
            snr = self.snr_weights[timesteps]
            loss = loss * snr
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
            
        return loss.mean()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute training step with memory optimization."""
        try:
            # Track memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated() / 1024**2
            
            # Get model inputs and embeddings
            model_input = batch["latents"].to(self.device, dtype=self.model_dtype, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(self.device, non_blocking=True)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, non_blocking=True)
            
            # Get SDXL conditioning
            add_time_ids = get_add_time_ids(
                original_sizes=batch["original_size"],
                crops_coords_top_lefts=batch["crop_top_left"],
                target_sizes=[(self.config.data.image_size[0], self.config.data.image_size[1])] * len(model_input),
                dtype=self.model_dtype,
                device=self.device
            )
            
            # Apply tag weights if available
            if "tag_weights" in batch:
                tag_weights = batch["tag_weights"].to(self.device, non_blocking=True)
                prompt_embeds = prompt_embeds * tag_weights.unsqueeze(-1)
            
            # Sample timesteps
            timesteps = torch.randint(0, self.num_train_timesteps, (model_input.shape[0],), device=self.device)
            
            # Generate noise
            noise = generate_noise(
                model_input.shape,
                self.device,
                self.model_dtype,
                model_input  # Use model_input as layout template
            )
            
            # Get Karras noise schedule scalings
            c_skip, c_out, c_in = get_karras_scalings(self.sigmas, timesteps)
            c_skip = c_skip.to(dtype=self.model_dtype)
            c_out = c_out.to(dtype=self.model_dtype)
            c_in = c_in.to(dtype=self.model_dtype)
            
            # Get sigmas and add noise
            sigmas = self.sigmas[timesteps].to(dtype=self.model_dtype)
            noisy_latents = model_input + sigmas[:, None, None, None] * noise
            
            # Scale input based on prediction type
            if self.config.training.prediction_type == "v_prediction":
                scaled_input = c_in[:, None, None, None] * noisy_latents
            else:
                scaled_input = noisy_latents
                
            # Forward pass
            model_pred = self.model(
                scaled_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
                return_dict=False
            )[0]
            
            # Get target based on prediction type
            if self.config.training.prediction_type == "epsilon":
                target = noise
            else:  # v_prediction
                target = (
                    c_skip[:, None, None, None] * model_input +
                    c_out[:, None, None, None] * noise
                )
            
            # Compute loss
            loss = self.compute_loss(model_pred, target, timesteps, batch["tag_weights"])
            
            # Backward pass
            loss.backward()
            
            # Log memory usage
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                mem_diff = peak_mem - start_mem
                logger.debug(f"Memory usage: {peak_mem:.0f}MB (+{mem_diff:.0f}MB)")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        if self.dataloader is None:
            raise ValueError("Dataloader not set. Call set_dataloader() first.")
        
        for batch_idx, batch in enumerate(self.dataloader):
            try:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Accumulate gradients
                for i in range(self.config.training.gradient_accumulation_steps):
                    loss = self.training_step(batch)
                    total_loss += loss.item()
                
                # Update weights
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update progress
                num_batches += 1
                self.global_step += 1
                
                # Log metrics
                if self.global_step % self.config.training.log_steps == 0:
                    self.log_metrics({
                        'loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.global_step
                    })
                
                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(self.config.paths.checkpoints_dir, epoch)
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Log epoch metrics
        self.log_metrics({
            'epoch': epoch,
            'epoch_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }, step_type="epoch")

    def save_checkpoint(self, checkpoints_dir: str, epoch: int) -> None:
        """Save model checkpoint."""
        try:
            checkpoint_path = os.path.join(
                checkpoints_dir,
                f"checkpoint_epoch_{epoch:03d}_step_{self.global_step:06d}.pt"
            )
            
            # Save checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'epoch': epoch
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, Any], step_type: str = "step") -> None:
        """Log metrics using utility function."""
        utils_log_metrics(
            metrics=metrics,
            step=self.global_step,
            is_main_process=True,  # TODO: Add distributed training support
            use_wandb=self.config.training.use_wandb,
            step_type=step_type
        )

    def set_dataloader(self, dataloader: DataLoader) -> None:
        """Set the dataloader."""
        self.dataloader = dataloader

    def __del__(self):
        """Remove wandb cleanup since it's handled by cleanup_logging"""
        pass

