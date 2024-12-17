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
from src.utils.setup import setup_memory_optimizations, verify_memory_optimizations
from src.utils.model import configure_model_memory_format
from src.training.scheduler import configure_noise_scheduler, get_karras_scalings
from src.utils.noise import generate_noise
from src.utils.embeddings import get_add_time_ids

logger = logging.getLogger(__name__)

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    """Trainer for NovelAI Diffusion V3 with optimized memory usage."""
    
    def __init__(
        self,
        config_path: str,
        model: Optional[UNet2DConditionModel] = None,
        vae: Optional[AutoencoderKL] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer with optimized memory usage."""
        super().__init__()
        
        # Load and validate config
        self.config = Config.load(config_path)
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and move to device
        self.model = model
        if self.model is None:
            self.model = UNet2DConditionModel.from_pretrained(
                self.config.model.pretrained_model_name_or_path,
                subfolder="unet"
            )
        self.model.to(self.device)
        
        # Get model dtype
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Configure memory format
        configure_model_memory_format(
            model=self.model,
            channels_last=self.config.system.channels_last
        )
        
        # Setup memory optimizations
        try:
            batch_size = self.config.training.batch_size
            micro_batch_size = batch_size // self.config.training.gradient_accumulation_steps
            
            # Set up memory optimizations
            memory_setup = setup_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size
            )
            
            # Verify optimizations
            optimization_states = verify_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=logger
            )
            
            if not all(optimization_states.values()):
                logger.warning("Some memory optimizations failed to initialize")
                
        except Exception as e:
            logger.error(f"Error setting up memory optimizations: {e}")
            raise
            
        # Configure noise scheduler
        try:
            scheduler_params = configure_noise_scheduler(self.config, self.device)
            self.scheduler = scheduler_params['scheduler']
            self.sigmas = scheduler_params['sigmas']
            self.alphas = scheduler_params['alphas']
            self.betas = scheduler_params['betas']
            self.alphas_cumprod = scheduler_params['alphas_cumprod']
            self.snr_values = scheduler_params['snr_values']
            self.snr_weights = scheduler_params['snr_weights']
            self.c_skip = scheduler_params['c_skip']
            self.c_out = scheduler_params['c_out']
            self.c_in = scheduler_params['c_in']
            self.num_train_timesteps = len(self.sigmas)
        except Exception as e:
            logger.error(f"Error configuring scheduler: {e}")
            raise
            
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize optimizer and scheduler
        self.setup_optimizer()
        
        logger.info(
            f"Initialized trainer on {self.device}\n"
            f"Model dtype: {self.model_dtype}"
        )

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        try:
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.adam_betas,
                weight_decay=self.config.training.weight_decay,
                eps=self.config.training.adam_epsilon
            )
            
            # Create scheduler if needed
            if self.config.training.lr_scheduler == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.training.max_train_steps
                )
            elif self.config.training.lr_scheduler == "linear":
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=self.config.training.max_train_steps
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
            
            # Process inputs
            model_input = batch["model_input"].to(
                device=self.device,
                dtype=self.model_dtype,
                memory_format=torch.channels_last if self.config.system.channels_last else torch.contiguous_format,
                non_blocking=True
            )
            
            # Process embeddings
            text_embeds = batch["text_embeds"]
            encoder_hidden_states = text_embeds["text_embeds"].to(self.device, non_blocking=True)
            pooled_embeds = text_embeds["pooled_text_embeds"].to(self.device, non_blocking=True)
            tag_weights = batch["tag_weights"].to(self.device, non_blocking=True)
            
            # Sample timesteps
            timesteps = torch.randint(0, self.num_train_timesteps, (model_input.shape[0],), device=self.device)
            
            # Generate noise
            noise = generate_noise(
                model_input.shape,
                self.device,
                self.model_dtype,
                model_input  # Use model_input as layout template
            )
            
            # Get sigmas and add noise
            sigmas = self.sigmas[timesteps].to(dtype=self.model_dtype)
            noisy_latents = model_input + sigmas[:, None, None, None] * noise
            
            # Scale input
            if self.config.training.prediction_type == "v_prediction":
                scaled_input = self.c_in[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * noisy_latents
            else:
                scaled_input = noisy_latents
                
            # Get time embeddings
            time_ids = get_add_time_ids(
                original_sizes=batch["original_sizes"],
                crops_coords_top_lefts=batch["crop_top_lefts"],
                target_sizes=batch["target_sizes"],
                batch_size=model_input.shape[0],
                dtype=self.model_dtype,
                device=self.device
            )
            
            # Forward pass
            model_pred = self.model(
                scaled_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={
                    "text_embeds": pooled_embeds,
                    "time_ids": time_ids
                },
                return_dict=False
            )[0]
            
            # Get target
            if self.config.training.prediction_type == "epsilon":
                target = noise
            else:  # v_prediction
                target = (
                    self.c_skip[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * model_input +
                    self.c_out[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * noise
                )
            
            # Compute loss
            loss = self.compute_loss(model_pred, target, timesteps, tag_weights)
            
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

    def train_epoch(self, dataloader: DataLoader) -> None:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Accumulate gradients
                for i in range(self.config.training.gradient_accumulation_steps):
                    loss = self.training_step(batch)
                    total_loss += loss.item()
                
                # Update weights
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
                        'epoch': self.current_epoch,
                        'step': self.global_step
                    })
                
                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Update epoch counter
        self.current_epoch += 1
        
        # Log epoch metrics
        self.log_metrics({
            'epoch': self.current_epoch,
            'epoch_loss': total_loss / num_batches
        }, step_type="epoch")

    def save_checkpoint(self) -> None:
        """Save model checkpoint."""
        try:
            checkpoint_path = os.path.join(
                self.config.paths.checkpoints_dir,
                f"checkpoint_{self.global_step:06d}.pt"
            )
            
            # Save checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, Any], step_type: str = "step") -> None:
        """Log metrics to wandb and console."""
        if self.config.training.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items()])
        logger.info(f"{step_type.capitalize()} {self.global_step}: {metrics_str}")

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create optimized dataloader."""
        sampler = dataset.get_sampler(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # Force single process
            pin_memory=pin_memory
        )

