import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import wandb
import os
from typing import Dict, Tuple, Optional, List, Union, Any
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from src.data.dataset import NovelAIDataset
from src.data.sampler import AspectBatchSampler
from src.data.thread_config import ThreadConfig
import yaml
from src.config.config import Config, VAEModelConfig
import numpy as np
import random
import logging
import sys
import time
import traceback
from src.utils.model import initialize_model_weights
from src.utils.setup import setup_memory_optimizations, verify_memory_optimizations, verify_buffer_states, verify_scheduler_parameters, check_memory_status
from src.training.scheduler import configure_noise_scheduler, get_karras_scalings
from src.utils.model import create_unet, create_vae, is_xformers_installed
from src.utils.metrics import compute_grad_norm, log_metrics
from src.utils.noise import generate_noise
from src.utils.embeddings import get_add_time_ids
from src.utils.checkpoints import save_checkpoint
import gc

logger = logging.getLogger(__name__)

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        config_path: str,
        model: Optional[UNet2DConditionModel] = None,
        vae: Optional[AutoencoderKL] = None,
        accelerator: Optional[Accelerator] = None
    ):
        """Initialize trainer with additional validations and optimizations."""
        super().__init__()
        
        # Load and validate config first
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        self.config = Config.from_yaml(config_path)
        
        # Set device and precision before scheduler configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_precision()
        
        # Initialize accelerator with proper configuration
        self.accelerator = accelerator or Accelerator(
            mixed_precision=self.config.system.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with="wandb",
            kwargs_handlers=[
                self._get_accelerator_kwargs()
            ]
        )
        
        # Update device after accelerator initialization
        self.device = self.accelerator.device
        
        # Initialize models with proper validation
        self.model = model or create_unet(self.config, self.model_dtype)
        self.vae = vae or create_vae(self.model_dtype)
        
        # Validate model architectures
        self._validate_model_architecture()
        
        # Move models to device and set dtype with verification
        try:
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self._setup_models()
            if torch.cuda.is_available():
                memory_increase = torch.cuda.memory_allocated() - initial_memory
                logger.info(f"Model loading increased memory usage by {memory_increase/(1024**3):.2f}GB")
                
                # Verify model device and dtype
                if next(self.model.parameters()).device != self.device:
                    raise RuntimeError("Model failed to move to correct device")
                if next(self.model.parameters()).dtype != self.model_dtype:
                    raise RuntimeError("Model failed to convert to correct dtype")
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
        
        # Initialize model weights if needed
        if model is None:
            initialize_model_weights(self.model)
        
        # Configure optimizer with proper parameters and verification
        try:
            self._setup_optimizer()
            
            # Verify optimizer state
            if not hasattr(self.optimizer, 'state'):
                raise RuntimeError("Optimizer initialization failed")
            
            # Verify parameter groups
            if len(self.optimizer.param_groups) == 0:
                raise RuntimeError("No parameter groups in optimizer")
                
            # Verify learning rate
            if any(group['lr'] != self.config.training.learning_rate for group in self.optimizer.param_groups):
                raise RuntimeError("Learning rate mismatch in optimizer")
                
        except Exception as e:
            logger.error(f"Error setting up optimizer: {e}")
            raise
        
        # Calculate and verify micro batch size
        batch_size = self.config.training.batch_size
        self.micro_batch_size = batch_size // self.config.training.gradient_accumulation_steps
        
        if batch_size % self.config.training.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by "
                f"gradient_accumulation_steps ({self.config.training.gradient_accumulation_steps})"
            )
            
        # Verify micro batch size is reasonable
        available_memory = (torch.cuda.get_device_properties(0).total_memory 
                          if torch.cuda.is_available() else None)
        if available_memory:
            estimated_batch_memory = self.micro_batch_size * 4 * 128 * 128 * 4  # Rough SDXL latent size
            if estimated_batch_memory > available_memory * 0.2:  # Using more than 20% for batch
                logger.warning("Micro batch size may be too large for available memory")
        
        # Set up memory optimizations and get buffers with verification from setup.py
        try:
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            memory_buffers = setup_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                batch_size=batch_size,
                micro_batch_size=self.micro_batch_size
            )
            
            if torch.cuda.is_available():
                memory_increase = torch.cuda.memory_allocated() - initial_memory
                logger.info(f"Memory optimization setup increased memory by {memory_increase/(1024**3):.2f}GB")
            
            # Verify buffer states using setup.py function
            buffer_states = verify_buffer_states(
                buffers=memory_buffers,
                micro_batch_size=self.micro_batch_size,
                model_dtype=self.model_dtype,
                device=self.device,
                logger=logger
            )
            
            if not all(buffer_states.values()):
                logger.error("Some buffers are invalid")
                raise RuntimeError("Invalid buffer states")
            
            # Unpack memory buffers
            self.noise_template = memory_buffers['noise_template']
            self.grad_norm_buffer = memory_buffers['grad_norm_buffer']
            self.noise_buffer = memory_buffers['noise_buffer']
            self.latent_buffer = memory_buffers['latent_buffer']
            self.timestep_buffer = memory_buffers['timestep_buffer']
            self.snr_weight_buffer = memory_buffers.get('snr_weight_buffer')
                
        except Exception as e:
            logger.error(f"Error setting up memory optimizations: {e}")
            # Try emergency cleanup
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            raise
        
        # Configure noise scheduler and get parameters with verification
        try:
            scheduler_params = configure_noise_scheduler(self.config, self.device)
            
            # Verify scheduler parameters using setup.py function
            param_states = verify_scheduler_parameters(
                scheduler_params=scheduler_params,
                device=self.device,
                logger=logger
            )
            
            if not all(param_states.values()):
                logger.error("Some scheduler parameters are invalid")
                raise RuntimeError("Invalid scheduler parameters")
            
            # Unpack scheduler parameters
            self.scheduler = scheduler_params['scheduler']
            self.alphas = scheduler_params['alphas']
            self.betas = scheduler_params['betas']
            self.alphas_cumprod = scheduler_params['alphas_cumprod']
            self.sigmas = scheduler_params['sigmas']
            self.snr_values = scheduler_params['snr_values']
            self.snr_weights = scheduler_params['snr_weights']
            self.c_skip = scheduler_params['c_skip']
            self.c_out = scheduler_params['c_out']
            self.c_in = scheduler_params['c_in']
                
        except Exception as e:
            logger.error(f"Error configuring scheduler: {e}")
            raise
        
        # Initialize training state
        self._initialize_training_state()
        
        # Final memory check using setup.py function
        if torch.cuda.is_available():
            memory_stats = check_memory_status(
                initial_memory=torch.cuda.memory_allocated() / (1024**3),
                device=self.device,
                logger=logger
            )
            
            if 'leak' in memory_stats:
                logger.warning(f"Memory leak detected during initialization: {memory_stats['leak']:.2f}GB")
            else:
                logger.info(f"Final GPU memory usage: {memory_stats.get('current', 0):.2f}GB")

    def _get_accelerator_kwargs(self):
        """Get additional kwargs for accelerator setup."""
        kwargs = {
            "even_batches": True,  # Ensure consistent batch sizes
            "dispatch_batches": None,  # Let accelerator handle batch dispatch
            "split_batches": False,  # Don't split batches across devices
            "step_scheduler_with_optimizer": True,  # Sync scheduler with optimizer steps
        }
        
        if self.config.system.dynamo_backend:
            kwargs["dynamo_backend"] = self.config.system.dynamo_backend
        
        return kwargs

    def _set_precision(self):
        """Configure model precision based on settings."""
        if self.config.system.mixed_precision == "bf16":
            self.model_dtype = torch.bfloat16
            # Enable auto-casting for better performance
            torch.set_float32_matmul_precision('high')
        elif self.config.system.mixed_precision == "fp16":
            self.model_dtype = torch.float16
        else:
            self.model_dtype = torch.float32

    def _setup_models(self):
        """Setup models with proper device and dtype."""
        try:
            # Add debug logging
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            
            logger.info(f"Target device: {self.device}")
            logger.info(f"Target dtype: {self.model_dtype}")
            
            # Check current device state
            current_model_device = next(self.model.parameters()).device
            current_vae_device = next(self.vae.parameters()).device
            
            logger.info(f"Current model device: {current_model_device}")
            logger.info(f"Current VAE device: {current_vae_device}")
            
            # Only move if needed - use device type comparison
            if current_model_device.type != self.device.type:
                logger.info(f"Moving model from {current_model_device} to {self.device}")
                try:
                    self.model = self.model.to(device=self.device, dtype=self.model_dtype)
                    logger.info("Model move completed")
                except Exception as e:
                    logger.error(f"Error moving model: {str(e)}")
                    logger.error(f"Model state dict keys: {self.model.state_dict().keys()}")
                    raise
            elif next(self.model.parameters()).dtype != self.model_dtype:
                logger.info(f"Converting model dtype to {self.model_dtype}")
                self.model = self.model.to(dtype=self.model_dtype)
                
            if current_vae_device.type != self.device.type:
                logger.info(f"Moving VAE from {current_vae_device} to {self.device}")
                try:
                    self.vae = self.vae.to(device=self.device, dtype=self.model_dtype)
                    logger.info("VAE move completed")
                except Exception as e:
                    logger.error(f"Error moving VAE: {str(e)}")
                    raise
            elif next(self.vae.parameters()).dtype != self.model_dtype:
                logger.info(f"Converting VAE dtype to {self.model_dtype}")
                self.vae = self.vae.to(dtype=self.model_dtype)
            
            # Set training/eval modes
            self.model.train()
            self.vae.eval()
            self.vae.requires_grad_(False)
            
            # Verify final state
            final_model_device = next(self.model.parameters()).device
            final_model_dtype = next(self.model.parameters()).dtype
            logger.info(f"Final model device: {final_model_device}")
            logger.info(f"Final model dtype: {final_model_dtype}")
            
            # Compare device types instead of exact string match
            if final_model_device.type != self.device.type:
                logger.error(f"Model device type mismatch: expected {self.device.type}, got {final_model_device.type}")
                raise RuntimeError("Model failed to move to correct device type")
            if final_model_dtype != self.model_dtype:
                logger.error(f"Model dtype mismatch: expected {self.model_dtype}, got {final_model_dtype}")
                raise RuntimeError("Model failed to convert to correct dtype")
                
        except Exception as e:
            logger.error(f"Error in _setup_models: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _validate_model_architecture(self):
        """Validate model is a proper SDXL UNet."""
        # Basic validation that this is an SDXL UNet
        if not isinstance(self.model, UNet2DConditionModel):
            raise ValueError("Model must be a UNet2DConditionModel")
        
        # Validate VAE
        if not isinstance(self.vae, AutoencoderKL):
            raise ValueError("VAE must be an AutoencoderKL")
        
        if self.vae.config.latent_channels != 4:
            raise ValueError("VAE must have 4 latent channels")

    def _setup_optimizer(self):
        """Configure optimizer with proper parameters."""
        # Get optimizer parameters from config
        optimizer_params = {
            'lr': self.config.training.learning_rate,
            'betas': self.config.training.optimizer_betas,
            'eps': self.config.training.optimizer_eps,
            'weight_decay': self.config.training.weight_decay
        }
        from adamw_bf16 import AdamWBF16
        # Create AdamW optimizer
        self.optimizer = AdamWBF16(
            self.model.parameters(),
            **optimizer_params
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = None  # Renamed from self.scheduler to self.lr_scheduler

    def _initialize_training_state(self):
        """Initialize training state variables.
        
        Sets up:
        1. Step counters
        2. Loss tracking
        3. Gradient scaler for mixed precision
        4. Progress tracking
        """
        # Initialize step counters
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize loss tracking
        self.running_loss = 0.0
        self.num_steps = 0
        self.best_loss = float('inf')
        
        # Initialize gradient scaler for mixed precision training
        if self.config.system.mixed_precision in ["fp16", "bf16"]:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize progress tracking
        self.train_time = 0.0
        self.samples_seen = 0
        self.steps_since_save = 0
        
        # Initialize validation tracking
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Initialize early stopping state
        if hasattr(self.config.training, 'early_stopping_patience'):
            self.early_stopping_counter = 0
            self.best_model_step = 0
        
        # Initialize gradient accumulation state
        self.current_accumulation_step = 0


    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted loss with error handling."""
        try:
            # Validate inputs
            if not isinstance(model_pred, torch.Tensor):
                raise ValueError("model_pred must be a tensor")
            if not isinstance(target, torch.Tensor):
                raise ValueError("target must be a tensor")
            if model_pred.shape != target.shape:
                raise ValueError(f"Shape mismatch: model_pred {model_pred.shape} vs target {target.shape}")

            # Basic MSE in float32 for stability
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            
            # Average over non-batch dimensions
            loss = loss.mean(dim=list(range(1, len(loss.shape))))
            
            # Apply SNR weights if enabled
            if self.config.training.snr_gamma is not None:
                snr_weights = self.compute_snr_weight(timesteps)
                if snr_weights is not None:
                    loss = loss * snr_weights.to(loss.device)
            
            # Apply sample weights if provided
            if weights is not None:
                if weights.shape != loss.shape:
                    raise ValueError(f"Weight shape {weights.shape} doesn't match loss shape {loss.shape}")
                loss = loss * weights
            
            return loss.mean()

        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise



    def training_step(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Execute optimized training step with comprehensive error handling."""
        try:
            # Track memory usage before step
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated() / 1024**2
            
            step_start = time.time()
            
            # Re-verify memory optimizations are still active using setup.py functions
            optimization_states = verify_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=logger
            )
            
            # Take action if optimizations were lost
            if not all(optimization_states.values()):
                logger.warning("Some memory optimizations were lost - attempting to restore")
                try:
                    memory_buffers = setup_memory_optimizations(
                        model=self.model,
                        config=self.config,
                        device=self.device,
                        batch_size=self.config.training.batch_size,
                        micro_batch_size=self.micro_batch_size
                    )
                    # Re-verify after restore attempt
                    optimization_states = verify_memory_optimizations(
                        model=self.model,
                        config=self.config,
                        device=self.device,
                        logger=logger
                    )
                    if not all(optimization_states.values()):
                        logger.error("Failed to restore memory optimizations")
                except Exception as e:
                    logger.error(f"Error restoring optimizations: {e}")
            
            with torch.autocast(device_type='cuda', dtype=self.model_dtype):
                try:
                    # Get and validate inputs
                    model_input = batch["model_input"].to(
                        device=self.device,
                        dtype=self.model_dtype,
                        memory_format=torch.channels_last if self.config.system.channels_last else torch.contiguous_format,
                        non_blocking=True
                    )
                except KeyError as e:
                    logger.error(f"Missing key in batch: {e}")
                    raise
                except RuntimeError as e:
                    logger.error(f"Error moving model input to device: {e}")
                    raise
                
                try:
                    # Process inputs efficiently
                    tag_weights = batch["tag_weights"].to(self.device, non_blocking=True)
                    text_embeds = batch["text_embeds"]
                    encoder_hidden_states = text_embeds["text_embeds"].to(self.device, non_blocking=True)
                    pooled_embeds = text_embeds["pooled_text_embeds"].to(self.device, non_blocking=True)
                except KeyError as e:
                    logger.error(f"Missing embedding key in batch: {e}")
                    raise
                except RuntimeError as e:
                    logger.error(f"Error processing embeddings: {e}")
                    raise

                # Initialize prediction tensor with correct memory format
                try:
                    total_v_pred = torch.empty_like(
                        model_input,
                        memory_format=torch.channels_last if self.config.system.channels_last else torch.contiguous_format
                    )
                except RuntimeError as e:
                    logger.error(f"Error creating prediction tensor: {e}")
                    raise
                
                # Clear gradients efficiently
                self.optimizer.zero_grad(set_to_none=True)
                
                total_loss = 0.0
                running_loss = 0.0
                
                # Process in micro-batches
                for i in range(self.config.training.gradient_accumulation_steps):
                    micro_batch_start = time.time()
                    start_idx = i * self.micro_batch_size
                    end_idx = start_idx + self.micro_batch_size
                    
                    try:
                        # Verify buffer states using setup.py function
                        buffer_states = verify_buffer_states(
                            buffers={
                                'noise_buffer': self.noise_buffer,
                                'latent_buffer': self.latent_buffer,
                                'timestep_buffer': self.timestep_buffer,
                                'noise_template': self.noise_template,
                                'grad_norm_buffer': self.grad_norm_buffer,
                                **(({'snr_weight_buffer': self.snr_weight_buffer} 
                                   if self.snr_weight_buffer is not None else {}))
                            },
                            micro_batch_size=self.micro_batch_size,
                            model_dtype=self.model_dtype,
                            device=self.device,
                            logger=logger
                        )
                        
                        # Take action if buffers are invalid
                        if not all(buffer_states.values()):
                            logger.warning("Some buffers are invalid - attempting to restore")
                            try:
                                memory_buffers = setup_memory_optimizations(
                                    model=self.model,
                                    config=self.config,
                                    device=self.device,
                                    batch_size=self.config.training.batch_size,
                                    micro_batch_size=self.micro_batch_size
                                )
                                # Update buffer references
                                self.noise_buffer = memory_buffers['noise_buffer']
                                self.latent_buffer = memory_buffers['latent_buffer']
                                self.timestep_buffer = memory_buffers['timestep_buffer']
                                self.noise_template = memory_buffers['noise_template']
                                self.grad_norm_buffer = memory_buffers['grad_norm_buffer']
                                if 'snr_weight_buffer' in memory_buffers:
                                    self.snr_weight_buffer = memory_buffers['snr_weight_buffer']
                            except Exception as e:
                                logger.error(f"Error restoring buffers: {e}")
                                raise
                            
                        micro_batch = {
                            'latents': model_input[start_idx:end_idx],
                            'encoder_states': encoder_hidden_states[start_idx:end_idx],
                            'pooled': pooled_embeds[start_idx:end_idx],
                            'tag_weights': tag_weights[start_idx:end_idx],
                        }
                        
                        # Generate noise efficiently using pre-allocated buffer
                        self.noise_buffer.normal_()
                        noise = self.noise_buffer[:micro_batch['latents'].shape[0]]
                        
                        # Sample timesteps
                        timesteps = torch.randint(
                            0, self.num_timesteps, (self.micro_batch_size,),
                            device=self.device
                        )
                        
                        # Get sigmas and add noise using pre-allocated buffer
                        sigmas = self.scheduler.sigmas[timesteps].to(dtype=self.model_dtype)
                        self.latent_buffer.copy_(micro_batch['latents'])
                        self.latent_buffer.addcmul_(sigmas[:, None, None, None], noise)
                        noisy_latents = self.latent_buffer[:micro_batch['latents'].shape[0]]
                        
                        # Scale input
                        if self.config.training.prediction_type == "v_prediction":
                            scaled_input = self.c_in[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * noisy_latents
                        else:
                            scaled_input = noisy_latents
                            
                        # Get time embeddings
                        time_ids = self._get_add_time_ids(
                            original_sizes=batch["original_sizes"][start_idx:end_idx],
                            crops_coords_top_lefts=batch["crop_top_lefts"][start_idx:end_idx],
                            target_sizes=batch["target_sizes"][start_idx:end_idx],
                            batch_size=self.micro_batch_size
                        )
                        
                        # Run model forward pass
                        model_pred = self.model(
                            scaled_input,
                            timesteps,
                            encoder_hidden_states=micro_batch['encoder_states'],
                            added_cond_kwargs={
                                "text_embeds": micro_batch['pooled'],
                                "time_ids": time_ids
                            },
                            return_dict=False
                        )[0]
                        
                        # Get target
                        target = noise if self.config.training.prediction_type == "epsilon" else (
                            self.c_skip[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * micro_batch['latents'] +
                            self.c_out[timesteps].to(dtype=self.model_dtype)[:, None, None, None] * noise
                        )
                        
                        # Compute loss
                        loss = self.compute_loss(
                            model_pred, 
                            target,
                            timesteps,
                            micro_batch['tag_weights']
                        ) / self.config.training.gradient_accumulation_steps
                        
                        # Backward pass with memory tracking
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.reset_peak_memory_stats()
                            self.accelerator.backward(loss)
                            if torch.cuda.is_available():
                                backward_peak = torch.cuda.max_memory_allocated() / 1024**2
                                if backward_peak > start_mem * 2:  # More than 2x increase
                                    logger.warning(f"High memory usage in backward pass: {backward_peak:.0f}MB")
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logger.error(f"OOM in backward pass: {e}")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            raise
                        
                        # Update metrics
                        loss_value = loss.item()
                        total_loss += loss_value
                        running_loss += loss_value
                        
                        # Store predictions
                        total_v_pred[start_idx:end_idx] = model_pred.detach()
                        
                        # Log micro-batch metrics
                        if self.accelerator.is_main_process:
                            micro_batch_time = time.time() - micro_batch_start
                            logger.debug(
                                f"Micro-batch {i+1}/{self.config.training.gradient_accumulation_steps} - "
                                f"Loss: {loss_value:.4f}, Time: {micro_batch_time:.2f}s"
                            )
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(f"OOM in micro-batch {i+1}: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise
                        logger.error(f"Error in micro-batch {i+1}: {e}")
                        raise
                
                    # Log metrics if main process
                    if self.accelerator.is_main_process and self.global_step % self.config.training.log_steps == 0:
                        self._log_training_step(loss_value, running_loss, i, micro_batch['tag_weights'])
                
                # Apply gradient clipping
                if self.config.training.max_grad_norm is not None:
                    try:
                        # Track gradient norms
                        grad_norm = 0.0
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad_norm = max(grad_norm, param.grad.norm().item())
                                
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                        
                        # Verify clipping worked
                        clipped_grad_norm = 0.0
                        for param in self.model.parameters():
                            if param.grad is not None:
                                clipped_grad_norm = max(clipped_grad_norm, param.grad.norm().item())
                                
                        if clipped_grad_norm > self.config.training.max_grad_norm * 1.1:  # 10% tolerance
                            logger.warning(f"Gradient clipping may not be working: {clipped_grad_norm:.2f} > {self.config.training.max_grad_norm}")
                            
                    except RuntimeError as e:
                        logger.error(f"Error during gradient clipping: {e}")
                        raise
                
                # Update params
                try:
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                except RuntimeError as e:
                    logger.error(f"Error during optimizer step: {e}")
                    raise

                avg_loss = running_loss / self.config.training.gradient_accumulation_steps
                
                # Log step completion metrics and check memory using setup.py function
                if self.accelerator.is_main_process:
                    step_time = time.time() - step_start
                    if torch.cuda.is_available():
                        memory_stats = check_memory_status(
                            initial_memory=start_mem / 1024,  # Convert MB to GB
                            device=self.device,
                            logger=logger
                        )
                        
                        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                        mem_diff = peak_mem - start_mem
                        logger.info(
                            f"Step completed - "
                            f"Avg Loss: {avg_loss:.4f}, "
                            f"Time: {step_time:.2f}s, "
                            f"Peak Memory: {peak_mem:.0f}MB "
                            f"(+{mem_diff:.0f}MB)"
                        )
                        
                        # Log any memory leaks detected
                        if 'leak' in memory_stats:
                            logger.warning(f"Memory leak detected: {memory_stats['leak']:.2f}GB")
                
                return total_loss, model_input, total_v_pred, timesteps, avg_loss
            
        except Exception as e:
            logger.error(f"Unexpected error in training step: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Emergency cleanup
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            raise

    def train_epoch(
        self,
        epoch: int,
        train_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Train for one epoch with improved error handling and monitoring."""
        if len(train_dataloader) == 0:
            raise ValueError("No training data available")

        self.current_epoch = epoch
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_dataloader)
        last_logging_time = time.time()
        
        # Initialize progress bar with more informative metrics
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True  # Adapt to terminal width
        )

        try:
            for batch_idx, batch in enumerate(progress_bar):
                # Add batch processing timing
                batch_start = time.time()
                
                # Validate batch data
                if not isinstance(batch, dict) or "model_input" not in batch:
                    logger.error(f"Invalid batch format at index {batch_idx}")
                    continue
                    
                try:
                    # Process batch with timing
                    with torch.cuda.amp.autocast(enabled=self.config.system.mixed_precision != "no"):
                        loss, _, _, _, avg_batch_loss = self.training_step(batch)
                    
                    total_loss += avg_batch_loss
                    
                    # Update progress more frequently
                    if batch_idx % 10 == 0:  # Update every 10 batches
                        progress_bar.set_postfix({
                            'loss': f'{avg_batch_loss:.4f}',
                            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                            'batch_time': f'{(time.time() - batch_start):.2f}s'
                        })
                    
                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_logging_time > self.config.training.log_interval:
                        if self.accelerator.is_main_process:
                            # Log detailed metrics
                            self._log_detailed_metrics(
                                batch_idx, 
                                num_batches,
                                avg_batch_loss, 
                                total_loss/(batch_idx+1),
                                current_time - batch_start
                            )
                        last_logging_time = current_time
                    
                    # Memory management
                    if batch_idx % 100 == 0:  # Every 100 batches
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"OOM error in batch {batch_idx}. Attempting recovery...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                self.global_step += 1
                
                # Save checkpoint if needed
                if self.global_step % self.config.training.save_steps == 0:
                    self._save_checkpoint(batch_idx, avg_batch_loss)
                    
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise e
        finally:
            progress_bar.close()
            
        return total_loss / num_batches



    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[Tuple[int, int]]]]:
        """Efficiently collate batch data with optimal padding and dimension handling."""
        try:
            # Validate input batch
            if not batch:
                raise ValueError("Empty batch received")
            if not all(len(item) == 3 for item in batch):
                raise ValueError("Each batch item must contain (image, text_embeds, tag_weight)")

            # Unpack batch efficiently
            try:
                images, text_embeds_dicts, tag_weights = [], [], []
                for i, (img, txt, w) in enumerate(batch):
                    if not isinstance(img, torch.Tensor):
                        raise TypeError(f"Image at index {i} is not a tensor")
                    if not isinstance(txt, dict):
                        raise TypeError(f"Text embeddings at index {i} is not a dictionary")
                    if not isinstance(w, torch.Tensor):
                        raise TypeError(f"Tag weight at index {i} is not a tensor")
                    
                    images.append(img)
                    text_embeds_dicts.append(txt)
                    tag_weights.append(w)
            except Exception as e:
                logger.error(f"Error unpacking batch: {str(e)}")
                raise

            # Pre-compute dimensions once
            try:
                vae_scale_factor = 8
                max_res = 1024
                shapes = torch.tensor([[img.shape[1], img.shape[2]] for img in images])
                max_height, max_width = shapes.max(0)[0].tolist()
            except Exception as e:
                logger.error(f"Error computing image dimensions: {str(e)}")
                raise

            # Prepare storage with pre-allocated lists
            target_sizes = []
            padded_images = []
            original_sizes = []
            crop_top_lefts = []

            # Process each image efficiently
            try:
                for i, (img, (h, w)) in enumerate(zip(images, shapes)):
                    try:
                        # Convert to pixel space once
                        orig_height = h * vae_scale_factor
                        orig_width = w * vae_scale_factor
                        original_sizes.append((orig_height, orig_width))

                        # Calculate target size efficiently
                        aspect_ratio = w / h
                        if aspect_ratio >= 1:
                            target_height = min(max_res, orig_height)
                            target_width = min(max_res, int(target_height * aspect_ratio))
                        else:
                            target_width = min(max_res, orig_width)
                            target_height = min(max_res, int(target_width / aspect_ratio))

                        # Ensure dimensions are multiples of 8 using bit operations
                        target_height = target_height & ~7
                        target_width = target_width & ~7
                        target_sizes.append((target_height, target_width))

                        # Calculate crops efficiently
                        vae_target_h = target_height // vae_scale_factor
                        vae_target_w = target_width // vae_scale_factor
                        crop_top = ((h - vae_target_h) // 2) * vae_scale_factor
                        crop_left = ((w - vae_target_w) // 2) * vae_scale_factor
                        crop_top_lefts.append((max(0, crop_top), max(0, crop_left)))

                        # Optimize padding
                        if h != max_height or w != max_width:
                            pad_h = max_height - h
                            pad_w = max_width - w
                            padding = [0, pad_w, 0, pad_h]  # Left, Right, Top, Bottom
                            padded = F.pad(img, padding, mode='constant', value=0)
                            padded_images.append(padded)
                        else:
                            padded_images.append(img)
                    except Exception as e:
                        logger.error(f"Error processing image {i}: {str(e)}")
                        raise
            except Exception as e:
                logger.error("Error in image processing loop")
                raise

            # Stack tensors efficiently
            try:
                stacked_images = torch.stack(padded_images, dim=0)
                tag_weights = torch.stack(tag_weights, dim=0)
            except Exception as e:
                logger.error(f"Error stacking tensors: {str(e)}")
                raise

            # Process embeddings
            try:
                def process_embeddings(embeds_list: List[torch.Tensor], name: str) -> torch.Tensor:
                    try:
                        stacked = torch.stack(embeds_list, dim=0)
                        if stacked.dim() == 4:
                            stacked = stacked.squeeze(1)
                        elif stacked.dim() == 3 and stacked.size(1) == 1:
                            stacked = stacked.squeeze(1)
                        return stacked
                    except Exception as e:
                        logger.error(f"Error processing {name} embeddings: {str(e)}")
                        raise

                # Get embeddings efficiently
                base_embeds = process_embeddings([d["base_text_embeds"] for d in text_embeds_dicts], "base")
                large_embeds = process_embeddings([d["large_text_embeds"] for d in text_embeds_dicts], "large")
                large_pooled = process_embeddings([d["large_pooled_embeds"] for d in text_embeds_dicts], "pooled")

            except KeyError as e:
                logger.error(f"Missing key in text embeddings dictionary: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error processing embeddings: {str(e)}")
                raise

            # Create embeddings dict without redundant storage
            text_embeds_dict = {
                "text_embeds": large_embeds,
                "pooled_text_embeds": large_pooled,
                "text_embeds_large": large_embeds,
                "pooled_text_embeds_large": large_pooled,
                "text_embeds_small": base_embeds,
                "pooled_text_embeds_small": None,
            }

            return {
                "model_input": stacked_images,
                "text_embeds": text_embeds_dict,
                "tag_weights": tag_weights,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes": target_sizes
            }

        except Exception as e:
            logger.error(f"Fatal error in collate_fn: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create optimized dataloader."""
        # Create sampler for distributed training
        sampler = dataset.get_sampler(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # Force single process
            pin_memory=pin_memory,
            collate_fn=NovelAIDiffusionV3Trainer.collate_fn
        )

    
    def _save_checkpoint(self, batch_idx, loss):
        """Internal method to save checkpoint during training."""
        try:
            checkpoint_path = os.path.join(
                self.config.paths.checkpoints_dir,
                f"checkpoint_{self.global_step:06d}.pt"
            )
            save_checkpoint(
                path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                config=self.config,
                global_step=self.global_step,
                current_epoch=self.current_epoch
            )
            logger.info(f"Saved checkpoint at step {self.global_step} (batch {batch_idx}, loss: {loss:.4f})")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def save_checkpoint(self, save_dir: str, epoch: int):
        """Public method to save checkpoint at epoch end."""
        try:
            checkpoint_path = os.path.join(
                save_dir,
                f"checkpoint_epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                config=self.config,
                global_step=self.global_step,
                current_epoch=epoch
            )
            logger.info(f"Saved checkpoint for epoch {epoch}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prepare hidden states with error handling."""
        try:
            # Validate input dimensions
            if hidden_states.dim() not in [3, 4]:
                raise ValueError(f"Expected hidden states to have 3 or 4 dimensions, got {hidden_states.dim()}")
            
            # Get shape information
            if hidden_states.dim() == 4:
                batch_size, _, seq_len, hidden_size = hidden_states.shape
                hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)
            else:
                batch_size, seq_len, hidden_size = hidden_states.shape
            
            # Project to cross attention dim if needed
            if hasattr(self, 'hidden_proj'):
                # Validate projection matrix shape
                expected_shape = (self.model.config.cross_attention_dim, hidden_size)
                actual_shape = self.hidden_proj.weight.shape
                if actual_shape != expected_shape:
                    raise ValueError(f"Projection matrix must have shape {expected_shape}, got {actual_shape}")
                
                # Project efficiently
                hidden_states = hidden_states.reshape(-1, hidden_size)  # Flatten
                hidden_states = self.hidden_proj(hidden_states)  # Project
                hidden_states = hidden_states.view(batch_size, seq_len, -1)  # Restore
            else:
                # Initialize projection layer if needed
                self._initialize_hidden_proj(hidden_size)
                hidden_states = self._project_hidden_states(hidden_states, batch_size, seq_len)
            
            return hidden_states

        except Exception as e:
            logger.error(f"Error in prepare_hidden_states: {str(e)}")
            logger.error(f"Input tensor shape: {hidden_states.shape}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def compute_grad_norm(self) -> float:
        return compute_grad_norm(self.model, self.grad_norm_buffer)

    def _generate_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return generate_noise(shape, self.device, self.model_dtype, self.noise_template)
    
    def _get_add_time_ids(self, original_sizes, crops_coords_top_lefts, target_sizes, batch_size) -> torch.Tensor:
        return get_add_time_ids(
            original_sizes, crops_coords_top_lefts, target_sizes, batch_size,
            self.model_dtype, self.device
        )

    def log_metrics(self, metrics: Dict[str, Any], step_type: str = "step"):
        log_metrics(
            metrics, self.global_step, self.accelerator.is_main_process,
            self.config.training.use_wandb, step_type
        )

    def get_karras_scalings(self, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get Karras noise schedule scalings for given timesteps."""
        return get_karras_scalings(self.sigmas, timestep_indices)

