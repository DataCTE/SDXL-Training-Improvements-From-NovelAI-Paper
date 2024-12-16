import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import wandb
import os
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from src.data.dataset import NovelAIDataset
from src.data.sampler import AspectBatchSampler
import yaml
from src.config.config import Config, VAEModelConfig
import numpy as np
import random

def is_xformers_installed():
    """Check if xformers is available."""
    try:
        import xformers
        import xformers.ops
        return True
    except ImportError:
        return False

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
        
        # Configure noise scheduler with device and precision set
        self.configure_noise_scheduler()
        
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
        self.model = model or self._create_unet()
        self.vae = vae or self._create_vae()
        
        # Move models to device and set dtype
        self._setup_models()
        
        # Initialize model weights if needed
        if model is None:
            self._initialize_model_weights()
        
        # Set up distributed training
        if self.config.system.use_fsdp:
            self._setup_distributed()
        
        # Configure optimizer with proper parameters
        self._setup_optimizer()
        
        # Set up caching and memory optimizations
        self._setup_memory_optimizations()
        
        # Pre-compute buffers
        self._initialize_buffers()
        
        # Initialize training state
        self._initialize_training_state()


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
        # Move models to device
        self.model = self.model.to(device=self.device, dtype=self.model_dtype)
        self.vae = self.vae.to(device=self.device, dtype=self.model_dtype)
        
        # Set training/eval modes
        self.model.train()
        self.vae.eval()
        self.vae.requires_grad_(False)

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

    def _create_unet(self) -> UNet2DConditionModel:
        """Create and configure UNet model.
        
        Returns:
            UNet2DConditionModel: Configured UNet model
        """
        # Create UNet with SDXL architecture
        unet = UNet2DConditionModel.from_pretrained(
            self.config.model.pretrained_model_name,
            subfolder="unet",
            torch_dtype=self.model_dtype
        )
        
        # Enable memory efficient attention if configured
        if self.config.system.enable_xformers and is_xformers_installed():
            unet.enable_xformers_memory_efficient_attention()
        
        # Enable gradient checkpointing if configured
        if self.config.system.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        return unet

    def _create_vae(self) -> AutoencoderKL:
        """Create and configure VAE model.
        
        Returns:
            AutoencoderKL: Configured VAE model
        """
        # Load pretrained VAE directly
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.model_dtype
        )
        
        # Freeze VAE parameters
        vae.requires_grad_(False)
        vae.eval()
        
        return vae

    def _initialize_model_weights(self):
        """Initialize model weights using improved techniques.
        
        Implements weight initialization as described in the NovelAI technical report:
        1. Scaled initialization for attention layers
        2. Proper initialization for time embedding
        3. Zero initialization for output projection
        """
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Initialize attention layer weights with scaled normal distribution
                if "attn" in module._get_name().lower():
                    scale = 1 / math.sqrt(module.in_features if hasattr(module, "in_features") else module.in_channels)
                    nn.init.normal_(module.weight, mean=0.0, std=scale)
                else:
                    # Standard initialization for other layers
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Embedding):
                # Initialize time embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Apply initialization
        self.model.apply(_init_weights)
        
        # Zero initialize the output projection
        if hasattr(self.model, "conv_out"):
            nn.init.zeros_(self.model.conv_out.weight)
            if self.model.conv_out.bias is not None:
                nn.init.zeros_(self.model.conv_out.bias)

    def _setup_distributed(self):
        """Setup distributed training components"""
        # Import here to avoid circular imports
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            MixedPrecision,
            BackwardPrefetch,
            CPUOffload,
            ShardingStrategy,
        )
        from functools import partial

        if dist.is_initialized():
            # Create a proper size-based auto wrap policy function
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.system.min_num_params_per_shard,
            )

            # Configure mixed precision policy
            mixed_precision_policy = MixedPrecision(
                param_dtype=self.model_dtype,
                reduce_dtype=self.model_dtype,
                buffer_dtype=self.model_dtype,
            )

            # Configure sharding strategy
            sharding_strategy = (ShardingStrategy.FULL_SHARD 
                               if self.config.system.full_shard 
                               else ShardingStrategy.SHARD_GRAD_OP)

            # Apply FSDP wrapping to the model
            if hasattr(self, 'model'):
                self.model = FSDP(
                    self.model,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=torch.cuda.current_device(),
                    mixed_precision=mixed_precision_policy,
                    sharding_strategy=sharding_strategy,
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE if self.config.system.backward_prefetch else None,
                    cpu_offload=CPUOffload(offload_params=True) if self.config.system.cpu_offload else None,
                    forward_prefetch=self.config.system.forward_prefetch,
                    limit_all_gathers=self.config.system.limit_all_gathers,
                )

            # Enable gradient checkpointing if configured
            if self.config.system.gradient_checkpointing:
                self.model.enable_gradient_checkpointing()

            # Initialize process group if not already done
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=self.config.system.backend,
                    init_method='env://'
                )
                
            # Set device to current GPU
            torch.cuda.set_device(torch.cuda.current_device())

            # Optimize CUDA operations
            torch.backends.cudnn.benchmark = self.config.system.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Sync batch norm if configured
            if self.config.system.sync_batch_norm:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def _setup_memory_optimizations(self):
        """Optimize memory usage with NovelAI's techniques.
        
        Implements several memory optimization techniques:
        1. Channels last memory format for tensors
        2. Gradient checkpointing
        3. Efficient attention implementation
        4. Pre-allocated buffers
        """
        # Set memory format if configured
        if self.config.system.channels_last:
            # Check if model is wrapped in DDP or FSDP
            if not isinstance(self.model, (torch.nn.parallel.DistributedDataParallel, 
                                         torch.distributed.fsdp.FullyShardedDataParallel)):
                try:
                    # Convert model to channels_last format
                    self.model = self.model.to(memory_format=torch.channels_last)
                except Exception as e:
                    print(f"Warning: Failed to convert model to channels_last format: {e}")
                    # Continue without channels_last if conversion fails
                    pass
            
        # Enable gradient checkpointing if configured
        if self.config.system.gradient_checkpointing:
            try:
                self.model.enable_gradient_checkpointing()
            except Exception as e:
                print(f"Warning: Failed to enable gradient checkpointing: {e}")
            
        # Enable xformers if available and configured
        if self.config.system.enable_xformers and is_xformers_installed():
            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Warning: Failed to enable xformers: {e}")
            
        # Pre-allocate reusable buffers with appropriate memory format
        memory_format = (torch.channels_last 
                        if self.config.system.channels_last 
                        else torch.contiguous_format)
        
        # Create noise template with correct memory format
        try:
            self.noise_template = torch.empty(
                (self.config.training.batch_size, 4, 64, 64),  # Standard latent size
                device=self.device,
                dtype=self.model_dtype,
                memory_format=memory_format
            )
        except Exception as e:
            print(f"Warning: Failed to create noise template with specified format: {e}")
            # Fallback to default memory format
            self.noise_template = torch.empty(
                (self.config.training.batch_size, 4, 64, 64),
                device=self.device,
                dtype=self.model_dtype
            )
        
        # Pre-allocate gradient norm buffer (1D tensor, memory format doesn't matter)
        self.grad_norm_buffer = torch.zeros(
            len(list(self.model.parameters())), 
            device=self.device
        )
        
        # Enable TF32 for better performance if CUDA is available
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Optimize CUDA operations
                if self.config.system.cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
            except Exception as e:
                print(f"Warning: Failed to configure CUDA optimizations: {e}")

    def _setup_optimizer(self):
        """Configure optimizer with proper parameters."""
        # Get optimizer parameters from config
        optimizer_params = {
            'lr': self.config.training.learning_rate,
            'betas': self.config.training.optimizer_betas,
            'eps': self.config.training.optimizer_eps,
            'weight_decay': self.config.training.weight_decay
        }
        
        # Create AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_params
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = None  # Renamed from self.scheduler to self.lr_scheduler

    def _initialize_buffers(self):
        """Initialize reusable buffers for training.
        
        Pre-allocates:
        1. Micro batch size
        2. Noise buffers
        3. Latent buffers
        4. Timestep buffers
        5. SNR weight buffers
        """
        # Calculate micro batch size for gradient accumulation
        batch_size = self.config.training.batch_size
        self.micro_batch_size = batch_size // self.config.training.gradient_accumulation_steps
        
        if batch_size % self.config.training.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by "
                f"gradient_accumulation_steps ({self.config.training.gradient_accumulation_steps})"
            )
        
        # Get memory format based on config
        memory_format = (torch.channels_last 
                        if self.config.system.channels_last 
                        else torch.contiguous_format)
        
        # Get maximum bucket dimensions
        max_height, max_width = self.config.data.image_size
        vae_scale_factor = 8  # SDXL VAE downscales by 8
        max_latent_height = max_height // vae_scale_factor
        max_latent_width = max_width // vae_scale_factor
        
        # Pre-allocate noise buffer for maximum possible size
        self.noise_buffer = torch.empty(
            (self.micro_batch_size, 4, max_latent_height, max_latent_width),
            device=self.device,
            dtype=self.model_dtype,
            memory_format=memory_format
        )
        
        # Pre-allocate latent buffer
        self.latent_buffer = torch.empty_like(
            self.noise_buffer,
            memory_format=memory_format
        )
        
        # Pre-allocate timestep buffer
        self.timestep_buffer = torch.empty(
            (self.micro_batch_size,),
            device=self.device,
            dtype=torch.long
        )
        
        # Pre-allocate SNR weight buffer if using SNR weighting
        if self.config.training.snr_gamma is not None:
            self.snr_weight_buffer = torch.empty(
                (self.micro_batch_size,),
                device=self.device,
                dtype=self.model_dtype
            )
        
        # Pre-allocate gradient norm buffer
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.grad_norm_buffer = torch.zeros(
            num_params,
            device=self.device,
            dtype=self.model_dtype
        )

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

    def compute_snr_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute SNR weights efficiently.
        
        Args:
            timesteps: Timestep indices
            
        Returns:
            SNR weights tensor
        """
        if not hasattr(self, 'snr_weights') or self.snr_weights is None:
            return None
        
        return self.snr_weights[timesteps]

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted loss with SNR weighting.
        
        Args:
            model_pred: Model prediction
            target: Target tensor (noise or velocity)
            timesteps: Timestep indices
            weights: Optional sample weights
            
        Returns:
            Total weighted loss
        """
        # Basic MSE in float32 for stability
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        
        # Average over non-batch dimensions
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        # Apply SNR weights if enabled
        if self.config.training.snr_gamma is not None:
            snr_weights = self.compute_snr_weight(timesteps)
            loss = loss * snr_weights.to(loss.device)
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()

    def _log_training_step(
        self,
        loss_value: float,
        running_loss: float,
        step: int,
        tag_weights: torch.Tensor
    ):
        """Log training metrics."""
        wandb.log({
            'loss/step': loss_value,
            'loss/running_avg': running_loss / (step + 1),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'global_step': self.global_step,
            'tag_weights/mean': tag_weights.mean().item(),
            'tag_weights/min': tag_weights.min().item(),
            'tag_weights/max': tag_weights.max().item()
        }, step=self.global_step)

    def training_step(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Execute optimized training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (total_loss, model_input, predictions, timesteps, avg_loss)
        """
        with torch.autocast(device_type='cuda', dtype=self.model_dtype):
            # Get and validate inputs
            model_input = batch["model_input"].to(
                device=self.device,
                dtype=self.model_dtype,
                memory_format=torch.channels_last if self.config.system.channels_last else torch.contiguous_format,
                non_blocking=True
            )
            
            # Process inputs efficiently
            tag_weights = batch["tag_weights"].to(self.device, non_blocking=True)
            text_embeds = batch["text_embeds"]
            encoder_hidden_states = text_embeds["text_embeds"].to(self.device, non_blocking=True)
            pooled_embeds = text_embeds["pooled_text_embeds"].to(self.device, non_blocking=True)
            
            # Initialize prediction tensor with correct memory format
            total_v_pred = torch.empty_like(
                model_input,
                memory_format=torch.channels_last if self.config.system.channels_last else torch.contiguous_format
            )
            
            # Clear gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            total_loss = 0.0
            running_loss = 0.0
            
            # Process in micro-batches
            for i in range(self.config.training.gradient_accumulation_steps):
                start_idx = i * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                
                micro_batch = {
                    'latents': model_input[start_idx:end_idx],
                    'encoder_states': encoder_hidden_states[start_idx:end_idx],
                    'pooled': pooled_embeds[start_idx:end_idx],
                    'tag_weights': tag_weights[start_idx:end_idx],
                }
                
                # Generate noise efficiently
                noise = self._generate_noise(micro_batch['latents'].shape)
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, self.num_timesteps, (self.micro_batch_size,),
                    device=self.device
                )
                
                # Get sigmas and add noise
                sigmas = self.scheduler.sigmas[timesteps].to(dtype=self.model_dtype)
                noisy_latents = micro_batch['latents'] + sigmas[:, None, None, None] * noise
                
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
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Update metrics
                loss_value = loss.item()
                total_loss += loss_value
                running_loss += loss_value
                
                # Store predictions
                total_v_pred[start_idx:end_idx] = model_pred.detach()
                
                # Log metrics if main process
                if self.accelerator.is_main_process and self.global_step % self.config.training.log_steps == 0:
                    self._log_training_step(loss_value, running_loss, i, micro_batch['tag_weights'])
            
            # Apply gradient clipping
            if self.config.training.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Update params
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            avg_loss = running_loss / self.config.training.gradient_accumulation_steps
            
            return total_loss, model_input, total_v_pred, timesteps, avg_loss

    def train_epoch(
        self,
        epoch: int,
        train_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Train for one epoch
        
        Args:
            epoch: Current epoch number
            train_dataloader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
            
        Raises:
            ValueError: If no training data is available
        """
        if len(train_dataloader) == 0:
            raise ValueError(
                "No training data available. Please ensure image_dirs contains valid paths with images."
            )

        self.current_epoch = epoch
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for batch in progress_bar:
            loss, _, _, _, avg_batch_loss = self.training_step(
                batch
            )
            
            total_loss += avg_batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_batch_loss:.4f}',
                'avg_loss': f'{total_loss/(progress_bar.n+1):.4f}'
            })
            
            # Log metrics if main process
            if self.accelerator.is_main_process and self.global_step % self.config.training.log_steps == 0:
                wandb.log({
                    'loss/batch': avg_batch_loss,
                    'loss/average': total_loss/(progress_bar.n+1),
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                }, step=self.global_step)
            
            self.global_step += 1
            
            # Save checkpoint if needed
            if self.global_step % self.config.training.save_steps == 0:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoints_dir,
                    f"checkpoint_{self.global_step:06d}.pt"
                )
                self.save_checkpoint(checkpoint_path)
        
        progress_bar.close()
        return total_loss / num_batches

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[Tuple[int, int]]]]:
        """Efficiently collate batch data with optimal padding and dimension handling.
        
        Args:
            batch: List of (image, text_embeds, tag_weight) tuples
            
        Returns:
            Dictionary containing batched and processed data
        """
        # Unpack batch efficiently
        images, text_embeds_dicts, tag_weights = [], [], []
        for img, txt, w in batch:
            images.append(img)
            text_embeds_dicts.append(txt)
            tag_weights.append(w)

        # Pre-compute dimensions once
        vae_scale_factor = 8
        max_res = 1024
        shapes = torch.tensor([[img.shape[1], img.shape[2]] for img in images])
        max_height, max_width = shapes.max(0)[0].tolist()
        
        # Prepare storage with pre-allocated lists
        target_sizes = []
        padded_images = []
        original_sizes = []
        crop_top_lefts = []
        
        # Process each image efficiently
        for img, (h, w) in zip(images, shapes):
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
            target_height = target_height & ~7  # Equivalent to - (target_height % 8)
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

        # Stack tensors efficiently
        stacked_images = torch.stack(padded_images, dim=0)
        tag_weights = torch.stack(tag_weights, dim=0)
        
        # Process embeddings with dimension checks
        def process_embeddings(embeds_list: List[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(embeds_list, dim=0)
            if stacked.dim() == 4:
                stacked = stacked.squeeze(1)
            elif stacked.dim() == 3 and stacked.size(1) == 1:
                stacked = stacked.squeeze(1)
            return stacked
        
        # Get embeddings efficiently
        base_embeds = process_embeddings([d["base_text_embeds"] for d in text_embeds_dicts])
        large_embeds = process_embeddings([d["large_text_embeds"] for d in text_embeds_dicts])
        large_pooled = process_embeddings([d["large_pooled_embeds"] for d in text_embeds_dicts])
        
        # Create embeddings dict without redundant storage
        text_embeds_dict = {
            "text_embeds": large_embeds,  # Main text embeddings
            "pooled_text_embeds": large_pooled,  # Main pooled embeddings
            "text_embeds_large": large_embeds,  # Reference to main
            "pooled_text_embeds_large": large_pooled,  # Reference to main
            "text_embeds_small": base_embeds,
            "pooled_text_embeds_small": None,  # Not used in SDXL
        }
        
        return {
            "model_input": stacked_images,
            "text_embeds": text_embeds_dict,
            "tag_weights": tag_weights,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "target_sizes": target_sizes
        }
    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Create optimized dataloader with efficient data loading."""
        
        # Create distributed sampler if using DDP
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=42,
                drop_last=True
            )
            batch_sampler = None  # Don't use batch_sampler with DistributedSampler
        else:
            sampler = None
            # Use AspectBatchSampler to ensure consistent sizes within batches
            batch_sampler = AspectBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True  # Drop last to ensure consistent batch sizes
            )

        # Configure dataloader with optimized settings
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_sampler is None else 1,  # Only set batch_size if not using batch_sampler
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=NovelAIDiffusionV3Trainer.collate_fn,
            worker_init_fn=NovelAIDiffusionV3Trainer._worker_init_fn,
        )

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Initialize worker with optimized settings."""
        # Set worker seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        # Pin memory for faster data transfer
        torch.cuda.empty_cache()
        
        # Set thread settings for worker
        torch.set_num_threads(1)

    def save_checkpoint(self, path: str):
        """Save checkpoint efficiently.
        
        Args:
            path: Path to save checkpoint to
        """
        if not dist.is_initialized() or dist.get_rank() == 0:  # Only save on main process
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config': self.config,
                'step': self.global_step,
                'epoch': self.current_epoch,
            }
            
            # Save in half precision for smaller file size
            for k, v in checkpoint['model'].items():
                if v.dtype in [torch.float32, torch.float64]:
                    checkpoint['model'][k] = v.half()
                    
            # Save atomically
            tmp_path = path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)  # Atomic operation

    

    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prepare hidden states for SDXL UNet processing.
        
        Args:
            hidden_states: Text embeddings tensor of shape (batch_size, seq_len, hidden_size)
                or (batch_size, 1, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Processed hidden states ready for UNet
        
        Raises:
            ValueError: If input tensor has incorrect number of dimensions
        """
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
            hidden_states = hidden_states.reshape(-1, hidden_size)  # Flatten to (batch_size * seq_len, hidden_size)
            hidden_states = self.hidden_proj(hidden_states)  # Project to cross attention dim
            hidden_states = hidden_states.view(batch_size, seq_len, -1)  # Restore batch dimension
        else:
            # Initialize projection layer if it doesn't exist
            self.hidden_proj = nn.Linear(
                hidden_size,
                self.model.config.cross_attention_dim,
                bias=False
            ).to(device=self.device, dtype=self.model_dtype)
            
            # Initialize weights
            with torch.no_grad():
                # Initialize to identity matrix for the overlapping dimensions
                min_dim = min(hidden_size, self.model.config.cross_attention_dim)
                self.hidden_proj.weight[:min_dim, :min_dim].copy_(torch.eye(min_dim))
                
                # Zero-initialize the rest
                if self.hidden_proj.weight.shape[0] > min_dim:
                    nn.init.zeros_(self.hidden_proj.weight[min_dim:, :])
                if self.hidden_proj.weight.shape[1] > min_dim:
                    nn.init.zeros_(self.hidden_proj.weight[:, min_dim:])
            
            # Project the hidden states
            hidden_states = hidden_states.reshape(-1, hidden_size)  # Flatten to (batch_size * seq_len, hidden_size)
            hidden_states = self.hidden_proj(hidden_states)  # Project to cross attention dim
            hidden_states = hidden_states.view(batch_size, seq_len, -1)  # Restore batch dimension
            
            # Freeze the projection layer
            for param in self.hidden_proj.parameters():
                param.requires_grad = False
        
        return hidden_states
    
    def compute_grad_norm(self) -> float:
        """Compute gradient norm using pre-allocated buffer.
        
        Returns:
            float: The gradient norm
        """
        # Reset buffer
        self.grad_norm_buffer.zero_()
        
        # Compute norms for each parameter's gradient
        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                self.grad_norm_buffer[i] = p.grad.data.norm(2).item()
        
        # Compute total norm
        return torch.sqrt(torch.sum(self.grad_norm_buffer * self.grad_norm_buffer)).item()

    def _precompute_snr_weights(self, alphas: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Pre-compute SNR weights for all possible timesteps.
        
        Args:
            alphas: Alpha values from noise scheduler
            timesteps: Timestep indices
            
        Returns:
            torch.Tensor: SNR weights clamped by min_snr_gamma
        """
        # Get parameters from config
        min_snr_gamma = self.config.model.min_snr_gamma
        sigma_data = self.config.model.sigma_data
        
        # Get alpha values for timesteps
        alpha_t = alphas[timesteps]
        
        # Compute signal and noise variances
        sigma_signal = sigma_data * alpha_t.sqrt()
        sigma_noise = sigma_data * (1 - alpha_t).sqrt()
        
        # Compute SNR = (sigma_signal/sigma_noise)^2
        snr = (sigma_signal / sigma_noise).square()
        
        # Clamp SNR values using min_snr_gamma
        min_snr = torch.tensor(min_snr_gamma, device=self.device)
        return torch.minimum(snr, min_snr).float()

    def _generate_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate noise using pre-allocated template.
        
        Args:
            shape: Shape of noise tensor to generate
            
        Returns:
            torch.Tensor: Generated noise tensor
        """
        return torch.randn(
            shape,
            device=self.device,
            dtype=self.model_dtype,
            generator=None,
            layout=self.noise_template.layout
        )
    
    def get_karras_scalings(self, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get Karras noise schedule scalings for given timesteps.
        
        Args:
            timestep_indices: Timestep indices
            
        Returns:
            Tuple containing:
            - c_skip: Skip connection scaling
            - c_out: Output scaling
            - c_in: Input scaling
        """
        # Get sigmas for timesteps
        sigmas = self.scheduler.sigmas[timestep_indices]
        
        # Compute scaling factors
        c_skip = 1 / (sigmas**2 + 1).sqrt()
        c_out = -sigmas / (sigmas**2 + 1).sqrt()
        c_in = 1 / (sigmas**2 + 1).sqrt()
        
        return c_skip, c_out, c_in

    @torch.no_grad()
    def get_velocity(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute the v_prediction target.
        
        Args:
            latents: Input latent tensor
            noise: Noise tensor
            timesteps: Timestep indices
            
        Returns:
            torch.Tensor: Velocity prediction
        """
        return self.scheduler.get_velocity(latents, noise, timesteps)

    def get_sigmas(self) -> torch.Tensor:
        """Generate noise schedule for ZTSNR with optimized scaling.
        
        Uses a modified ramp function to ensure:
        1. First step has σ = σ_min (0.002)
        2. Last step has σ = σ_max (20000) as practical infinity
        3. Intermediate steps follow power-law scaling with ρ=7
        
        Returns:
            torch.Tensor: Noise schedule sigmas of shape [num_timesteps]
        """
        # Get parameters from config
        num_timesteps = self.config.model.num_timesteps
        sigma_min = self.config.model.sigma_min
        sigma_max = self.config.model.sigma_max
        rho = self.config.model.rho
        
        # Generate ramp on device directly
        ramp = torch.linspace(0, 1, num_timesteps, device=self.device)
        
        # Compute inverse rho values
        min_inv_rho = sigma_min ** (1/rho)
        max_inv_rho = sigma_max ** (1/rho)
        
        # Generate full schedule with vectorized operations
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # Ensure exact values at endpoints
        sigmas[0] = sigma_min  # First step
        sigmas[-1] = sigma_max  # ZTSNR step
        
        return sigmas

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Signal-to-Noise Ratio for given timesteps.
        
        Args:
            timesteps: Timestep indices
            
        Returns:
            torch.Tensor: SNR values
        """
        # Get sigma_data from config
        sigma_data = self.config.model.sigma_data
        
        # Get alphas for timesteps
        alphas = self.scheduler.alphas_cumprod[timesteps]
        
        # Compute signal and noise variances
        sigma_signal = sigma_data * alphas.sqrt()
        sigma_noise = sigma_data * (1 - alphas).sqrt()
        
        # Compute SNR = (sigma_signal/sigma_noise)^2
        snr = (sigma_signal / sigma_noise).square()
        
        return snr

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get pre-computed SNR weights for given timesteps."""
        return self.snr_weights[timesteps]

    def configure_noise_scheduler(self):
        """Configure noise scheduler with Karras schedule and pre-compute training parameters."""
        # Initialize scheduler first with config values
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config.model.num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=self.config.training.prediction_type,
            clip_sample=False,
            thresholding=False
        )
        
        # Generate Karras noise schedule with ZTSNR
        sigmas = self.get_sigmas()  # Implements σ_max ≈ ∞ for ZTSNR
        
        # Pre-compute all noise schedule parameters
        self.alphas = 1 / (sigmas**2 + 1)
        self.betas = 1 - self.alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Update scheduler with our pre-computed values
        self.scheduler.alphas = self.alphas 
        self.scheduler.betas = self.betas
        self.scheduler.alphas_cumprod = self.alphas_cumprod
        self.scheduler.sigmas = sigmas
        self.scheduler.init_noise_sigma = sigmas.max()
        
        # Pre-compute SNR values and weights for MinSNR
        self.snr_values = 1 / (sigmas ** 2)
        if self.config.training.snr_gamma is not None:
            self.snr_weights = torch.minimum(
                self.snr_values,
                torch.tensor(self.config.training.snr_gamma, device=self.device)
            ).float()
        
        # Pre-compute scaling factors based on prediction type
        if self.config.training.prediction_type == "v_prediction":
            self.c_skip = 1 / (sigmas**2 + 1).sqrt()
            self.c_out = -sigmas / (sigmas**2 + 1).sqrt()
            self.c_in = 1 / (sigmas**2 + 1).sqrt()
        else:  # epsilon prediction
            self.c_skip = self.alphas_cumprod.sqrt()
            self.c_out = (1 - self.alphas_cumprod).sqrt()
            self.c_in = torch.ones_like(sigmas)

