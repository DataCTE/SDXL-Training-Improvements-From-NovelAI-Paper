import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import wandb
import os
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from src.data.dataset import NovelAIDataset
from src.data.sampler import AspectBatchSampler
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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

@dataclass
class ModelConfig:
    hidden_size: int
    cross_attention_dim: int
    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float
    num_timesteps: int
    min_snr_gamma: float

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    save_steps: int
    log_steps: int
    mixed_precision: str
    weight_decay: float
    optimizer_eps: float
    optimizer_betas: Tuple[float, float]

@dataclass
class DataConfig:
    image_size: List[int]
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    shuffle: bool

@dataclass
class ScoringConfig:
    aesthetic_score: float
    crop_score: float

@dataclass
class SystemConfig:
    enable_xformers: bool
    channels_last: bool
    gradient_checkpointing: bool
    cudnn_benchmark: bool
    disable_debug_apis: bool
    mixed_precision: str
    gradient_accumulation_steps: int
    use_fsdp: bool
    cpu_offload: bool
    full_shard: bool

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    scoring: ScoringConfig
    system: SystemConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            scoring=ScoringConfig(**config_dict['scoring']),
            system=SystemConfig(**config_dict['system'])
        )

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    SIGMA_MIN = 0.002   # Minimum sigma value
    SIGMA_MAX = 20000.0  # Maximum sigma value (approximating infinity for ZTSNR)
    
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        config: Config,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        precomputed_proj_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        # Initialize training state counters
        self.current_epoch = 0
        self.global_step = 0
        
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        self.config = config

        # Initialize memory optimizations
        if config.system.enable_xformers and is_xformers_installed():
            self.model.enable_xformers_memory_efficient_attention()
            if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                self.vae.enable_xformers_memory_efficient_attention()
        
        if config.system.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        
        # Use channels last memory format for better performance
        if config.system.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.vae = self.vae.to(memory_format=torch.channels_last)

        # Setup mixed precision training
        self.mixed_precision_dtype = (torch.float16 if config.system.mixed_precision == "fp16" 
                                    else torch.bfloat16 if config.system.mixed_precision == "bf16" 
                                    else torch.float32)
        
        # Initialize gradient accumulation
        self.gradient_accumulation_steps = config.system.gradient_accumulation_steps
        
        # Setup distributed training if using multiple GPUs
        if torch.cuda.device_count() > 1 and config.system.use_fsdp:
            self._setup_distributed()

        # Initialize batch sizes
        self.max_batch_size = config.training.batch_size
        self.micro_batch_size = self.max_batch_size // config.training.gradient_accumulation_steps
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        if precomputed_proj_matrix is not None:
            # Validate the projection matrix shape
            # After initializing precomputed_proj_matrix and before copying
            expected_shape = (model.config.cross_attention_dim, 768)
            #print(f"Projection matrix shape: {precomputed_proj_matrix.shape}")
            #print(f"Expected shape: {expected_shape}")
            assert precomputed_proj_matrix.shape == expected_shape, \
                f"Projection matrix must have shape {expected_shape}, got {precomputed_proj_matrix.shape}"

            # Initialize the linear layer
            self.hidden_proj = nn.Linear(
                768,
                model.config.cross_attention_dim,
                bias=False
            ).to(device=device, dtype=torch.bfloat16)

            # Copy the projection matrix
            with torch.no_grad():
                self.hidden_proj.weight.copy_(precomputed_proj_matrix)

            # Debug after copying
            #print(f"Hidden projection weight shape after copy: {self.hidden_proj.weight.shape}")

            
            # Freeze the projection layer to prevent training
            for param in self.hidden_proj.parameters():
                param.requires_grad = False
        else:
            # Initialize the linear layer as usual
            self.hidden_proj = nn.Linear(
                768, 
                model.config.cross_attention_dim
            ).to(device=device, dtype=torch.bfloat16)

        # Initialize model parameters including ZTSNR settings
        self._init_model_params()
        
        # Pre-compute and cache all static values
        self._init_static_buffers()
        
        # Pre-allocate memory buffers
        self._init_memory_buffers()
        
        # Apply system configurations
        self._apply_system_config()

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

    def _init_model_params(self):
        """Initialize model-specific parameters and constants"""
        # ZTSNR parameters
        self.sigma_data = self.config.model.sigma_data
        self.sigma_min = self.config.model.sigma_min
        self.sigma_max = self.config.model.sigma_max
        self.rho = self.config.model.rho
        self.num_timesteps = self.config.model.num_timesteps
        self.min_snr_gamma = self.config.model.min_snr_gamma
        
        # Cache dimensions
        self.hidden_size = self.config.model.hidden_size
        self.cross_attention_dim = self.model.config.cross_attention_dim
        
        # Cache memory formats
        self.channels_last_format = torch.channels_last
        self.contiguous_format = torch.contiguous_format
        
        # Cache model dtype
        self.model_dtype = next(self.model.parameters()).dtype

    def _init_static_buffers(self):
        """Initialize all static buffers and pre-computed values"""
        # Scoring buffers
        self.register_buffer('aesthetic_score', torch.tensor(
            self.config.scoring.aesthetic_score, dtype=torch.bfloat16))
        self.register_buffer('crop_score', torch.tensor(
            self.config.scoring.crop_score, dtype=torch.bfloat16))
        self.register_buffer('zero_score', torch.tensor(0.0, dtype=torch.bfloat16))
        
        # Time embeddings
        self.register_buffer('cached_time_ids', self._precompute_time_ids(self.max_batch_size))
        
        # Noise schedule
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Karras scalings
        c_skip, c_out, c_in = self._compute_karras_scalings(self.sigmas)
        self.register_buffer('c_skip', c_skip.view(-1,1,1,1))
        self.register_buffer('c_out', c_out.view(-1,1,1,1))
        self.register_buffer('c_in', c_in.view(-1,1,1,1))
        
        # Pre-compute SNR weights
        timesteps = torch.arange(self.scheduler.config.num_train_timesteps, device=self.device)
        alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.register_buffer('snr_weights', self._precompute_snr_weights(alphas, timesteps))
        
        # Other pre-computed values
        self.register_buffer('velocity_scale', torch.sqrt(alphas / (1 - alphas)))
        self.register_buffer('noise_scale_factors',
            torch.tensor([1.0 / math.sqrt(x) for x in range(1, self.max_batch_size + 1)],
                        device=self.device, dtype=torch.float32))

    def _init_memory_buffers(self):
        """Initialize all memory buffers for efficient computation"""
        # Pre-allocate intermediate tensors - Fixed dimensions to match model
        self.register_buffer('hidden_states_buffer', torch.zeros(
            (self.micro_batch_size, 77, 768),  # Fixed to standard SDXL dimensions
            device=self.device, dtype=self.model_dtype))
        
        # Pre-allocate batch indices
        self.register_buffer('batch_indices',
            torch.arange(self.max_batch_size, device=self.device)
            .view(-1, self.micro_batch_size))
        
        # Pre-allocate gradient buffers
        self.register_buffer('grad_norm_buffer',
            torch.zeros(len(list(self.model.parameters())),
                    device=self.device, dtype=torch.float32))
        
        # Pre-allocate timestep indices buffer
        self.register_buffer('timestep_indices',
            torch.zeros(self.micro_batch_size, dtype=torch.long, device=self.device))

    def _apply_system_config(self):
        """Apply system-wide configurations"""
        if self.config.system.enable_xformers:
            self.model.enable_xformers_memory_efficient_attention()
        
        if self.config.system.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        if self.config.system.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        
        if self.config.system.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        
        if self.config.system.disable_debug_apis:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)

    def _precompute_time_ids(self, max_batch_size: int) -> torch.Tensor:
        """Pre-compute time_ids for all possible batch sizes."""
        time_ids = torch.empty(
            (max_batch_size, 2, 4), 
            device=self.device, 
            dtype=torch.bfloat16
        )
        
        # Use config.data directly instead of self.data
        time_ids[:, 0, 0] = self.config.data.image_size[0]  # orig_height
        time_ids[:, 0, 1] = self.config.data.image_size[1]  # orig_width
        time_ids[:, 0, 2] = self.aesthetic_score
        time_ids[:, 0, 3] = self.zero_score
        time_ids[:, 1, 0] = self.config.data.image_size[0]  # orig_height
        time_ids[:, 1, 1] = self.config.data.image_size[1]  # orig_width
        time_ids[:, 1, 2] = self.crop_score
        time_ids[:, 1, 3] = self.zero_score
        
        return time_ids.reshape(max_batch_size, -1)

    def _get_add_time_ids(
        self, 
        batch_size: int, 
        height: Optional[int] = None, 
        width: Optional[int] = None
    ) -> torch.Tensor:
        """Dynamically compute time_ids for current batch dimensions.
        
        Args:
            batch_size: Number of samples in batch
            height: Current image height (optional)
            width: Current image width (optional)
        
        Returns:
            torch.Tensor: Time embeddings tensor of shape (batch_size, 8)
        """
        time_ids = torch.empty(
            (batch_size, 2, 4), 
            device=self.device, 
            dtype=torch.bfloat16
        )
        
        # Use current image dimensions if provided, otherwise use config defaults
        curr_height = height if height is not None else self.config.data.image_size[0]
        curr_width = width if width is not None else self.config.data.image_size[1]
        
        time_ids[:, 0, 0] = curr_height  # orig_height
        time_ids[:, 0, 1] = curr_width   # orig_width
        time_ids[:, 0, 2] = self.aesthetic_score
        time_ids[:, 0, 3] = self.zero_score
        time_ids[:, 1, 0] = curr_height  # orig_height
        time_ids[:, 1, 1] = curr_width   # orig_width
        time_ids[:, 1, 2] = self.crop_score
        time_ids[:, 1, 3] = self.zero_score
        
        return time_ids.reshape(batch_size, -1)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            print("No training state found, starting from scratch with pretrained weights")

    def get_karras_scalings(self, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get pre-computed Karras scalings for given timestep indices."""
        return (
            self.c_skip[timestep_indices],
            self.c_out[timestep_indices],
            self.c_in[timestep_indices]
        )

    @torch.no_grad()
    def get_velocity(scheduler: DDPMScheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        # Computes the v_prediction target as described in the paper.
        # v = (x - x_0) / sqrt(sigma^2 + sigma_data^2)
        return scheduler.get_velocity(latents, noise, timesteps)

    def get_sigmas(self) -> torch.Tensor:
            """Generate noise schedule for ZTSNR with optimized scaling.
            
            Uses a modified ramp function to ensure:
            1. First step has σ = σ_min (0.002)
            2. Last step has σ = σ_max (20000) as practical infinity
            3. Intermediate steps follow power-law scaling with ρ=7
            
            Returns:
                torch.Tensor: Noise schedule sigmas of shape [num_timesteps]
            """
            # Generate ramp on device directly
            ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
            
            # Compute inverse rho values
            min_inv_rho = self.sigma_min ** (1/self.rho)
            max_inv_rho = self.sigma_max ** (1/self.rho)
            
            # Generate full schedule with vectorized operations
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            
            # Ensure exact values at endpoints
            sigmas[0] = self.sigma_min
            sigmas[-1] = self.sigma_max
            
            return sigmas

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        # SNR(σ) = (sigma_data / sigma)^2
        return (self.sigma_data / sigma).square()

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get pre-computed SNR weights for given timesteps."""
        return self.snr_weights[timesteps]

    def validate_batch(self, text_embeds: Dict[str, torch.Tensor], start_idx: int, end_idx: int) -> None:
        """Fix text embeddings dimensions for training step."""
        # Step 1: Get and reshape base_hidden to guaranteed 4D
        base_hidden = text_embeds["base_text_embeds"][start_idx:end_idx]
        batch_size = end_idx - start_idx
        
        # Always reshape to 4D [batch, 1, seq_len, hidden]
        base_hidden = base_hidden.reshape(batch_size, 1, 77, 768)
        
        # Step 2: Get and reshape base_pooled to guaranteed 2D
        base_pooled = text_embeds["base_pooled_embeds"][start_idx:end_idx]
        base_pooled = base_pooled.reshape(batch_size, self.config.model.hidden_size)
        
        # Step 3: Update tensors with fixed dimensions
        text_embeds["base_text_embeds"][start_idx:end_idx] = base_hidden
        text_embeds["base_pooled_embeds"][start_idx:end_idx] = base_pooled
        
        return True

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Execute training step with optimized gradient accumulation.
        
        Implements:
        1. v-prediction parameterization
        2. Zero Terminal SNR (ZTSNR) noise schedule
        3. MinSNR loss weighting
        4. Tag-based loss weighting
        """
        
        # Pre-allocate tensors for the entire batch
        total_v_pred = torch.empty(
            (len(images), *images.shape[1:]),
            device=self.device,
            dtype=self.model_dtype,
            memory_format=torch.channels_last
        )

        # Move data to device efficiently using non_blocking transfers
        images = images.to(
            device=self.device,
            non_blocking=True,
            memory_format=torch.channels_last,
            dtype=self.model_dtype
        )
        
        text_embeds = {
            k: v.to(device=self.device, non_blocking=True, dtype=self.model_dtype)
            for k, v in text_embeds.items()
        }
        
        tag_weights = tag_weights.to(
            device=self.device,
            non_blocking=True,
            dtype=self.model_dtype
        )

        # Clear gradients efficiently
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        running_loss = 0.0

        # Pre-generate all noise tensors for the batch
        all_noise = torch.randn(
            size=images.shape,  # Pass as size argument
            dtype=self.model_dtype,
            device=self.device,
        ).to(memory_format=torch.channels_last)  # Convert to channels last after creation

        # Pre-generate all timesteps
        all_timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (len(images),), device=self.device
        )

        # Get all Karras scalings at once
        c_skip_all, c_out_all, c_in_all = self.get_karras_scalings(all_timesteps)
        sigma_expanded_all = self.sigmas[all_timesteps].view(-1, 1, 1, 1)

        # Process in micro-batches for gradient accumulation
        for i in range(self.gradient_accumulation_steps):
            start_idx = i * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
            
            # Extract current micro-batch tensors
            batch_indices = slice(start_idx, end_idx)
            batch_size = end_idx - start_idx
            
            # Get current batch data using views instead of copies
            batch_latents = images[batch_indices]
            batch_noise = all_noise[batch_indices]
            timesteps = all_timesteps[batch_indices]
            
            # Get current batch scalings
            c_skip = c_skip_all[batch_indices]
            c_out = c_out_all[batch_indices]
            c_in = c_in_all[batch_indices]
            sigma_expanded = sigma_expanded_all[batch_indices]

            # Scale latents efficiently
            noise_scale = self.get_noise_scale(batch_size)
            batch_latents = batch_latents * noise_scale

            # After reshaping encoder_hidden_states

            # Get text embeddings and reshape directly to correct format
            encoder_hidden_states = text_embeds["base_text_embeds"][batch_indices]
            #print(f"Before reshape: {encoder_hidden_states.shape}")  # [8, 1, 77, 768]
            
            # Let prepare_hidden_states handle reshaping and projection
            encoder_hidden_states = self.prepare_hidden_states(encoder_hidden_states)
            #print(f"After projection: {encoder_hidden_states.shape}")  # Should be [8, 77, cross_attention_dim]
        
            # Add noise efficiently using fused operations
            noisy_latents = batch_latents + sigma_expanded * batch_noise
            v_target = self.scheduler.get_velocity(batch_latents, batch_noise, timesteps)

            # Forward pass with automatic mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Get pooled embeddings
                pooled_embeds = text_embeds["base_pooled_embeds"][batch_indices]
                pooled_embeds = pooled_embeds.view(batch_size, -1)  # Ensure 2D shape
                
                # Get time embeddings
                time_ids = self._get_add_time_ids(
                    batch_size,
                    height=batch_latents.size(2) * 8,
                    width=batch_latents.size(3) * 8
                )

                # Forward pass with memory-efficient attention
                model_output = self.model(
                    c_in * noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": time_ids
                    }
                ).sample

                # Compute denoised output using fused operations
                D_out = c_skip * noisy_latents + c_out * model_output

                # Compute loss efficiently
                loss_per_sample = F.mse_loss(
                    D_out.float(),
                    v_target.float(),
                    reduction='none'
                ).mean(dim=[1, 2, 3])

                # Apply weights and compute final loss
                snr_weights = self.get_minsnr_weights(timesteps)
                loss_per_sample = loss_per_sample * tag_weights[batch_indices].squeeze() * snr_weights
                loss = loss_per_sample.mean() / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Store metrics
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value
            
            # Store predictions efficiently
            total_v_pred[start_idx:end_idx] = D_out.detach()

            # Log metrics if main process
            if self.accelerator.is_main_process and self.global_step % self.config.training.log_steps == 0:
                wandb.log({
                    'loss/step': loss_value,
                    'loss/running_avg': running_loss / (i + 1),
                    'grad_norm': self.compute_grad_norm(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                })

        avg_loss = running_loss / self.gradient_accumulation_steps
        
        return total_loss, images, total_v_pred, all_timesteps, avg_loss

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
            images, text_embeds, tag_weights = batch
            
            loss, _, _, _, avg_batch_loss = self.training_step(
                images, text_embeds, tag_weights
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
        
        progress_bar.close()
        return total_loss / num_batches

    @staticmethod
    def collate_fn(batch):
        """Optimized collate function with efficient tensor operations."""
        # Separate batch elements
        images, text_embeds, tag_weights = zip(*batch)
        
        # Process images efficiently
        if len(set(img.shape for img in images)) <= 1:
            # Fast path for same-sized images
            images = torch.stack(images, dim=0)
        else:
            # Efficient padding for variable-sized images
            max_h = max(img.shape[-2] for img in images)
            max_w = max(img.shape[-1] for img in images)
            
            # Pre-allocate output tensor
            batch_size = len(images)
            channels = images[0].shape[0]
            padded_images = torch.zeros(
                (batch_size, channels, max_h, max_w),
                dtype=images[0].dtype,
                device=images[0].device
            )
            
            # Fill padded tensor efficiently
            for i, img in enumerate(images):
                h, w = img.shape[-2:]
                padded_images[i, :, :h, :w] = img
            
            images = padded_images
        
        # Process text embeddings efficiently
        batched_text_embeds = {}
        for key in text_embeds[0].keys():
            if len(set(emb[key].shape for emb in text_embeds)) <= 1:
                # Fast path for same-sized embeddings
                batched_text_embeds[key] = torch.stack([emb[key] for emb in text_embeds])
            else:
                # Efficient padding for variable-length sequences
                max_len = max(emb[key].shape[1] for emb in text_embeds)
                hidden_dim = text_embeds[0][key].shape[-1]
                
                # Pre-allocate padded tensor
                padded = torch.zeros(
                    (len(text_embeds), max_len, hidden_dim),
                    dtype=text_embeds[0][key].dtype,
                    device=text_embeds[0][key].device
                )
                
                # Fill padded tensor efficiently
                for i, emb in enumerate(text_embeds):
                    curr_len = emb[key].shape[1]
                    padded[i, :curr_len] = emb[key]
                
                batched_text_embeds[key] = padded
        
        # Stack tag weights efficiently
        tag_weights = torch.stack(tag_weights)
        
        return images, batched_text_embeds, tag_weights

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
            batch_sampler = AspectBatchSampler(dataset, batch_size, shuffle)

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
        """Save checkpoint efficiently"""
        if not dist.is_initialized() or dist.get_rank() == 0:  # Only save on main process
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
                    
            torch.save(checkpoint, path)

    def compute_grad_norm(self):
        """Compute gradient norm using pre-allocated buffer."""
        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                self.grad_norm_buffer[i] = p.grad.data.norm(2)
        return torch.sqrt(torch.sum(self.grad_norm_buffer * self.grad_norm_buffer))

    def get_batch_indices(self, i: int) -> torch.Tensor:
        """Get pre-computed batch indices for current iteration."""
        return self.batch_indices[i]

    def get_noise_scale(self, batch_size: int) -> torch.Tensor:
        """Get pre-computed noise scale factor."""
        return self.noise_scale_factors[batch_size - 1]


    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #print(f"Inside prepare_hidden_states - Input shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        # Remove the extra dimension if present
        if hidden_states.dim() == 4 and hidden_states.size(1) == 1:
            hidden_states = hidden_states.squeeze(1)  # Shape becomes (batch_size, seq_len, hidden_size)
            #print(f"After squeezing - Shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        # Get dimensions
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.reshape(-1, hidden_size)
        #print(f"Before projection - Shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        # Create projection layer if not exists
        if not hasattr(self, 'hidden_proj'):
            self.hidden_proj = nn.Linear(hidden_size, self.cross_attention_dim, bias=True)
            # Initialize weights if needed
            if hasattr(self, 'proj_matrix'):
                self.hidden_proj.weight.data.copy_(self.proj_matrix)
                #print(f"Projection matrix loaded with shape: {self.hidden_proj.weight.shape}")

        # Apply projection
        projected = self.hidden_proj(hidden_states)  # Shape: (batch_size * seq_len, cross_attention_dim)
        #print(f"After projection - Shape: {projected.shape}, dtype: {projected.dtype}")

        # Reshape back to (batch_size, seq_len, cross_attention_dim)
        projected = projected.reshape(batch_size, seq_len, -1)
        #print(f"After final reshape - Shape: {projected.shape}, dtype: {projected.dtype}")

        return projected



    def _precompute_snr_weights(self, alphas: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Pre-compute SNR weights for all possible timesteps.
        
        Implements MinSNR loss weighting as described in the NovelAI Diffusion V3 technical report.
        This treats diffusion as a multi-task learning problem, balancing the learning of each 
        timestep according to difficulty, and avoiding focusing too much training on low-noise timesteps.
        
        Args:
            alphas: Alpha values from noise scheduler
            timesteps: Timestep indices
            
        Returns:
            torch.Tensor: SNR weights clamped by min_snr_gamma
        """
        # Compute SNR for each timestep
        alpha_t = alphas[timesteps]
        snr = alpha_t / (1 - alpha_t)
        
        # Clamp SNR values using min_snr_gamma as described in technical report
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        return torch.minimum(snr, min_snr).float()

    def _generate_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate noise using pre-allocated template."""
        return torch.randn(
            shape,
            device=self.device,
            dtype=torch.bfloat16,
            generator=None,
            layout=self.noise_template.layout
        )

    def get_velocity(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute velocity using pre-computed scales."""
        return (latents - noise) * self.velocity_scale[timesteps].view(-1, 1, 1, 1)


    def _compute_karras_scalings(self, sigmas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial Karras scalings."""
        sigma_data = 0.5  # Default value from Karras et al.
        
        c_skip = sigma_data**2 / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigmas**2 + sigma_data**2).sqrt()
        
        return c_skip, c_out, c_in

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
                min_num_params=1e6,
            )

            # Configure mixed precision policy
            mixed_precision_policy = MixedPrecision(
                param_dtype=self.mixed_precision_dtype,
                reduce_dtype=self.mixed_precision_dtype,
                buffer_dtype=self.mixed_precision_dtype,
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
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    cpu_offload=CPUOffload(offload_params=True) if self.config.system.cpu_offload else None,
                    forward_prefetch=True,
                    limit_all_gathers=True,
                )

            # Enable gradient checkpointing if configured
            if self.config.system.gradient_checkpointing:
                self.model.enable_gradient_checkpointing()
            from torch.amp import GradScaler

            # Setup gradient scaler for mixed precision training
            self.scaler = GradScaler(
                enabled=self.config.system.mixed_precision != "no",
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=100,
            )


            # Initialize process group if not already done
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://'
                )
                
            # Set device to current GPU
            torch.cuda.set_device(torch.cuda.current_device())

            # Optimize CUDA operations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True