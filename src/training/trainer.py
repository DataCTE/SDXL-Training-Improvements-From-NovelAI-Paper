import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Initialize batch sizes
        self.max_batch_size = config.training.batch_size
        self.micro_batch_size = self.max_batch_size // config.training.gradient_accumulation_steps
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize projection layer with correct dimensions and dtype
        self.hidden_proj = nn.Linear(
            768,  # CLIP hidden size
            model.config.cross_attention_dim  # UNet cross attention dim
        ).to(device=device, dtype=torch.bfloat16)

        # Initialize model parameters
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
        # SNR(σ) = (��_data / σ)²
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
        """Execute training step with gradient accumulation."""
        
        # Move data to device with optimal memory format
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

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        running_loss = 0.0
        total_v_pred = None

        for i in range(self.gradient_accumulation_steps):
            batch_indices = self.get_batch_indices(i)
            batch_size = len(batch_indices)
            
            # Get current batch latents
            batch_latents = images[batch_indices].clone(memory_format=torch.channels_last)
            
            # Create workspace tensors dynamically for this batch
            noisy_latents = torch.zeros_like(batch_latents, device=self.device, dtype=torch.bfloat16)
            
            noise_scale = self.get_noise_scale(batch_size)
            batch_latents.mul_(noise_scale)

            # Prepare text embeddings with correct shapes
            encoder_hidden_states = text_embeds["base_text_embeds"][batch_indices]
            # Reshape to (batch_size, seq_len, hidden_dim)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, 768)
            # Project to correct dimension
            encoder_hidden_states = self.prepare_hidden_states(encoder_hidden_states)

            # Generate noise
            noise = torch.randn_like(
                noisy_latents,
                device=self.device,
                dtype=batch_latents.dtype,
                memory_format=self.channels_last_format
            )

            # Use pre-allocated timesteps
            timesteps = self.timestep_indices[:batch_size]
            torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
                generator=None,
                out=timesteps
            )

            # Get pre-computed scalings
            c_skip = self.c_skip[timesteps]
            c_out = self.c_out[timesteps]
            c_in = self.c_in[timesteps]
            sigma_expanded = self.sigmas[timesteps].view(-1, 1, 1, 1)

            # Add noise using dynamic workspace
            noisy_latents.copy_(batch_latents)
            noisy_latents.add_(sigma_expanded * noise)
            v_target = self.scheduler.get_velocity(batch_latents, noise, timesteps)

            # Forward pass using dynamic workspace
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Prepare pooled embeddings with correct shape
                pooled_embeds = text_embeds["base_pooled_embeds"][batch_indices]
                pooled_embeds = pooled_embeds.view(batch_size, -1)  # Ensure 2D shape
                
                # Get time embeddings
                time_ids = self._get_add_time_ids(
                    batch_size,
                    height=batch_latents.size(2) * 8,
                    width=batch_latents.size(3) * 8
                )

                model_output = self.model(
                    c_in.view(-1, 1, 1, 1) * noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": time_ids
                    }
                ).sample

                # Compute denoised output
                D_out = c_skip.view(-1, 1, 1, 1) * noisy_latents + c_out.view(-1, 1, 1, 1) * model_output

                # Compute loss using reduction mask
                loss_per_sample = F.mse_loss(
                    D_out.float(),
                    v_target.float(),
                    reduction='none'
                ).mean(dim=[1, 2, 3])

                snr_weights = self.get_minsnr_weights(timesteps)
                loss_per_sample = loss_per_sample * tag_weights[batch_indices].squeeze() * snr_weights
                loss = loss_per_sample.mean() / self.gradient_accumulation_steps

            loss.backward()

            # Calculate gradient norm
            grad_norm = self.compute_grad_norm()

            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value
            
            # Store predictions
            total_v_pred = D_out.detach() if total_v_pred is None else torch.cat([total_v_pred, D_out.detach()], dim=0)

            # Log metrics if main process
            if self.accelerator.is_main_process:
                wandb.log({
                    'loss/step': loss_value,
                    'loss/running_avg': running_loss / (i + 1),
                    'grad_norm': grad_norm,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                })

        avg_loss = running_loss / self.gradient_accumulation_steps
        
        # Log final metrics for the full step
        if self.accelerator.is_main_process:
            wandb.log({
                'loss/batch_avg': avg_loss,
                'epoch': self.current_epoch,
            })
        
        return total_loss, batch_latents, total_v_pred, timesteps, avg_loss

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
        """Custom collate function to handle variable sized data.
        
        Args:
            batch: List of tuples (image, text_embeds, tag_weights)
        
        Returns:
            Tuple of batched tensors
        """
        # Separate batch elements
        images, text_embeds, tag_weights = zip(*batch)
        
        # Stack images if they're all the same size, otherwise pad
        if len(set(img.shape for img in images)) <= 1:
            images = torch.stack(images)
        else:
            # Get max dimensions
            max_h = max(img.shape[-2] for img in images)
            max_w = max(img.shape[-1] for img in images)
            
            # Pad images to max size
            padded_images = []
            for img in images:
                h, w = img.shape[-2:]
                pad_h = max_h - h
                pad_w = max_w - w
                padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                padded_images.append(padded)
            images = torch.stack(padded_images)
        
        # Handle text embeddings dictionary
        batched_text_embeds = {}
        for key in text_embeds[0].keys():
            if len(set(emb[key].shape for emb in text_embeds)) <= 1:
                # All same size, can stack directly
                batched_text_embeds[key] = torch.stack([emb[key] for emb in text_embeds])
            else:
                # Need to pad sequences
                max_len = max(emb[key].shape[1] for emb in text_embeds)
                padded_embs = []
                for emb in text_embeds:
                    curr_len = emb[key].shape[1]
                    if curr_len < max_len:
                        pad = torch.zeros(emb[key].shape[0], max_len - curr_len, emb[key].shape[2],
                                        dtype=emb[key].dtype, device=emb[key].device)
                        padded = torch.cat([emb[key], pad], dim=1)
                    else:
                        padded = emb[key]
                    padded_embs.append(padded)
                batched_text_embeds[key] = torch.stack(padded_embs)
        
        # Stack tag weights
        tag_weights = torch.stack(tag_weights)
        
        return images, batched_text_embeds, tag_weights

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ) -> DataLoader:
        sampler = AspectBatchSampler(dataset, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=NovelAIDiffusionV3Trainer.collate_fn
        )

    def save_checkpoint(self, checkpoint_dir: str, epoch: int) -> None:
        """Save model and training state to checkpoint directory using diffusers format in FP16.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
        """
        if not self.accelerator.is_main_process:
            return
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save UNet in fp16
        self.model.save_pretrained(
            os.path.join(checkpoint_dir, "unet"),
            safe_serialization=True,
            dtype=torch.float16
        )
        self.model.to(dtype=torch.bfloat16)  # Restore original dtype
        
        # Save VAE in fp16 if it exists
        if self.vae is not None:
            orig_dtype = self.vae.dtype
            self.vae.save_pretrained(
                os.path.join(checkpoint_dir, "vae"),
                safe_serialization=True,
                dtype=torch.float16
            )
            self.vae.to(dtype=orig_dtype)  # Restore original dtype
        
        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_config": self.scheduler.config
        }
        
        if hasattr(self, 'lr_scheduler'):
            training_state["lr_scheduler_state"] = self.lr_scheduler.state_dict()
        
        # Save training state using safetensors
        training_state_path = os.path.join(checkpoint_dir, "training_state.safetensors")
        from safetensors.torch import save_file
        
        # Convert non-tensor values to tensors for safetensors
        training_state = {
            k: (v if isinstance(v, torch.Tensor) else torch.tensor(v))
            for k, v in training_state.items() 
            if v is not None
        }
        
        save_file(training_state, training_state_path)
        
        # Save model config
        self.model.config.save_pretrained(os.path.join(checkpoint_dir, "unet"))
        
        # Log to wandb if available
        if wandb.run is not None:
            artifact = wandb.Artifact(
                f"model-epoch-{epoch}", 
                type="model",
                description=f"Model checkpoint from epoch {epoch}"
            )
            artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(artifact)
        
        print(f"Saved checkpoint for epoch {epoch} to {checkpoint_dir} in fp16")

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
        """Prepare hidden states using pre-allocated buffer."""
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        
        # Ensure input is bfloat16
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        
        # Reshape to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.reshape(-1, 768)
        
        # Project
        projected = self.hidden_proj(hidden_states)
        
        # Reshape back to (batch_size, seq_len, cross_attention_dim)
        return projected.reshape(batch_size, seq_len, -1)

    def _precompute_snr_weights(self, alphas: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Pre-compute SNR weights for all possible timesteps."""
        alpha_t = alphas[timesteps]
        snr = alpha_t / (1 - alpha_t)
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

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states using pre-computed projection matrix."""
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, 768)
        projected = torch.matmul(hidden_states, self.proj_matrix.t())
        return projected.reshape(batch_size, seq_len, -1)

    def _compute_karras_scalings(self, sigmas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial Karras scalings."""
        sigma_data = 0.5  # Default value from Karras et al.
        
        c_skip = sigma_data**2 / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigmas**2 + sigma_data**2).sqrt()
        
        return c_skip, c_out, c_in