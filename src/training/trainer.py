import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import wandb
import os
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from ..data.dataset import NovelAIDataset
from ..data.sampler import AspectBatchSampler
import tqdm

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        gradient_accumulation_steps: int = 4,
    ):
        super().__init__()
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        
        # Add gradient accumulation steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Add projection layer for CLIP embeddings
        self.hidden_proj = nn.Linear(768, model.config.cross_attention_dim).to(
            device=device, 
            dtype=torch.float32
        )
        
        # Pre-allocate tensors for time embeddings
        self.register_buffer('base_area', torch.tensor(1024 * 1024, dtype=torch.float32))
        self.register_buffer('aesthetic_score', torch.tensor(6.0, dtype=torch.bfloat16))
        self.register_buffer('crop_score', torch.tensor(3.0, dtype=torch.bfloat16))
        self.register_buffer('zero_score', torch.tensor(0.0, dtype=torch.bfloat16))
        
        # Initialize ZTSNR parameters
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = 20000.0
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Enable memory efficient attention
        model.enable_xformers_memory_efficient_attention()
        
        # Enable channels last memory format
        model = model.to(memory_format=torch.channels_last)
        
        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Disable debug APIs
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

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

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized time_ids computation for H100"""
        batch_size = images.shape[0]
        orig_height = images.shape[2] * 8
        orig_width = images.shape[3] * 8
        
        add_time_ids = torch.empty((batch_size, 2, 4), device=self.device, dtype=torch.bfloat16)
        add_time_ids[:, 0, 0] = orig_height
        add_time_ids[:, 0, 1] = orig_width
        add_time_ids[:, 0, 2] = self.aesthetic_score
        add_time_ids[:, 0, 3] = self.zero_score
        add_time_ids[:, 1, 0] = orig_height
        add_time_ids[:, 1, 1] = orig_width
        add_time_ids[:, 1, 2] = self.crop_score
        add_time_ids[:, 1, 3] = self.zero_score
        
        return add_time_ids.reshape(batch_size, -1)

    def get_karras_scalings(self, sigma, sigma_data=1.0):
        # sigma: [batch_size] tensor
        sigma_sq = sigma * sigma
        sigma_data_sq = sigma_data * sigma_data
        denominator = sigma_data_sq + sigma_sq
        c_skip = sigma_data_sq / denominator
        c_out = -sigma_data * sigma / torch.sqrt(denominator)
        c_in = 1.0 / torch.sqrt(denominator)
        return c_skip, c_out, c_in

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
        # SNR(σ) = (σ_data / σ)²
        return (self.sigma_data / sigma).square()

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Uses the scheduler's alpha values to compute SNR(t) = α/(1-α)
        # w(t) = min(SNR(t), γ)
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]  # [batch_size]
        
        # SNR in terms of alpha: SNR = α/(1-α)
        snr = alpha_t / (1 - alpha_t)
        
        # Clamp to min_snr_gamma
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr, min_snr).float()
        return weights

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # Pre-compute device transfers with non-blocking for better parallelization
        images = images.to(self.device, non_blocking=True)
        text_embeds = {k: v.to(self.device, non_blocking=True) for k,v in text_embeds.items()}
        tag_weights = tag_weights.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        running_loss = 0.0
        total_loss = 0.0
        total_v_pred = None

        for i in range(self.gradient_accumulation_steps):
            torch.cuda.empty_cache()  # Keep this from old version as it helped with memory

            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size

            # Extract micro-batch and scale - use clone() for safety
            batch_latents = images[start_idx:end_idx].clone()
            
            # Apply area-based noise scaling
            height, width = batch_latents.shape[2:4]
            area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
            noise_scale = torch.sqrt(area / self.base_area)
            batch_latents = batch_latents * noise_scale

            batch_tag_weights = tag_weights[start_idx:end_idx].view(-1,1,1,1)

            # Prepare text embeddings with explicit cloning
            base_hidden = text_embeds["base_text_embeds"][start_idx:end_idx].squeeze(1).clone()
            base_pooled = text_embeds["base_pooled_embeds"][start_idx:end_idx].squeeze(1).clone()

            # Project text embeddings with explicit float32 conversion
            base_hidden_float32 = base_hidden.to(dtype=torch.float32)
            batch_size, seq_len, _ = base_hidden_float32.shape
            encoder_hidden_states = self.hidden_proj(
                base_hidden_float32.view(-1, 768)
            ).view(batch_size, seq_len, -1)

            # Generate noise with channels_last memory format
            noise = torch.randn_like(
                batch_latents,
                device=self.device,
                dtype=batch_latents.dtype,
                memory_format=torch.channels_last
            )

            # More efficient timesteps generation
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (micro_batch_size,),
                device=self.device,
                dtype=torch.long,
                requires_grad=False
            )

            # Get sigmas and compute scalings
            sigmas = self.sigmas[timesteps]
            c_skip, c_out, c_in = self.get_karras_scalings(sigmas)

            # Add noise efficiently
            sigma_expanded = sigmas.view(-1,1,1,1)
            noisy_latents = batch_latents + sigma_expanded * noise
            v_target = self.scheduler.get_velocity(batch_latents, noise, timesteps)

            # Generate time embeddings
            time_ids = self._get_add_time_ids(batch_latents).clone()

            # Prepare model inputs with explicit gradient requirements
            scaled_input = (c_in.view(-1,1,1,1) * noisy_latents).clone().requires_grad_(True)
            encoder_hidden_states = encoder_hidden_states.clone().requires_grad_(True)
            timesteps = timesteps.clone()

            # Forward pass with autocast
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                F_out = self.model(
                    scaled_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": base_pooled.clone(),
                        "time_ids": time_ids
                    }
                ).sample

                # Compute denoised output
                D_out = c_skip.view(-1,1,1,1) * noisy_latents + c_out.view(-1,1,1,1) * F_out

                # Compute loss efficiently
                loss_per_sample = F.mse_loss(
                    D_out.float(),
                    v_target.float(),
                    reduction='none'
                ).mean(dim=[1,2,3])

                # Apply weights
                snr_weights = self.get_minsnr_weights(timesteps)
                loss_per_sample = loss_per_sample * batch_tag_weights.squeeze() * snr_weights
                loss = loss_per_sample.mean() / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update loss tracking
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value

            # Accumulate predictions
            total_v_pred = D_out.detach() if total_v_pred is None else torch.cat([total_v_pred, D_out.detach()], dim=0)

        avg_loss = running_loss / self.gradient_accumulation_steps
        return total_loss, batch_latents, total_v_pred, timesteps, avg_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> float:
        self.current_epoch = epoch
        self.model.train()
        
        # Pre-allocate metrics tracking
        total_loss = 0.0
        num_batches = len(dataloader)
        running_grad_norm = 0.0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            total=num_batches,
            disable=not self.accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Efficient batch transfer to device - use non_blocking for async transfer
            images, text_embeds, tag_weights = [
                x.to(self.device, dtype=torch.bfloat16, non_blocking=True) 
                if isinstance(x, torch.Tensor) else 
                {k: v.to(self.device, dtype=torch.bfloat16, non_blocking=True) for k,v in x.items()}
                for x in batch
            ]
            
            # Training step with autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, pred_images, v_pred, timesteps, avg_batch_loss = self.training_step(
                    images, text_embeds, tag_weights
                )
            
            # Compute gradient norm before optimizer step
            grad_norm = self.compute_grad_norm()
            running_grad_norm = 0.9 * running_grad_norm + 0.1 * grad_norm
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics efficiently
            total_loss += avg_batch_loss
            avg_loss = total_loss / (batch_idx + 1)
            
            # Log metrics less frequently to reduce overhead
            if self.accelerator.is_main_process and batch_idx % log_interval == 0:
                metrics = {
                    'grad/norm': running_grad_norm,
                    'loss/batch': avg_batch_loss,
                    'loss/running_avg': avg_loss,
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'memory/allocated_gb': torch.cuda.memory_allocated()/1e9,
                    'memory/reserved_gb': torch.cuda.max_memory_reserved()/1e9
                }
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{avg_batch_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'grad': f'{running_grad_norm:.4f}',
                    'lr': f'{metrics["lr"]:.2e}',
                    'mem': f'{metrics["memory/allocated_gb"]:.1f}GB'
                })
                
                # Log to wandb asynchronously
                wandb.log(metrics, step=self.global_step)
            
            self.global_step += 1
            
            # Optional: Update learning rate scheduler if using one
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()
        
        # Clean up progress bar
        progress_bar.close()
        
        # Return average loss for the epoch
        return total_loss / num_batches

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
            persistent_workers=persistent_workers
        )