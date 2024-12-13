import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from memory.Manager import MemoryManager
from data.dataset import NovelAIDataset
from data.sampler import AspectBatchSampler
from accelerate import Accelerator
import time
import os
import wandb
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from .training_utils import TrainingProfiler, AutoTuner
from configs.training_config import TrainingConfig
from .progress import TrainProgress


class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        batch_size: int = 1,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        max_vram_usage: float = 0.8,
        gradient_accumulation_steps: int = 4,
        config: Optional[TrainingConfig] = None
    ):
        super().__init__()
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize memory management systems
        self.memory_manager = MemoryManager(max_vram_usage=max_vram_usage)
        # Setup memory management before registering modules
        self.memory_manager.setup_memory_management(model, device)
        
        # Initialize activation allocator
        from memory.layeroffloading import StaticActivationAllocator
        self.activation_allocator = StaticActivationAllocator(model)
        self.activation_allocator.allocate_buffers(device)
        
        # Register all layers with memory manager
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:
                self.memory_manager.register_module(name, module)
        
        # Initialize profiler and auto-tuner with config and memory management
        self.profiler = TrainingProfiler(window_size=50, config=config)
        self.auto_tuner = AutoTuner(
            initial_params={
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            config=config,
            min_samples=50
        )
        
        # Add memory monitoring to profiler
        if hasattr(self.memory_manager, 'get_current_usage'):
            self.profiler.add_memory_callback(self.memory_manager.get_current_usage)
        else:
            print("Warning: Memory manager does not support current usage tracking")
        
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
        
        # Update ZTSNR parameters
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = float('inf')
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Replace individual progress tracking with TrainProgress
        self.progress = TrainProgress()
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        # Allocate activation buffers after model is fully initialized
        self.activation_allocator.allocate_buffers(device)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        added_cond_kwargs: Dict = None,
        current_layer: str = None,
        phase: str = 'forward'
    ) -> torch.Tensor:
        """Forward pass with memory-efficient layer handling"""
        # Handle layer memory management if needed
        if current_layer:
            self.memory_manager.before_layer_computation(current_layer, phase)
        
        try:
            # Call UNet's forward pass
            return self.model(
                sample=x,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
        finally:
            if current_layer:
                self.memory_manager.after_layer_computation(current_layer)

    def training_step(
        self,
        latents: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-efficient training step with gradient accumulation"""
        # Memory management at start of step
        self.memory_manager.clear_cache()
        
        # Initialize accumulated loss
        accumulated_loss = 0
        
        for accumulation_step in range(self.gradient_accumulation_steps):
            # Get batch slice for current accumulation step
            batch_size = latents.shape[0] // self.gradient_accumulation_steps
            start_idx = accumulation_step * batch_size
            end_idx = start_idx + batch_size
            
            batch_latents = latents[start_idx:end_idx].reshape(batch_size, -1, latents.shape[-2], latents.shape[-1])
            batch_text_embeds = {k: v[start_idx:end_idx] for k, v in text_embeds.items()}
            batch_tag_weights = tag_weights[start_idx:end_idx]
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Ensure tag_weights has correct shape for broadcasting
                batch_tag_weights = batch_tag_weights.view(-1, 1, 1, 1)
                
                with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16):
                    # Get latent dimensions
                    height = batch_latents.shape[2] * 8
                    width = batch_latents.shape[3] * 8
                    
                    area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
                    noise_scale = torch.sqrt(area / self.base_area)
                    
                    # Sample timesteps and get sigmas
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
                    sigma = self.sigmas[timesteps] * noise_scale
                    
                    # Generate noise
                    noise = torch.randn_like(batch_latents, dtype=torch.float32, device=self.device)
                    is_infinite = torch.isinf(sigma)
                    
                    # Apply noising
                    noisy_images = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        noise,
                        batch_latents + noise * sigma.view(-1, 1, 1, 1)
                    )
                    
                    # Get Karras scalings
                    c_skip, c_out, c_in = self.get_karras_scalings(sigma)
                    model_input = noisy_images * c_in.view(-1, 1, 1, 1)
                    
                    # Process text embeddings efficiently with non-blocking transfers
                    base_hidden = batch_text_embeds["base_text_embeds"].to(
                        device=self.device,
                        dtype=torch.bfloat16,
                        non_blocking=True
                    ).squeeze(1)
                    
                    base_pooled = batch_text_embeds["base_pooled_embeds"].to(
                        device=self.device,
                        dtype=torch.bfloat16,
                        non_blocking=True
                    ).squeeze(1)
                    
                    # Project text embeddings
                    batch_size, seq_len, _ = base_hidden.shape
                    base_hidden_float32 = base_hidden.to(dtype=torch.float32)
                    encoder_hidden_states = self.hidden_proj(
                        base_hidden_float32.view(-1, 768)
                    ).view(batch_size, seq_len, -1)
                    
                    # Get time embeddings
                    time_ids = self._get_add_time_ids(batch_latents)
                    added_cond_kwargs = {
                        "text_embeds": base_pooled,
                        "time_ids": time_ids
                    }
                
                # Forward pass with layer offloading and activation checkpointing
                v_prediction = None
                for layer_idx, (name, layer) in enumerate(self.model.named_modules()):
                    if not list(layer.parameters()):
                        continue
                    
                    if v_prediction is not None:
                        self.activation_allocator.store_activation(name, v_prediction)
                    
                    self.layer_conductor.before_layer(layer_idx, name)
                    
                    with torch.amp.autocast(dtype=torch.bfloat16):
                        v_prediction = self.forward(
                            model_input if layer_idx == 0 else v_prediction,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs,
                            current_layer=name,
                            phase='forward'
                        ).sample
                    
                    self.layer_conductor.after_layer(layer_idx, name)
                    self.memory_manager.clear_cache()
                
                # Compute prediction and loss
                with torch.amp.autocast(dtype=torch.bfloat16):
                    pred_images = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        -self.sigma_data * v_prediction,
                        noisy_images * c_skip.view(-1, 1, 1, 1) + v_prediction * c_out.view(-1, 1, 1, 1)
                    )
                    
                    # Compute loss with SNR weighting
                    loss = F.mse_loss(pred_images, batch_latents, reduction='none')
                    snr = self.get_snr(sigma)
                    min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
                    snr_weights = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        min_snr.view(-1, 1, 1, 1),
                        torch.minimum(snr, min_snr).view(-1, 1, 1, 1)
                    )
                    
                    # Apply tag weights and reduce
                    loss = loss * snr_weights * batch_tag_weights
                    loss = loss.mean() / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            
            # Clear activations and cache
            self.activation_allocator.clear()
            self.memory_manager.clear_cache()
        
        # Clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return accumulated_loss, pred_images, v_prediction, timesteps

    def load_checkpoint(self, path: str):
        """Load checkpoint with progress information"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore progress state
        if 'progress' in checkpoint:
            self.progress = checkpoint['progress']
        
        # Restore profiler and auto-tuner states if available
        if 'profiler_state' in checkpoint:
            self.profiler.metrics = checkpoint['profiler_state']
        if 'auto_tuner_state' in checkpoint:
            self.auto_tuner.current_params = checkpoint['auto_tuner_state']['current_params']
            self.auto_tuner.best_params = checkpoint['auto_tuner_state']['best_params']
            self.auto_tuner.best_throughput = checkpoint['auto_tuner_state']['best_throughput']

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

    def get_sigmas(self) -> torch.Tensor:
        """
        Generate noise schedule with zero-terminal SNR, handling infinite noise timestep.
        The last timestep is set to sigma = ∞ for ZTSNR.
        """
        # Create regular sigma schedule for t in [0,1)
        ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigmas = torch.empty_like(ramp)
        
        # Regular steps (all except last)
        sigmas[:-1] = (min_inv_rho * (1 - ramp[:-1])) ** self.rho
        
        # Set final step to infinity for ZTSNR
        sigmas[-1] = float('inf')
        
        return sigmas.to(self.device)

    def get_karras_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified Karras Preconditioner scaling factors for v-prediction with ZTSNR support.
        For σ = ∞:
            cskip(σ) = 0
            cout(σ) = -σ_data
            cin(σ) = 1/√(σ² + σ_data²)
        """
        is_infinite = torch.isinf(sigma)
        sigma_sq = sigma * sigma
        sigma_data_sq = self.sigma_data * self.sigma_data
        denominator = sigma_data_sq + sigma_sq
        denominator_sqrt = torch.sqrt(denominator)
        
        # Handle infinite sigma case explicitly
        c_skip = torch.where(is_infinite,
                            torch.zeros_like(sigma),  # cskip(∞) = 0
                            sigma_data_sq / denominator)
        
        c_out = torch.where(is_infinite,
                           -self.sigma_data * torch.ones_like(sigma),  # cout(∞) = -σ_data
                           -sigma * self.sigma_data / denominator_sqrt)
        
        c_in = 1.0 / denominator_sqrt  # cin(σ) = 1/√(σ² + σ_data²)
        
        return c_skip, c_out, c_in

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-Noise Ratio as defined in the paper for given sigma:
        SNR(σ) = (σ_data / σ)²
        """
        return (self.sigma_data / sigma)**2

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute MinSNR weights as described in the paper.
        Uses alphas from the scheduler:
        snr_t = alpha_t / (1 - alpha_t), and then clamp snr_t by min_snr_gamma.
        
        weights = min(snr_t, min_snr_gamma)
        """
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]
        snr_t = alpha_t / (1 - alpha_t)
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr_t, min_snr)
        return weights.float()

    def train_epoch(self, dataloader: DataLoader, epoch: int, log_interval: int = 10) -> float:
        """Train for one epoch with performance profiling"""
        self.progress.epoch = epoch
        self.model.train()
        total_loss = 0
        
        with self.profiler.start_profiling() as prof:
            progress_bar = tqdm.tqdm(
                dataloader,
                desc=f"Epoch {self.progress.epoch}",
                leave=True,
                dynamic_ncols=True
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                start_time = time.time()
                
                with self.profiler.profile_range("training_step"):
                    loss, _, _, _ = self.training_step(batch[0], batch[1], batch[2])
                
                # Update progress tracking
                self.progress.next_step(batch[0].shape[0])
                
                # Record metrics
                batch_time = time.time() - start_time
                memory_used = torch.cuda.memory_allocated()
                self.profiler.record_step(
                    batch_time=batch_time,
                    batch_size=batch[0].shape[0],
                    memory_used=memory_used,
                    loss=loss.item()
                )
                
                # Auto-tune hyperparameters
                new_params = self.auto_tuner.update(self.profiler)
                if new_params != self.auto_tuner.current_params:
                    print("\nUpdating hyperparameters:")
                    for k, v in new_params.items():
                        print(f"  {k}: {v}")
                
                if prof is not None:
                    prof.step()
                
                total_loss += loss.item()
                
                # Update progress bar with more info
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'memory': f'{memory_used/1e9:.1f}GB',
                    'batch_time': f'{batch_time:.3f}s',
                    'global_step': self.progress.global_step,
                    'samples': self.progress.epoch_sample
                })
        
        # Update epoch progress
        self.progress.next_epoch()
        return total_loss / len(dataloader)

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True
    ) -> DataLoader:
        """Create optimized dataloader for training"""
        if batch_size is None:
            batch_size = 1  # Default batch size if not specified
        
        # Create sampler that groups by latent dimensions
        sampler = AspectBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # Create dataloader with optimized settings
        return DataLoader(
            dataset,
            batch_sampler=sampler,  # Use custom sampler for dimension-matched batches
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator().manual_seed(42),  # Reproducible shuffling
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),  # Seed workers
        )

    def save_checkpoint(self, path: str):
        """Save checkpoint with progress information"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'progress': self.progress,  # Save progress state
            'profiler_state': self.profiler.metrics,
            'auto_tuner_state': {
                'current_params': self.auto_tuner.current_params,
                'best_params': self.auto_tuner.best_params,
                'best_throughput': self.auto_tuner.best_throughput
            }
        }
        torch.save(checkpoint, path)

    def compute_grad_norm(self):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def log_detailed_metrics(self,
                            loss: torch.Tensor,
                            v_pred: torch.Tensor,
                            grad_norm: float,
                            timesteps: torch.Tensor):
        """Log detailed training metrics to W&B"""
        if not self.accelerator.is_main_process:
            return
        
        # Compute v-prediction statistics
        v_pred_mean = v_pred.mean().item()
        v_pred_std = v_pred.std().item()
        v_pred_min = v_pred.min().item()
        v_pred_max = v_pred.max().item()
        
        # Log detailed metrics
        wandb.log({
            'loss/total': loss.item(),
            'v_pred/mean': v_pred_mean,
            'v_pred/std': v_pred_std,
            'v_pred/min': v_pred_min,
            'v_pred/max': v_pred_max,
            'grad/norm': grad_norm,
            'timesteps/mean': timesteps.float().mean().item(),
            'timesteps/std': timesteps.float().std().item()
        })
