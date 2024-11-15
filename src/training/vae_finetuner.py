import torch
import logging
import traceback
import os
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class VAELossMeter:
    """Thread-safe loss meter for VAE training."""
    window_size: int = 100
    _losses: list = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    
    def update(self, loss: float) -> None:
        with self._lock:
            self._losses.append(loss)
            if len(self._losses) > self.window_size:
                self._losses.pop(0)
    
    @property
    def mean(self) -> float:
        with self._lock:
            return np.mean(self._losses) if self._losses else 0.0
    
    @property
    def std(self) -> float:
        with self._lock:
            return np.std(self._losses) if len(self._losses) > 1 else 0.0

@torch.jit.script
def _normalize_latents(
    latents: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled latent normalization"""
    return (latents - means) / stds

@torch.jit.script
def _denormalize_latents(
    latents: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled latent denormalization"""
    return latents * stds + means

@torch.jit.script
def _compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """JIT-compiled KL divergence computation"""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

class VAEFineTuner:
    def __init__(self, vae, **kwargs):
        """Initialize VAE finetuner with optimizations"""
        try:
            self._init_base_config(kwargs)
            self._init_model(vae)
            self._init_optimizer(kwargs)
            self._init_tracking()
            
            # Pre-compile critical ops
            self._init_compiled_ops()
            
            # Pre-allocate buffers
            self._init_buffers()
            
            logger.info("VAEFineTuner initialized successfully")
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            raise

    def _init_base_config(self, config: Dict[str, Any]) -> None:
        """Initialize base configuration parameters."""
        self.device = config.get('device') or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_snr_gamma = config.get('min_snr_gamma', 5.0)
        self.adaptive_loss_scale = config.get('adaptive_loss_scale', True)
        self.use_channel_scaling = config.get('use_channel_scaling', True)
        self.mixed_precision = config.get('mixed_precision', 'bf16')
        self.use_amp = config.get('use_amp', True)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Initialize loss weights
        self.kl_weight = config.get('kl_weight', 0.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.0)
        self.scale_factor = config.get('scale_factor', 1.0)

    def _init_model(self, vae: torch.nn.Module) -> None:
        """Initialize and configure VAE model."""
        self.vae = vae.to(self.device)
        
        # Configure model precision
        if self.use_amp and self.mixed_precision == "fp16":
            self.vae = self.vae.to(dtype=torch.float16)
        elif self.mixed_precision == "bf16":
            self.vae = self.vae.to(dtype=torch.bfloat16)
            
        # Enable optimizations
        if hasattr(self.vae, 'enable_xformers_memory_efficient_attention'):
            self.vae.enable_xformers_memory_efficient_attention()
        if self.gradient_checkpointing and hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()

    def _init_optimizer(self, config: Dict[str, Any]) -> None:
        """Initialize optimizer and related components."""
        optim_cls = AdamW8bit if config.get('use_8bit_adam', True) else torch.optim.AdamW
        self.optimizer = optim_cls(
            self.vae.parameters(),
            lr=config.get('learning_rate', 1e-6),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            weight_decay=config.get('adam_weight_decay', 0.01),
            eps=config.get('adam_epsilon', 1e-8)
        )
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 1000),
            num_training_steps=10000
        )
        
        self.scaler = GradScaler() if self.use_amp else None

    def _init_tracking(self) -> None:
        """Initialize tracking metrics."""
        self.loss_meter = VAELossMeter(window_size=100)
        self.latent_count = 0
        self.latent_means = None
        self.latent_m2 = None

    def _init_compiled_ops(self):
        """Initialize compiled operations"""
        # Register VAE statistics once
        self.register_vae_statistics()
        
        # Pre-compile VGG features extraction
        if self.perceptual_weight > 0:
            self._init_perceptual_model()
            
        # Initialize CUDA streams
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()

    def _init_buffers(self):
        """Pre-allocate reusable buffers"""
        self.loss_buffer = torch.zeros(1, device=self.device)
        if self.perceptual_weight > 0:
            self.perceptual_buffer = torch.zeros(
                3, 224, 224, 
                device=self.device,
                dtype=self.vae.dtype
            )

    @torch.cuda.amp.autocast()
    def _forward_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """Optimized VAE forward pass"""
        with torch.cuda.stream(self.compute_stream):
            if self.use_channel_scaling:
                latents = _normalize_latents(
                    latents,
                    self.vae.latent_means.view(1, -1, 1, 1),
                    self.vae.latent_stds.view(1, -1, 1, 1)
                )
            return self.vae.decode(latents).sample

    @torch.no_grad()
    def _compute_perceptual(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """Optimized perceptual loss computation"""
        # Reuse pre-allocated buffer
        x_recon = F.interpolate(
            x_recon, 
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
            out=self.perceptual_buffer
        )
        x_target = F.interpolate(
            x_target,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
            out=self.perceptual_buffer
        )
        
        with torch.cuda.stream(self.compute_stream):
            target_features = self.perceptual_model(x_target)
            recon_features = self.perceptual_model(x_recon)
            return F.mse_loss(recon_features, target_features)

    def training_step(self, batch: dict) -> torch.Tensor:
        """Optimized VAE training step"""
        try:
            # Asynchronous data transfer
            with torch.cuda.stream(self.transfer_stream):
                latents = batch['latents'].to(
                    self.device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                original_images = batch.get('original_images')
                if original_images is not None:
                    original_images = original_images.to(
                        self.device,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )

            # Fast path for cached results
            if self.use_cache:
                cache_key = hash(latents.cpu().numpy().tobytes())
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Synchronized compute
            torch.cuda.synchronize()
            
            # Core training step
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Decode latents
                decoded = self._forward_vae(latents)
                
                # Compute losses efficiently
                loss = torch.zeros_like(self.loss_buffer)
                
                if original_images is not None:
                    loss += F.mse_loss(decoded, original_images)
                
                if self.kl_weight > 0:
                    posterior = self.vae.encode(decoded).latent_dist
                    loss += _compute_kl_loss(
                        posterior.mean,
                        posterior.logvar
                    ) * self.kl_weight
                
                if self.perceptual_weight > 0 and original_images is not None:
                    loss += self._compute_perceptual(
                        decoded,
                        original_images
                    ) * self.perceptual_weight
                
                # Scale loss
                if self.adaptive_loss_scale:
                    self.update_scale_factor(loss)
                    loss *= self.scale_factor

            # Optimizer step with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

            # Fast cleanup
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            
            # Update tracking
            self.loss_meter.update(loss.item())
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = loss.item()

            return loss

        except Exception as e:
            logger.error(f"VAE training step failed: {str(e)}")
            return torch.zeros_like(self.loss_buffer)
    
    def save_pretrained(self, save_dir: str):
        """Save VAE model and finetuning state"""
        os.makedirs(save_dir, exist_ok=True)
        self.vae.save_pretrained(save_dir)
        
        # Save finetuning state
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scale_factor': self.scale_factor,
            'loss_history': list(self.loss_meter._losses)
        }
        torch.save(state, os.path.join(save_dir, 'finetuning_state.pt'))
    
    def load_pretrained(self, load_dir: str):
        """Load VAE model and finetuning state"""
        self.vae.from_pretrained(load_dir)
        
        # Load finetuning state
        state_path = os.path.join(load_dir, 'finetuning_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.scale_factor = state['scale_factor']
            self.loss_meter._losses = deque(state['loss_history'], maxlen=self.loss_meter.window_size)

    def _get_cache_key(self, latents: torch.Tensor) -> str:
        """Generate a cache key for the input latents"""
        # Using a hash of the tensor data as the cache key
        return str(hash(latents.cpu().numpy().tobytes()))
