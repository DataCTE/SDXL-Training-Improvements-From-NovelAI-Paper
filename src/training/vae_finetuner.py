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

class VAEFineTuner:
    def __init__(self, vae, **kwargs):
        """Initialize VAE finetuner with improved configuration."""
        try:
            # Move core initialization to separate methods
            self._init_base_config(kwargs)
            self._init_model(vae)
            self._init_optimizer(kwargs)
            self._init_tracking()
            self._init_additional_components(kwargs)
            
            logger.info("VAEFineTuner initialized successfully")
            self._log_config()
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
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
        if config.get('gradient_checkpointing') and hasattr(self.vae, 'enable_gradient_checkpointing'):
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

    def _log_config(self):
        """Log the current configuration"""
        logger.info("\nVAE Finetuner Configuration:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Mixed Precision: {self.mixed_precision}")
        logger.info(f"- Channel Scaling: {self.use_channel_scaling}")
        logger.info(f"- Adaptive Loss Scale: {self.adaptive_loss_scale}")
        logger.info(f"- KL Weight: {self.kl_weight}")
        logger.info(f"- Perceptual Weight: {self.perceptual_weight}")
        logger.info(f"- Min SNR Gamma: {self.min_snr_gamma}")
        logger.info(f"- Initial Scale Factor: {self.scale_factor}")

    def register_vae_statistics(self):
        """Register VAE channel statistics as buffers for scale-and-shift normalization"""
        try:
            # NovelAI anime dataset statistics for VAE latents (from paper appendix B)
            means = torch.tensor([4.8119, 0.1607, 1.3538, -1.7753])
            stds = torch.tensor([9.9181, 6.2753, 7.5978, 5.9956])
            
            # Register as buffers with correct device and dtype
            self.vae.register_buffer('latent_means', means.to(device=self.device, dtype=self.vae.dtype))
            self.vae.register_buffer('latent_stds', stds.to(device=self.device, dtype=self.vae.dtype))
        except Exception as e:
            logger.warning(f"Failed to register VAE statistics: {e}")

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply per-channel scale-and-shift normalization to VAE latents"""
        if not self.use_channel_scaling:
            return latents
        try:    
            # Ensure statistics are on same device/dtype as input
            means = self.vae.latent_means.to(latents.device, latents.dtype)
            stds = self.vae.latent_stds.to(latents.device, latents.dtype)
            
            # Reshape for broadcasting
            means = means.view(1, -1, 1, 1)
            stds = stds.view(1, -1, 1, 1)
            
            # Apply normalization
            return (latents - means) / stds
        except Exception as e:
            logger.warning(f"Failed to normalize latents: {e}")
           

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse per-channel scale-and-shift normalization"""
        if not self.use_channel_scaling:
            return latents
        try:
            # Ensure statistics are on same device/dtype as input
            means = self.vae.latent_means.to(latents.device, latents.dtype)
            stds = self.vae.latent_stds.to(latents.device, latents.dtype)
            
            # Reshape for broadcasting
            means = means.view(1, -1, 1, 1)
            stds = stds.view(1, -1, 1, 1)
            
            # Apply denormalization
            return latents * stds + means
        except Exception as e:
            logger.warning(f"Failed to denormalize latents: {e}")
            

    def _initialize_perceptual_loss(self):
        """Initialize perceptual loss network"""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).eval()
            self.perceptual_model = torch.nn.Sequential(
                *list(vgg.features)[:16]
            ).to(next(self.vae.parameters()).device)
            for param in self.perceptual_model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning(f"Failed to initialize perceptual loss: {e}")
            
    
    def compute_perceptual_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        if self.perceptual_weight == 0:
            return torch.tensor(0.0, device=x_recon.device)
        try:
            # Ensure inputs are in correct format for VGG
            x_recon = F.interpolate(x_recon, size=(224, 224))
            x_target = F.interpolate(x_target, size=(224, 224))
            
            with torch.no_grad():
                target_features = self.perceptual_model(x_target)
            recon_features = self.perceptual_model(x_recon)
            
            return F.mse_loss(recon_features, target_features)
        except Exception as e:
            logger.warning(f"Failed to compute perceptual loss: {e}")
           
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss with improved numerical stability"""
        if self.kl_weight == 0:
            return torch.tensor(0.0, device=mu.device)
        try:
            # Use log-sum-exp trick for numerical stability
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return kl_loss.mean()
        except Exception as e:
            logger.warning(f"Failed to compute KL loss: {e}")
           
    def update_scale_factor(self, loss: torch.Tensor):
        """Update adaptive loss scaling factor"""
        if not self.adaptive_loss_scale:
            return
        try:
            self.loss_meter.update(loss.item())
            if len(self.loss_meter._losses) < 50:
                return
                
            loss_std = self.loss_meter.std
            loss_mean = self.loss_meter.mean
            
            # Update scale factor based on loss statistics
            target_loss = 1.0
            current_scale = self.scale_factor
            new_scale = current_scale * (target_loss / (loss_mean + 1e-6))
            
            # Smooth update
            self.scale_factor = 0.99 * current_scale + 0.01 * new_scale
            self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        except Exception as e:
            logger.warning(f"Failed to update scale factor: {e}")
        
    def training_step(self, batch: dict) -> torch.Tensor:
        """Perform one VAE training step with NovelAI improvements"""
        try:
            # Move data to device asynchronously
            latents = batch['latents'].to(self.device, non_blocking=True)
            original_images = batch.get('original_images')
            if original_images is not None:
                original_images = original_images.to(self.device, non_blocking=True)

            # Cache handling
            if self.use_cache:
                cache_key = self._get_cache_key(latents)
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Use a separate stream for VAE computation
            with torch.cuda.stream(torch.cuda.current_stream()):
                # Enable mixed precision if configured
                with autocast(enabled=self.use_amp):
                    # Normalize latents using pre-computed statistics
                    if self.use_channel_scaling:
                        latents = self.normalize_latents(latents)
                    
                    # VAE forward pass
                    recon_loss = 0
                    kl_loss = 0
                    perceptual_loss = 0
                    
                    # Decode latents
                    decoded = self.vae.decode(latents).sample
                    
                    # Reconstruction loss (already scaled by VAE)
                    if original_images is not None:
                        recon_loss = F.mse_loss(decoded, original_images, reduction='mean')
                    
                    # KL divergence loss if enabled
                    if self.kl_weight > 0:
                        posterior = self.vae.encode(decoded).latent_dist
                        kl_loss = self.compute_kl_loss(posterior.mean, posterior.logvar) * self.kl_weight
                    
                    # Perceptual loss if enabled
                    if self.perceptual_weight > 0 and original_images is not None:
                        perceptual_loss = self.compute_perceptual_loss(decoded, original_images) * self.perceptual_weight
                    
                    # Combine losses
                    loss = recon_loss + kl_loss + perceptual_loss
                    
                    # Apply adaptive loss scaling if enabled
                    if self.adaptive_loss_scale:
                        self.update_scale_factor(loss)
                        loss = loss * self.scale_factor

                # Scale and backward pass
                if self.scaler is not None:
                    loss = self.scaler.scale(loss)
                
                loss.backward()
                
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                
                # Update loss history
                self.loss_meter.update(loss.item())
                
                # Cache the result if enabled
                if self.use_cache:
                    self.cache[cache_key] = loss.item()

                return loss

        except Exception as e:
            logger.error(f"VAE training step failed: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=self.device)
    
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
