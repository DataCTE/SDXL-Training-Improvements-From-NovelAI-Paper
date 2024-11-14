import torch
import logging
import traceback
import os
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class VAEFineTuner:
    def __init__(
        self,
        vae,
        learning_rate=1e-6,
        min_snr_gamma=5.0,
        adaptive_loss_scale=True,
        kl_weight=0.0,
        perceptual_weight=0.0,
        use_8bit_adam=True,
        gradient_checkpointing=True,
        mixed_precision="bf16",
        use_channel_scaling=True  # Enable per-channel scale-and-shift
    ):
        """
        Initialize VAE finetuner with NovelAI improvements.
        
        Args:
            vae: VAE model to finetune
            learning_rate: Base learning rate
            min_snr_gamma: Minimum SNR gamma for loss weighting
            adaptive_loss_scale: Whether to use adaptive loss scaling
            kl_weight: Weight for KL divergence loss
            perceptual_weight: Weight for perceptual loss
            use_8bit_adam: Whether to use 8-bit Adam
            gradient_checkpointing: Whether to use gradient checkpointing
            mixed_precision: Mixed precision type ("no", "fp16", "bf16")
            use_channel_scaling: Whether to use per-channel scale-and-shift normalization
        """
        try:
            self.vae = vae
            self.min_snr_gamma = min_snr_gamma
            self.adaptive_loss_scale = adaptive_loss_scale
            self.kl_weight = kl_weight
            self.perceptual_weight = perceptual_weight
            self.mixed_precision = mixed_precision
            self.use_channel_scaling = use_channel_scaling
            
            # Initialize optimizer
            optim_cls = AdamW8bit if use_8bit_adam else torch.optim.AdamW
            self.optimizer = optim_cls(
                vae.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            
            # Initialize mixed precision training
            self.scaler = GradScaler() if mixed_precision == "fp16" else None
            
            # Convert model precision
            if mixed_precision == "fp16":
                self.vae = self.vae.to(dtype=torch.float16)
            elif mixed_precision == "bf16":
                self.vae = self.vae.to(dtype=torch.bfloat16)
            
            # Move channel statistics to device and dtype
            self.register_vae_statistics()
            
            # Enable memory optimizations
            self.vae.enable_xformers_memory_efficient_attention()
            if gradient_checkpointing:
                self.vae.enable_gradient_checkpointing()
            
            # Initialize statistics tracking
            self.latent_count = 0
            self.latent_means = None
            self.latent_m2 = None
            self.loss_history = deque(maxlen=100)
            self.scale_factor = 1.0
            
            # Initialize perceptual loss if needed
            if self.perceptual_weight > 0:
                self._initialize_perceptual_loss()
            
            logger.info("VAEFineTuner initialized successfully")
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise

    def register_vae_statistics(self):
        """Register VAE channel statistics as buffers"""
        # NovelAI anime dataset statistics for VAE latents (from paper appendix B)
        means = torch.tensor([4.8119, 0.1607, 1.3538, -1.7753])
        stds = torch.tensor([9.9181, 6.2753, 7.5978, 5.9956])
        
        # Register as buffers with correct device and dtype
        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        
        self.vae.register_buffer('latent_means', means.to(device, dtype))
        self.vae.register_buffer('latent_stds', stds.to(device, dtype))
    
    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply per-channel normalization to latents"""
        if not self.use_channel_scaling:
            return latents
            
        means = self.vae.latent_means.view(1, -1, 1, 1)
        stds = self.vae.latent_stds.view(1, -1, 1, 1)
        return (latents - means) / stds
    
    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse per-channel normalization of latents"""
        if not self.use_channel_scaling:
            return latents
            
        means = self.vae.latent_means.view(1, -1, 1, 1)
        stds = self.vae.latent_stds.view(1, -1, 1, 1)
        return latents * stds + means
    
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
            self.perceptual_weight = 0.0
    
    def compute_perceptual_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        if self.perceptual_weight == 0:
            return torch.tensor(0.0, device=x_recon.device)
            
        # Ensure inputs are in correct format for VGG
        x_recon = F.interpolate(x_recon, size=(224, 224))
        x_target = F.interpolate(x_target, size=(224, 224))
        
        with torch.no_grad():
            target_features = self.perceptual_model(x_target)
        recon_features = self.perceptual_model(x_recon)
        
        return F.mse_loss(recon_features, target_features)
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss with improved numerical stability"""
        if self.kl_weight == 0:
            return torch.tensor(0.0, device=mu.device)
            
        # Use log-sum-exp trick for numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()
    
    def update_scale_factor(self, loss: torch.Tensor):
        """Update adaptive loss scaling factor"""
        if not self.adaptive_loss_scale:
            return
            
        self.loss_history.append(loss.item())
        if len(self.loss_history) < 50:
            return
            
        loss_std = np.std(self.loss_history)
        loss_mean = np.mean(self.loss_history)
        
        # Update scale factor based on loss statistics
        target_loss = 1.0
        current_scale = self.scale_factor
        new_scale = current_scale * (target_loss / (loss_mean + 1e-6))
        
        # Smooth update
        self.scale_factor = 0.99 * current_scale + 0.01 * new_scale
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
    
    def training_step(self, batch: dict) -> torch.Tensor:
        """Perform one VAE training step with NovelAI improvements"""
        try:
            self.vae.train()
            images = batch['pixel_values']
            
            # Apply channel normalization
            if self.use_channel_scaling:
                images = self.normalize_latents(images)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.mixed_precision != "no"):
                # Encode-decode
                posterior = self.vae.encode(images)
                latents = posterior.sample()
                reconstruction = self.vae.decode(latents).sample
                
                # Compute reconstruction loss with adaptive scaling
                recon_loss = F.mse_loss(reconstruction, images)
                recon_loss = recon_loss * self.scale_factor
                
                # Compute additional losses
                kl_loss = self.compute_kl_loss(posterior.mean, posterior.logvar)
                perceptual_loss = self.compute_perceptual_loss(reconstruction, images)
                
                # Combine losses
                loss = (
                    recon_loss +
                    self.kl_weight * kl_loss +
                    self.perceptual_weight * perceptual_loss
                )
            
            # Scale loss and backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update adaptive scaling
            self.update_scale_factor(recon_loss)
            
            # Log metrics
            metrics = {
                'vae/recon_loss': recon_loss.item(),
                'vae/kl_loss': kl_loss.item(),
                'vae/perceptual_loss': perceptual_loss.item(),
                'vae/scale_factor': self.scale_factor
            }
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in VAE training step: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def save_pretrained(self, save_dir: str):
        """Save VAE model and finetuning state"""
        os.makedirs(save_dir, exist_ok=True)
        self.vae.save_pretrained(save_dir)
        
        # Save finetuning state
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scale_factor': self.scale_factor,
            'loss_history': list(self.loss_history)
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
            self.loss_history = deque(state['loss_history'], maxlen=self.loss_history.maxlen)
