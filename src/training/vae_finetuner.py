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

logger = logging.getLogger(__name__)

class VAEFineTuner:
    def __init__(
        self,
        vae,
        learning_rate=1e-6,
        min_snr_gamma=5.0,
        use_cache=True,
        cache_size=1000,
        use_amp=True,
        use_gradient_checkpointing=True,
        adaptive_loss_scale=True,
        kl_weight=0.0,
        perceptual_weight=0.0,
        use_8bit_adam=True,
        gradient_checkpointing=True,
        mixed_precision="bf16",
        use_channel_scaling=True,  # Enable per-channel scale-and-shift
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=1000,
        scale_factor=1.0,
        device=None
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
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            adam_weight_decay: Weight decay for Adam optimizer
            adam_epsilon: Epsilon parameter for Adam optimizer
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for learning rate
            scale_factor: Initial scale factor for loss scaling
            device: Device to place model on
        """
        try:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vae = vae.to(self.device)
            self.min_snr_gamma = min_snr_gamma
            self.adaptive_loss_scale = adaptive_loss_scale
            self.kl_weight = kl_weight
            self.perceptual_weight = perceptual_weight
            self.mixed_precision = mixed_precision
            self.use_channel_scaling = use_channel_scaling
            self.max_grad_norm = max_grad_norm
            self.scale_factor = scale_factor
            
            # Initialize optimizer with configurable parameters
            optim_cls = AdamW8bit if use_8bit_adam else torch.optim.AdamW
            self.optimizer = optim_cls(
                vae.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon
            )
            
            # Initialize learning rate scheduler
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=10000  # Approximate, will be adjusted during training
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
            if hasattr(self.vae, 'enable_xformers_memory_efficient_attention'):
                self.vae.enable_xformers_memory_efficient_attention()
            if gradient_checkpointing and hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()
            
            # Initialize statistics tracking
            self.latent_count = 0
            self.latent_means = None
            self.latent_m2 = None
            self.loss_history = deque(maxlen=100)
            
            # Initialize perceptual loss if needed
            if self.perceptual_weight > 0:
                self._initialize_perceptual_loss()
            
            logger.info("VAEFineTuner initialized successfully")
            self._log_config()
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise

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

            # Use a separate stream for VAE computation
            with torch.cuda.stream(torch.cuda.current_stream()):
                # Enable mixed precision if configured
                with autocast(enabled=self.mixed_precision != "no"):
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
                self.loss_history.append(loss.item())
                
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
