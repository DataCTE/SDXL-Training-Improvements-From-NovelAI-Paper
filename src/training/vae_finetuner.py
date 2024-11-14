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

# NovelAI anime dataset statistics for VAE latents
ANIME_VAE_MEANS = torch.tensor([4.8119, 0.1607, 1.3538, -1.7753])
ANIME_VAE_STDS = torch.tensor([9.9181, 6.2753, 7.5978, 5.9956])

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
        self.register_buffer('vae_means', ANIME_VAE_MEANS.to(self.vae.device, dtype=next(self.vae.parameters()).dtype))
        self.register_buffer('vae_stds', ANIME_VAE_STDS.to(self.vae.device, dtype=next(self.vae.parameters()).dtype))

    def normalize_latents(self, latents):
        """Apply per-channel scale-and-shift normalization"""
        if not self.use_channel_scaling:
            return latents
            
        # Ensure statistics are on same device/dtype
        if latents.device != self.vae_means.device or latents.dtype != self.vae_means.dtype:
            self.vae_means = self.vae_means.to(latents.device, dtype=latents.dtype)
            self.vae_stds = self.vae_stds.to(latents.device, dtype=latents.dtype)
            
        # Center and scale each channel
        return (latents - self.vae_means[None,:,None,None]) / self.vae_stds[None,:,None,None]

    def denormalize_latents(self, latents):
        """Reverse per-channel scale-and-shift normalization"""
        if not self.use_channel_scaling:
            return latents
            
        # Ensure statistics are on same device/dtype
        if latents.device != self.vae_means.device or latents.dtype != self.vae_means.dtype:
            self.vae_means = self.vae_means.to(latents.device, dtype=latents.dtype)
            self.vae_stds = self.vae_stds.to(latents.device, dtype=latents.dtype)
            
        # Rescale and recenter each channel
        return latents * self.vae_stds[None,:,None,None] + self.vae_means[None,:,None,None]

    def _initialize_perceptual_loss(self):
        """Initialize perceptual loss network"""
        try:
            from torchvision.models import vgg16
            import torch.nn as nn
            
            # Load pretrained VGG and freeze
            vgg = vgg16(pretrained=True)
            self.perceptual_net = nn.Sequential(*list(vgg.features)[:16]).eval()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
                
            # Move to same device and precision as VAE
            self.perceptual_net = self.perceptual_net.to(
                device=self.vae.device,
                dtype=next(self.vae.parameters()).dtype
            )
        except Exception as e:
            logger.error(f"Perceptual loss initialization failed: {str(e)}")
            self.perceptual_weight = 0.0

    def _compute_perceptual_loss(self, x, y):
        """Compute perceptual loss between x and y"""
        if self.perceptual_weight == 0:
            return 0.0
            
        x_features = self.perceptual_net(x)
        y_features = self.perceptual_net(y)
        return F.mse_loss(x_features, y_features)

    def _compute_kl_loss(self, posterior):
        """Compute KL divergence loss"""
        if self.kl_weight == 0:
            return 0.0
            
        kl_loss = posterior.kl().mean()
        return kl_loss * self.kl_weight

    def _compute_snr_weight(self, latents):
        """Compute SNR-based weight for reconstruction loss"""
        if self.min_snr_gamma <= 0:
            return 1.0
            
        # Calculate signal-to-noise ratio
        signal_power = torch.mean(torch.square(latents))
        noise_power = torch.mean(torch.square(torch.randn_like(latents)))
        snr = signal_power / (noise_power + 1e-8)
        
        # Apply minimum SNR gamma
        weight = torch.minimum(snr, torch.tensor(self.min_snr_gamma))
        return weight.item()

    def update_statistics(self, latents):
        """Update running mean and variance using Welford's online algorithm"""
        try:
            latents = latents.to(dtype=next(self.vae.parameters()).dtype)
            if self.latent_means is None:
                self.latent_means = torch.zeros(
                    latents.size(1),
                    device=latents.device,
                    dtype=latents.dtype
                )
                self.latent_m2 = torch.zeros_like(self.latent_means)
            
            flat_latents = latents.view(latents.size(0), latents.size(1), -1)
            
            for i in range(latents.size(0)):
                self.latent_count += 1
                delta = flat_latents[i].mean(dim=1) - self.latent_means
                self.latent_means += delta / self.latent_count
                delta2 = flat_latents[i].mean(dim=1) - self.latent_means
                self.latent_m2 += delta * delta2
                
        except Exception as e:
            logger.error(f"Statistics update failed: {str(e)}")
            logger.error(f"Update traceback: {traceback.format_exc()}")
    
    def get_statistics(self):
        """Get current latent statistics"""
        try:
            if self.latent_count < 2:
                return None, None
                
            variance = self.latent_m2 / (self.latent_count - 1)
            std = torch.sqrt(variance + 1e-8)
            return self.latent_means, std
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return None, None
    
    def training_step(self, latents=None, original_images=None):
        """Perform a single training step with improved loss calculation"""
        try:
            # Process input
            original_images = self._prepare_images(original_images, latents)
            
            # Initialize loss components
            total_loss = 0
            loss_dict = {}
            
            # Process in chunks with mixed precision
            batch_size = original_images.shape[0]
            chunk_size = min(batch_size, 2)
            
            for chunk in torch.split(original_images, chunk_size):
                with autocast(enabled=self.mixed_precision != "no"):
                    chunk_loss, chunk_dict = self._process_chunk(chunk)
                    total_loss += chunk_loss
                    
                    # Accumulate loss components
                    for k, v in chunk_dict.items():
                        loss_dict[k] = loss_dict.get(k, 0) + v
            
            # Update adaptive loss scale
            if self.adaptive_loss_scale:
                self._update_loss_scale(total_loss.item())
            
            # Average loss components
            loss_dict = {k: v / (batch_size / chunk_size) for k, v in loss_dict.items()}
            loss_dict["total_loss"] = total_loss.item() / (batch_size / chunk_size)
            
            return loss_dict
                
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            return {"total_loss": float('inf')}

    def _process_chunk(self, chunk):
        """Process a single chunk with improved loss calculation"""
        torch.cuda.empty_cache()
        
        # Encode
        posterior = self.vae.encode(chunk)
        latents = posterior.sample()
        
        # Apply channel normalization before statistics update
        if self.use_channel_scaling:
            latents = self.normalize_latents(latents)
            
        self.update_statistics(latents.detach())
        
        # Apply Welford statistics
        means, stds = self.get_statistics()
        if means is not None and stds is not None:
            latents = (latents - means[None,:,None,None]) / stds[None,:,None,None]
            decode_latents = latents * stds[None,:,None,None] + means[None,:,None,None]
        else:
            decode_latents = latents
            
        # Denormalize before decoding if channel scaling was applied
        if self.use_channel_scaling:
            decode_latents = self.denormalize_latents(decode_latents)
        
        # Decode with special focus on anime features
        decoded = self.vae.decode(decode_latents).sample
        
        # Calculate losses with emphasis on anime-specific features
        recon_loss = F.mse_loss(decoded, chunk, reduction="mean")
        
        # Additional loss weighting for eyes and textures
        # Extract eye regions using rough heuristic (upper third of image)
        _, _, H, W = decoded.shape
        eye_region = decoded[:, :, :H//3, :]
        eye_target = chunk[:, :, :H//3, :]
        eye_loss = F.mse_loss(eye_region, eye_target, reduction="mean")
        
        # Add texture loss using high-frequency components
        texture_loss = self._compute_texture_loss(decoded, chunk)
        
        # Combine losses with anime-specific weighting
        recon_loss = recon_loss + 2.0 * eye_loss + 0.5 * texture_loss
        
        kl_loss = self._compute_kl_loss(posterior)
        perceptual_loss = self._compute_perceptual_loss(decoded, chunk)
        
        # Apply SNR weighting
        snr_weight = self._compute_snr_weight(latents)
        
        # Combine losses
        total_loss = (
            recon_loss * snr_weight * self.scale_factor +
            kl_loss +
            perceptual_loss * self.perceptual_weight
        )
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return total_loss, {
            "recon_loss": recon_loss.item(),
            "eye_loss": eye_loss.item(),
            "texture_loss": texture_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "perceptual_loss": perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
            "snr_weight": snr_weight,
            "scale_factor": self.scale_factor
        }
        
    def _compute_texture_loss(self, x, y):
        """Compute texture loss using high-frequency components"""
        # Apply Sobel filters to detect edges/textures
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        
        # Compute gradients for both images
        x_grad_x = F.conv2d(x, sobel_x.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        x_grad_y = F.conv2d(x, sobel_y.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)
        y_grad_x = F.conv2d(y, sobel_x.expand(y.size(1), -1, -1, -1), groups=y.size(1), padding=1)
        y_grad_y = F.conv2d(y, sobel_y.expand(y.size(1), -1, -1, -1), groups=y.size(1), padding=1)
        
        # Compute texture loss as MSE of gradients
        return (F.mse_loss(x_grad_x, y_grad_x) + F.mse_loss(x_grad_y, y_grad_y)) / 2.0

    def _update_loss_scale(self, loss):
        """Update adaptive loss scale"""
        self.loss_history.append(loss)
        if len(self.loss_history) == self.loss_history.maxlen:
            current_std = torch.tensor(list(self.loss_history)).std().item()
            self.scale_factor = 1.0 / (1.0 + current_std)

    def _prepare_images(self, original_images, latents):
        """Prepare images for processing"""
        if original_images is None:
            if isinstance(latents, dict):
                original_images = latents["pixel_values"]
            else:
                raise ValueError("Either original_images or a batch dict must be provided")
        
        original_images = original_images.to(
            self.vae.device,
            dtype=next(self.vae.parameters()).dtype
        )
        
        # Ensure proper shape [B, C, H, W]
        if original_images.dim() == 3:
            original_images = original_images.unsqueeze(0)
        elif original_images.dim() > 4:
            original_images = original_images.squeeze(1)
            
        if original_images.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {original_images.shape}")
            
        return original_images
    
    def set_optimizer(self, optimizer, scheduler=None):
        """Set optimizer and optional scheduler for VAE finetuning"""
        self.optimizer = optimizer
        self.scheduler = scheduler

    def optimizer_step(self):
        """Perform optimizer and scheduler steps"""
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler:
                self.scheduler.step()

    def save_pretrained(self, path):
        """Save model and training state"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save VAE in diffusers format
            self.vae.save_pretrained(path, safe_serialization=True)
            
            # Save optimizer state
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
            
            # Save statistics and training state
            training_state = {
                "latent_means": self.latent_means.cpu() if self.latent_means is not None else None,
                "latent_m2": self.latent_m2.cpu() if self.latent_m2 is not None else None,
                "latent_count": self.latent_count,
                "loss_history": list(self.loss_history),
                "scale_factor": self.scale_factor
            }
            torch.save(training_state, os.path.join(path, "training_state.pt"))
            
            logger.info(f"Model and training state saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            logger.error(f"Save traceback: {traceback.format_exc()}")
            raise
