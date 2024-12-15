# src/training/vae_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL
from lpips import LPIPS
import torchvision
import wandb
import os
from typing import Dict, Tuple, Optional
from src.config.config import Config, VAEModelConfig

class VAEDiscriminator(nn.Module):
    """Patch-based discriminator for VAE training"""
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        ]
        
        for i in range(n_layers - 1):
            in_filters = ndf * min(2**i, 8)
            out_filters = ndf * min(2**(i+1), 8)
            layers.extend([
                nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, True)
            ])
            
        self.model = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(out_filters, 1, 4, 1, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

class VAETrainer:
    def __init__(
        self,
        config: Config,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.config = config
        self.accelerator = accelerator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize counters
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize VAE
        self.vae = self._init_vae()
        
        # Initialize discriminator if enabled
        self.discriminator = None
        if config.training.use_discriminator:
            self.discriminator = VAEDiscriminator(
                in_channels=config.model.vae.in_channels
            ).to(self.device)
        
        # Initialize optimizers
        self.optimizer = self._configure_optimizers()
        if self.discriminator:
            self.disc_optimizer = self._configure_discriminator_optimizer()
            
        # Initialize scheduler
        self.lr_scheduler = self._configure_scheduler()
        
        # Initialize LPIPS loss
        self.lpips = LPIPS(net='vgg').to(self.device)
        self.lpips.eval()
        
        # Initialize memory buffers
        self._init_memory_buffers()
        
        # Apply system configurations
        self._apply_system_config()
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Add memory optimizations
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        self.mixed_precision_dtype = torch.float16 if config.training.mixed_precision == "fp16" else torch.bfloat16
        
        # Enable memory efficient attention
        if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
            self.vae.enable_xformers_memory_efficient_attention()
        
        # Enable gradient checkpointing
        if config.system.gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
            
        # Use channels last memory format
        self.vae = self.vae.to(memory_format=torch.channels_last)

    def _init_vae(self) -> AutoencoderKL:
        """Initialize VAE from config"""
        vae_config = self.config.model.vae
        return AutoencoderKL.from_pretrained(
            vae_config.pretrained_vae_name,
            torch_dtype=torch.float16 if self.config.training.mixed_precision == "fp16" else torch.bfloat16
        ).to(self.device)

    def _init_memory_buffers(self):
        """Initialize pinned memory buffers"""
        self.latent_buffer = torch.zeros(
            (self.config.data.batch_size, self.config.model.vae.latent_channels, 64, 64),
            device=self.device,
            dtype=self.mixed_precision_dtype,
            pin_memory=True
        )

    def _configure_optimizers(self):
        """Configure VAE optimizer"""
        return torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.config.training.vae_learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps,
            weight_decay=self.config.training.weight_decay
        )

    def _configure_discriminator_optimizer(self):
        """Configure discriminator optimizer"""
        return torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.training.discriminator_learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps,
            weight_decay=self.config.training.weight_decay
        )

    def _configure_scheduler(self):
        """Configure learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
            eta_min=self.config.training.vae_min_lr
        )

    def _apply_system_config(self):
        """Apply system-wide configurations"""
        if self.config.system.enable_xformers:
            self.vae.enable_xformers_memory_efficient_attention()
        
        if self.config.system.channels_last:
            self.vae = self.vae.to(memory_format=torch.channels_last)
        
        if self.config.system.gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Optional[Dict[str, torch.Tensor]] = None,
        tag_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute VAE training step"""
        
        # Move images to device and normalize to [-1, 1]
        images = images.to(self.device, memory_format=torch.channels_last)
        images = 2 * images - 1
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        if self.discriminator:
            self.disc_optimizer.zero_grad(set_to_none=True)
            
        # Split batch for gradient accumulation
        micro_batch_size = images.shape[0] // self.gradient_accumulation_steps
        
        total_loss = 0
        for i in range(self.gradient_accumulation_steps):
            # Get micro batch
            micro_images = images[i * micro_batch_size:(i + 1) * micro_batch_size]
            
            with torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype):
                # Forward pass
                posterior = self.vae.encode(micro_images)
                latents = posterior.sample()
                
                # Use non-blocking tensor operations
                decoded = self.vae.decode(latents)
                
                # Compute losses
                loss = self._compute_losses(decoded, micro_images, posterior)
                loss = loss / self.gradient_accumulation_steps
                
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            
        # Optimizer step after accumulation
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Log metrics
        metrics = {
            'loss/recon': total_loss,
            'loss/kl': 0,
            'loss/lpips': 0,
            'loss/total': total_loss,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }
        
        if self.discriminator:
            metrics.update({
                'loss/disc': 0,
                'loss/gen': 0
            })
            
        # Log to wandb
        if wandb.run is not None and self.global_step % self.config.training.log_steps == 0:
            wandb.log(metrics, step=self.global_step)
            
            # Log sample images periodically
            if self.global_step % (self.config.training.log_steps * 10) == 0:
                self._log_sample_images(images, decoded)
                
        return total_loss, metrics

    def _train_discriminator(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train discriminator and compute adversarial losses"""
        
        with torch.cuda.amp.autocast():
            # Train discriminator
            real_pred = self.discriminator(real_images)
            fake_pred = self.discriminator(fake_images.detach())
            
            d_loss = (
                F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
                F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            )
            
            # Generator loss
            fake_pred = self.discriminator(fake_images)
            g_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
            
        # Backward pass for discriminator
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.disc_optimizer)
        
        return d_loss, g_loss

    def _log_sample_images(self, real_images: torch.Tensor, reconstructions: torch.Tensor):
        """Log sample images to wandb"""
        n_samples = min(4, real_images.size(0))
        real_samples = real_images[:n_samples]
        recon_samples = reconstructions[:n_samples]
        
        # Denormalize images
        real_samples = (real_samples + 1) / 2
        recon_samples = (recon_samples + 1) / 2
        
        # Create grid
        grid = torch.cat([real_samples, recon_samples], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=n_samples)
        
        wandb.log({
            "samples": wandb.Image(
                grid.cpu(),
                caption=f"Top: Real, Bottom: Reconstructed (Step {self.global_step})"
            )
        })

    def save_checkpoint(self, path: str):
        """Save checkpoint efficiently"""
        # Save in fp16 for smaller file size
        state_dict = self.vae.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.half()
            
        torch.save({
            'model': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'step': self.global_step
        }, path)

    def load_checkpoint(self, checkpoint_dir: str):
        """Load VAE checkpoint"""
        # Load VAE weights
        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(checkpoint_dir, "vae"),
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load discriminator if present
        if self.discriminator:
            disc_path = os.path.join(checkpoint_dir, "discriminator.pt")
            if os.path.exists(disc_path):
                self.discriminator.load_state_dict(torch.load(disc_path))
                
        # Load training state
        training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"))
        
        self.current_epoch = training_state["epoch"]
        self.global_step = training_state["global_step"]
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.lr_scheduler.load_state_dict(training_state["scheduler_state"])
        self.scaler.load_state_dict(training_state["scaler_state"])
        
        if self.discriminator and "disc_optimizer_state" in training_state:
            self.disc_optimizer.load_state_dict(training_state["disc_optimizer_state"])
            
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper method to register buffers"""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor

    def _compute_losses(self, decoded, images, posterior):
        # Fuse operations where possible
        recon_loss = F.mse_loss(decoded, images, reduction='sum')
        kl_loss = posterior.kl().sum()
        
        # Compute LPIPS more efficiently
        with torch.no_grad():
            lpips_loss = self.lpips(decoded, images).sum()
            
        return (
            recon_loss + 
            self.config.model.vae.kl_weight * kl_loss +
            self.config.model.vae.lpips_weight * lpips_loss
        ) / images.shape[0]

    def _setup_distributed(self):
        """Setup distributed training"""
        if torch.cuda.device_count() > 1:
            self.vae = nn.parallel.DistributedDataParallel(
                self.vae,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False
            )