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
import logging
import traceback
from typing import Dict, Tuple, Optional
from src.config.config import Config, VAEModelConfig

logger = logging.getLogger(__name__)

class VAEDiscriminator(nn.Module):
    """Patch-based discriminator for VAE training"""
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        try:
            super().__init__()
            
            if in_channels <= 0:
                raise ValueError(f"in_channels must be positive, got {in_channels}")
            if ndf <= 0:
                raise ValueError(f"ndf must be positive, got {ndf}")
            if n_layers <= 0:
                raise ValueError(f"n_layers must be positive, got {n_layers}")
            
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
            
        except Exception as e:
            logger.error(f"Error initializing VAEDiscriminator: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
        
    def _init_weights(self, m):
        try:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        except Exception as e:
            logger.error(f"Error initializing weights: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
            
    def forward(self, x):
        try:
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected input to be torch.Tensor, got {type(x)}")
            if x.dim() != 4:
                raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")
                
            features = self.model(x)
            return self.classifier(features)
        except Exception as e:
            logger.error(f"Error in discriminator forward pass: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

class VAETrainer:
    def __init__(
        self,
        config: Config,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        try:
            if not isinstance(config, Config):
                raise ValueError(f"Expected config to be Config, got {type(config)}")
                
            self.config = config
            self.accelerator = accelerator
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize counters
            self.current_epoch = 0
            self.global_step = 0
            
            # Initialize VAE
            try:
                self.vae = self._init_vae()
            except Exception as e:
                logger.error("Failed to initialize VAE")
                raise
            
            # Initialize discriminator if enabled
            self.discriminator = None
            if config.training.use_discriminator:
                try:
                    self.discriminator = VAEDiscriminator(
                        in_channels=config.model.vae.in_channels
                    ).to(self.device)
                except Exception as e:
                    logger.error("Failed to initialize discriminator")
                    raise
            
            # Initialize optimizers
            try:
                self.optimizer = self._configure_optimizers()
                if self.discriminator:
                    self.disc_optimizer = self._configure_discriminator_optimizer()
            except Exception as e:
                logger.error("Failed to configure optimizers")
                raise
                
            # Initialize scheduler
            try:
                self.lr_scheduler = self._configure_scheduler()
            except Exception as e:
                logger.error("Failed to configure scheduler")
                raise
            
            # Initialize LPIPS loss
            try:
                self.lpips = LPIPS(net='vgg').to(self.device)
                self.lpips.eval()
            except Exception as e:
                logger.error("Failed to initialize LPIPS loss")
                raise
            
            # Initialize memory buffers
            try:
                self._init_memory_buffers()
            except Exception as e:
                logger.error("Failed to initialize memory buffers")
                raise
            
            # Apply system configurations
            try:
                self._apply_system_config()
            except Exception as e:
                logger.error("Failed to apply system configurations")
                raise
            
            # Initialize mixed precision scaler
            self.scaler = torch.cuda.amp.GradScaler()
            
            # Resume from checkpoint if provided
            if resume_from_checkpoint:
                try:
                    self.load_checkpoint(resume_from_checkpoint)
                except Exception as e:
                    logger.error(f"Failed to load checkpoint from {resume_from_checkpoint}")
                    raise
            
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
            
        except Exception as e:
            logger.error(f"Error initializing VAETrainer: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _init_vae(self) -> AutoencoderKL:
        """Initialize VAE from config"""
        try:
            vae_config = self.config.model.vae
            if not hasattr(vae_config, 'pretrained_vae_name'):
                raise ValueError("VAE config missing pretrained_vae_name")
                
            return AutoencoderKL.from_pretrained(
                vae_config.pretrained_vae_name,
                torch_dtype=torch.float16 if self.config.training.mixed_precision == "fp16" else torch.bfloat16
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error initializing VAE: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _init_memory_buffers(self):
        """Initialize pinned memory buffers"""
        try:
            if not hasattr(self.config.data, 'batch_size'):
                raise ValueError("Config missing batch_size")
            if not hasattr(self.config.model.vae, 'latent_channels'):
                raise ValueError("VAE config missing latent_channels")
                
            self.latent_buffer = torch.zeros(
                (self.config.data.batch_size, self.config.model.vae.latent_channels, 64, 64),
                device=self.device,
                dtype=self.mixed_precision_dtype,
                pin_memory=True
            )
        except Exception as e:
            logger.error(f"Error initializing memory buffers: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _configure_optimizers(self):
        """Configure VAE optimizer"""
        try:
            if not hasattr(self.config.training, 'vae_learning_rate'):
                raise ValueError("Config missing vae_learning_rate")
                
            return torch.optim.AdamW(
                self.vae.parameters(),
                lr=self.config.training.vae_learning_rate,
                betas=self.config.training.optimizer_betas,
                eps=self.config.training.optimizer_eps,
                weight_decay=self.config.training.weight_decay
            )
        except Exception as e:
            logger.error(f"Error configuring VAE optimizer: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _configure_discriminator_optimizer(self):
        """Configure discriminator optimizer"""
        try:
            if not hasattr(self.config.training, 'discriminator_learning_rate'):
                raise ValueError("Config missing discriminator_learning_rate")
                
            return torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.training.discriminator_learning_rate,
                betas=self.config.training.optimizer_betas,
                eps=self.config.training.optimizer_eps,
                weight_decay=self.config.training.weight_decay
            )
        except Exception as e:
            logger.error(f"Error configuring discriminator optimizer: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _configure_scheduler(self):
        """Configure learning rate scheduler"""
        try:
            if not hasattr(self.config.training, 'num_epochs'):
                raise ValueError("Config missing num_epochs")
            if not hasattr(self.config.training, 'vae_min_lr'):
                raise ValueError("Config missing vae_min_lr")
                
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.vae_min_lr
            )
        except Exception as e:
            logger.error(f"Error configuring scheduler: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _apply_system_config(self):
        """Apply system-wide configurations"""
        try:
            if self.config.system.enable_xformers:
                try:
                    self.vae.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Failed to enable xformers: {str(e)}")
            
            if self.config.system.channels_last:
                try:
                    self.vae = self.vae.to(memory_format=torch.channels_last)
                except Exception as e:
                    logger.warning(f"Failed to set channels last memory format: {str(e)}")
            
            if self.config.system.gradient_checkpointing:
                try:
                    self.vae.enable_gradient_checkpointing()
                except Exception as e:
                    logger.warning(f"Failed to enable gradient checkpointing: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error applying system config: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Optional[Dict[str, torch.Tensor]] = None,
        tag_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute VAE training step"""
        try:
            if not isinstance(images, torch.Tensor):
                raise ValueError(f"Expected images to be torch.Tensor, got {type(images)}")
            if images.dim() != 4:
                raise ValueError(f"Expected 4D images (B,C,H,W), got {images.dim()}D")
            
            # Move images to device and normalize to [-1, 1]
            try:
                images = images.to(self.device, memory_format=torch.channels_last)
                images = 2 * images - 1
            except Exception as e:
                logger.error("Failed to preprocess images")
                raise
            
            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)
            if self.discriminator:
                self.disc_optimizer.zero_grad(set_to_none=True)
                
            # Split batch for gradient accumulation
            micro_batch_size = images.shape[0] // self.gradient_accumulation_steps
            
            total_loss = 0
            for i in range(self.gradient_accumulation_steps):
                try:
                    # Get micro batch
                    micro_images = images[i * micro_batch_size:(i + 1) * micro_batch_size]
                    
                    with torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype):
                        # Forward pass
                        try:
                            posterior = self.vae.encode(micro_images)
                            latents = posterior.sample()
                            decoded = self.vae.decode(latents)
                        except Exception as e:
                            logger.error(f"Error in VAE forward pass (batch {i})")
                            raise
                        
                        # Compute losses
                        try:
                            loss = self._compute_losses(decoded, micro_images, posterior)
                            loss = loss / self.gradient_accumulation_steps
                        except Exception as e:
                            logger.error(f"Error computing losses (batch {i})")
                            raise
                    
                    # Backward pass with gradient scaling
                    try:
                        self.scaler.scale(loss).backward()
                        total_loss += loss.item()
                    except Exception as e:
                        logger.error(f"Error in backward pass (batch {i})")
                        raise
                        
                except Exception as e:
                    logger.error(f"Error processing micro-batch {i}")
                    raise
                
            # Optimizer step after accumulation
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as e:
                logger.error("Failed optimizer step")
                raise
            
            # Update learning rate
            try:
                self.lr_scheduler.step()
            except Exception as e:
                logger.error("Failed learning rate update")
                raise
            
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
                try:
                    wandb.log(metrics, step=self.global_step)
                    
                    # Log sample images periodically
                    if self.global_step % (self.config.training.log_steps * 10) == 0:
                        self._log_sample_images(images, decoded)
                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {str(e)}")
                    
            return total_loss, metrics
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _train_discriminator(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train discriminator and compute adversarial losses"""
        try:
            if not isinstance(real_images, torch.Tensor):
                raise ValueError(f"Expected real_images to be torch.Tensor, got {type(real_images)}")
            if not isinstance(fake_images, torch.Tensor):
                raise ValueError(f"Expected fake_images to be torch.Tensor, got {type(fake_images)}")
            if real_images.shape != fake_images.shape:
                raise ValueError(f"Shape mismatch: real_images {real_images.shape} vs fake_images {fake_images.shape}")
                
            # Train with real images
            try:
                real_pred = self.discriminator(real_images)
                real_loss = F.binary_cross_entropy_with_logits(
                    real_pred, 
                    torch.ones_like(real_pred)
                )
            except Exception as e:
                logger.error("Error processing real images in discriminator")
                raise
                
            # Train with fake images
            try:
                fake_pred = self.discriminator(fake_images.detach())
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_pred,
                    torch.zeros_like(fake_pred)
                )
            except Exception as e:
                logger.error("Error processing fake images in discriminator")
                raise
                
            # Compute total discriminator loss
            disc_loss = (real_loss + fake_loss) / 2
            
            # Compute generator loss
            try:
                gen_pred = self.discriminator(fake_images)
                gen_loss = F.binary_cross_entropy_with_logits(
                    gen_pred,
                    torch.ones_like(gen_pred)
                )
            except Exception as e:
                logger.error("Error computing generator loss")
                raise
                
            return disc_loss, gen_loss
            
        except Exception as e:
            logger.error(f"Error in discriminator training: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _compute_losses(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
        posterior: torch.distributions.Distribution
    ) -> torch.Tensor:
        """Compute VAE losses including reconstruction, KL, and perceptual loss."""
        try:
            if not isinstance(decoded, torch.Tensor):
                raise ValueError(f"Expected decoded to be torch.Tensor, got {type(decoded)}")
            if not isinstance(target, torch.Tensor):
                raise ValueError(f"Expected target to be torch.Tensor, got {type(target)}")
                
            losses = {}
            
            # Reconstruction loss
            try:
                losses['recon'] = F.mse_loss(decoded, target, reduction='mean')
            except Exception as e:
                logger.error("Error computing reconstruction loss")
                raise
            
            # KL divergence loss
            try:
                losses['kl'] = posterior.kl().mean()
            except Exception as e:
                logger.error("Error computing KL divergence")
                raise
            
            # LPIPS perceptual loss
            try:
                with torch.no_grad():
                    losses['lpips'] = self.lpips(decoded, target).mean()
            except Exception as e:
                logger.error("Error computing LPIPS loss")
                raise
            
            # Compute total loss with weights
            total_loss = (
                self.config.training.recon_weight * losses['recon'] +
                self.config.training.kl_weight * losses['kl'] +
                self.config.training.perceptual_weight * losses['lpips']
            )
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing losses: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _log_sample_images(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """Log sample images to wandb for visualization."""
        try:
            if not isinstance(original, torch.Tensor):
                raise ValueError(f"Expected original to be torch.Tensor, got {type(original)}")
            if not isinstance(reconstructed, torch.Tensor):
                raise ValueError(f"Expected reconstructed to be torch.Tensor, got {type(reconstructed)}")
                
            # Denormalize images
            try:
                original = (original + 1) / 2
                reconstructed = (reconstructed + 1) / 2
            except Exception as e:
                logger.error("Error denormalizing images")
                raise
            
            # Create grid
            try:
                n_samples = min(4, original.shape[0])
                comparison = torch.cat([
                    original[:n_samples],
                    reconstructed[:n_samples]
                ])
                grid = torchvision.utils.make_grid(
                    comparison,
                    nrow=n_samples,
                    normalize=True,
                    value_range=(0, 1)
                )
            except Exception as e:
                logger.error("Error creating image grid")
                raise
            
            # Log to wandb
            try:
                wandb.log({
                    "samples": wandb.Image(grid.cpu())
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log images to wandb: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error logging sample images: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Don't raise here since this is not critical for training

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        try:
            if not isinstance(path, str):
                raise ValueError(f"Expected path to be str, got {type(path)}")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare checkpoint
            checkpoint = {
                'model': self.vae.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'step': self.global_step,
                'epoch': self.current_epoch,
            }
            
            if self.discriminator:
                checkpoint.update({
                    'discriminator': self.discriminator.state_dict(),
                    'disc_optimizer': self.disc_optimizer.state_dict()
                })
            
            # Save atomically
            tmp_path = path + ".tmp"
            try:
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, path)
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
                
            logger.info(f"Saved checkpoint to {path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        try:
            if not isinstance(path, str):
                raise ValueError(f"Expected path to be str, got {type(path)}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
                
            # Load checkpoint
            try:
                checkpoint = torch.load(path, map_location=self.device)
            except Exception as e:
                logger.error(f"Failed to load checkpoint file: {str(e)}")
                raise
                
            # Load model state
            try:
                self.vae.load_state_dict(checkpoint['model'])
            except Exception as e:
                logger.error("Failed to load model state")
                raise
                
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                logger.error("Failed to load optimizer state")
                raise
                
            # Load scheduler state
            try:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                logger.error("Failed to load scheduler state")
                raise
                
            # Load scaler state
            try:
                self.scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                logger.error("Failed to load scaler state")
                raise
                
            # Load discriminator if present
            if self.discriminator and 'discriminator' in checkpoint:
                try:
                    self.discriminator.load_state_dict(checkpoint['discriminator'])
                    self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
                except Exception as e:
                    logger.error("Failed to load discriminator state")
                    raise
                    
            # Load training state
            self.global_step = checkpoint['step']
            self.current_epoch = checkpoint['epoch']
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper method to register buffers"""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor

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