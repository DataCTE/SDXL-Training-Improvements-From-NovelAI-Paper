import torch
import logging
import traceback
import os
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit

logger = logging.getLogger(__name__)

class VAEFineTuner:
    def __init__(self, vae, learning_rate=1e-6):
        try:
            self.vae = vae
            self.optimizer = AdamW8bit(vae.parameters(), lr=learning_rate)
            
            # Convert VAE to bfloat16 and enable memory efficient attention
            self.vae = self.vae.to(dtype=torch.bfloat16)
            self.vae.enable_xformers_memory_efficient_attention()
            
            # Enable gradient checkpointing
            self.vae.enable_gradient_checkpointing()
            
            # Initialize Welford's online statistics
            self.latent_count = 0
            self.latent_means = None
            self.latent_m2 = None
            
            logger.info("VAEFineTuner initialized successfully")
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise
            
    def update_statistics(self, latents):
        """Update running mean and variance using Welford's online algorithm"""
        try:
            latents = latents.to(dtype=torch.bfloat16)  # Ensure bfloat16
            if self.latent_means is None:
                self.latent_means = torch.zeros(latents.size(1), device=latents.device, dtype=torch.bfloat16)
                self.latent_m2 = torch.zeros(latents.size(1), device=latents.device, dtype=torch.bfloat16)
                
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
            logger.error(f"Latents shape: {latents.shape}")
            logger.error(f"Current means shape: {self.latent_means.shape if self.latent_means is not None else None}")
            
    def get_statistics(self):
        try:
            if self.latent_count < 2:
                return None, None
                
            variance = self.latent_m2 / (self.latent_count - 1)
            std = torch.sqrt(variance + 1e-8)
            return self.latent_means, std
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            logger.error(f"Calculation traceback: {traceback.format_exc()}")
            return None, None
            
    def training_step(self, latents=None, original_images=None):
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Process input
                original_images = self._prepare_images(original_images, latents)
                
                # Process in chunks
                batch_size = original_images.shape[0]
                chunk_size = min(batch_size, 2)
                total_loss = 0
                
                for chunk in torch.split(original_images, chunk_size):
                    loss = self._process_chunk(chunk)
                    total_loss += loss
                
                return self._finalize_step(total_loss)
                
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            return {"total_loss": float('inf')}

    def _prepare_images(self, original_images, latents):
        """Prepare images for processing"""
        if original_images is None:
            if isinstance(latents, dict):
                original_images = latents["pixel_values"]
            else:
                raise ValueError("Either original_images or a batch dict must be provided")
        
        original_images = original_images.to(self.vae.device, dtype=torch.bfloat16)
        
        # Ensure proper shape [B, C, H, W]
        if original_images.dim() == 3:
            original_images = original_images.unsqueeze(0)
        elif original_images.dim() > 4:
            original_images = original_images.squeeze(1)
            
        if original_images.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {original_images.shape}")
            
        return original_images

    def _process_chunk(self, chunk):
        """Process a single chunk of images"""
        torch.cuda.empty_cache()
        
        # Encode
        latents = self.vae.encode(chunk).latent_dist.sample()
        self.update_statistics(latents.detach())
        
        # Apply statistics
        means, stds = self.get_statistics()
        if means is not None and stds is not None:
            latents = (latents - means[None,:,None,None]) / stds[None,:,None,None]
            decode_latents = latents * stds[None,:,None,None] + means[None,:,None,None]
        else:
            decode_latents = latents
        
        # Decode
        decoded = self.vae.decode(decode_latents).sample
        
        # Calculate loss
        loss = F.mse_loss(decoded, chunk, reduction="mean")
        loss.backward()
        
        return loss.item()

    def _finalize_step(self, total_loss):
        self.optimizer.step()
        
        return {
            "total_loss": total_loss,
            "latent_means": self.latent_means.detach().cpu() if self.latent_means is not None else None,
            "latent_stds": stds.detach().cpu() if stds is not None else None
        }

    def save_pretrained(self, path):
        """vae diffusers compatible save"""
        try:
            os.makedirs(path, exist_ok=True)
            self.vae.save_pretrained(path, safe_serialization=True)
            
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
            
            # Save statistics if available
            if self.latent_means is not None and self.latent_m2 is not None:
                stats = {
                    "means": self.latent_means.cpu(),
                    "m2": self.latent_m2.cpu(),
                    "count": self.latent_count
                }
                torch.save(stats, os.path.join(path, "latent_stats.pt"))
                
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            logger.error(f"Save traceback: {traceback.format_exc()}")
            raise
