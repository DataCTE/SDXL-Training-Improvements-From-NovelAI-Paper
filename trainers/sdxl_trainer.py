import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Any

from .base_trainer import BaseTrainer

class SDXLTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        config: Any,
        accelerator = None,
        checkpoint_manager = None
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            config=config,
            accelerator=accelerator,
            checkpoint_manager=checkpoint_manager
        )
        
        self.vae = vae
        self.scheduler = scheduler
        
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
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho
        self.num_timesteps = config.num_timesteps
        self.min_snr_gamma = config.min_snr_gamma
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())

    def get_sigmas(self) -> torch.Tensor:
        """Generate noise schedule with ZTSNR"""
        ramp = torch.linspace(0, 1, self.num_timesteps)
        min_inv_rho = self.sigma_min ** (1/self.rho)
        max_inv_rho = self.sigma_max ** (1/self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas.to(self.device)

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        """Calculate signal-to-noise ratio"""
        return (self.sigma_data / sigma) ** 2

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Calculate MinSNR loss weights"""
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]
        snr = alpha_t / (1 - alpha_t)
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr, min_snr)
        return weights.float()

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings"""
        batch_size = images.shape[0]
        orig_height = images.shape[2] * 8
        orig_width = images.shape[3] * 8
        
        add_time_ids = torch.empty(
            (batch_size, 2, 4), 
            device=self.device,
            dtype=torch.bfloat16
        )
        add_time_ids[:, 0, 0] = orig_height
        add_time_ids[:, 0, 1] = orig_width
        add_time_ids[:, 0, 2] = self.aesthetic_score
        add_time_ids[:, 0, 3] = self.zero_score
        add_time_ids[:, 1, 0] = orig_height
        add_time_ids[:, 1, 1] = orig_width
        add_time_ids[:, 1, 2] = self.crop_score
        add_time_ids[:, 1, 3] = self.zero_score
        
        return add_time_ids.reshape(batch_size, -1)

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute single training step"""
        # Process text embeddings
        base_hidden = text_embeds["base_text_embeds"].to(
            device=self.device,
            dtype=torch.bfloat16,
            non_blocking=True
        ).squeeze(1)
        
        base_pooled = text_embeds["base_pooled_embeds"].to(
            device=self.device,
            dtype=torch.bfloat16,
            non_blocking=True
        ).squeeze(1)
        
        # Process images
        images = images.to(
            device=self.device,
            dtype=torch.bfloat16,
            non_blocking=True,
            memory_format=torch.channels_last
        )
        
        # Project text embeddings
        batch_size, seq_len, _ = base_hidden.shape
        base_hidden_float32 = base_hidden.to(dtype=torch.float32)
        encoder_hidden_states = self.hidden_proj(
            base_hidden_float32.view(-1, 768)
        ).view(batch_size, seq_len, -1)
        
        # Setup time embeddings
        add_time_ids = self._get_add_time_ids(images)
        added_cond_kwargs = {
            "text_embeds": base_pooled,
            "time_ids": add_time_ids
        }
        
        # Calculate noise and scaling
        height, width = images.shape[2:]
        area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
        noise_scale = torch.sqrt(area / self.base_area)
        
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=self.device
        )
        sigma = self.sigmas[timesteps] * noise_scale
        
        # Calculate scaling factors
        sigma_squared = sigma * sigma
        sigma_data_squared = self.sigma_data * self.sigma_data
        denominator = sigma_squared + sigma_data_squared
        denominator_sqrt = denominator.sqrt()
        
        c_skip = sigma_data_squared / denominator
        c_out = -sigma * self.sigma_data / denominator_sqrt
        c_in = 1 / denominator_sqrt
        
        # Generate noise and noisy images
        noise = torch.randn_like(
            images,
            dtype=torch.float32,
            device=self.device,
            memory_format=torch.channels_last
        )
        noisy_images = images + noise * sigma.view(-1, 1, 1, 1)
        scaled_noisy_images = noisy_images * c_in.view(-1, 1, 1, 1)
        
        # Model forward pass
        v_prediction = self.model(
            scaled_noisy_images,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Calculate loss
        pred_images = (
            noisy_images * c_skip.view(-1, 1, 1, 1) + 
            v_prediction * c_out.view(-1, 1, 1, 1)
        )
        loss = torch.nn.functional.mse_loss(pred_images, images, reduction='none')
        
        # Apply weights
        tag_weights = tag_weights.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True
        )
        timestep_weights = self.get_minsnr_weights(timesteps)
        combined_weights = (
            tag_weights.view(-1, 1, 1, 1) *
            timestep_weights.view(-1, 1, 1, 1)
        )
        loss = (loss * combined_weights).mean()
        
        return loss, pred_images, v_prediction, timesteps

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                # Clear cache at start of batch
                torch.cuda.empty_cache()
                
                # Unpack batch
                images, text_embeds, tag_weights = batch
                
                # Training step
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, pred_images, v_pred, timesteps = self.training_step(
                        images, text_embeds, tag_weights
                    )
                
                # Backward pass
                loss.backward()
                
                # Log metrics
                grad_norm = self.compute_grad_norm()
                self.log_metrics({
                    'loss/total': loss.item(),
                    'v_pred/mean': v_pred.mean().item(),
                    'v_pred/std': v_pred.std().item(),
                    'v_pred/min': v_pred.min().item(),
                    'v_pred/max': v_pred.max().item(),
                    'grad/norm': grad_norm,
                    'timesteps/mean': timesteps.float().mean().item(),
                    'timesteps/std': timesteps.float().std().item()
                })
                
                self.optimizer.step()
                
                total_loss += loss.item()
                self.global_step += 1
                
                # Console logging
                if batch_idx % self.config.log_interval == 0:
                    print(f'Epoch {self.current_epoch} [{batch_idx}/{num_batches}]:')
                    print(f'  Loss = {loss.item():.4f}')
                    print(f'  Grad norm = {grad_norm:.4f}')
                    print(f'  Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB')
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Memory stats:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                raise
                
        return total_loss / num_batches 