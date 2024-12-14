import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import wandb
import os
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from ..data.dataset import NovelAIDataset
from ..data.sampler import AspectBatchSampler

class NovelAIDiffusionV3Trainer(nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        gradient_accumulation_steps: int = 4,
    ):
        super().__init__()
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
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
        self.sigma_min = 0.002
        self.sigma_max = 20000.0
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Enable memory efficient attention
        model.enable_xformers_memory_efficient_attention()
        
        # Enable channels last memory format
        model = model.to(memory_format=torch.channels_last)
        
        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Disable debug APIs
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    def get_sigmas(self) -> torch.Tensor:
        """Generate noise schedule for ZTSNR with optimized scaling"""
        ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        
        min_inv_rho = self.sigma_min ** (1/self.rho)
        max_inv_rho = self.sigma_max ** (1/self.rho)
        
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        
        sigmas[0] = self.sigma_min
        sigmas[-1] = self.sigma_max
        
        return sigmas

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        return (self.sigma_data / sigma).square()

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]
        
        snr = alpha_t / (1 - alpha_t)
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        return torch.minimum(snr, min_snr).float()

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized time_ids computation"""
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

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        images = images.to(self.device)
        text_embeds = {k: v.to(self.device) for k,v in text_embeds.items()}
        tag_weights = tag_weights.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        total_loss = 0.0
        total_v_pred = None
        running_loss = 0.0

        for i in range(self.gradient_accumulation_steps):
            torch.cuda.empty_cache()

            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size

            # Extract micro-batch
            batch_latents = images[start_idx:end_idx]

            # Apply area-based noise scaling
            height = batch_latents.shape[2]
            width = batch_latents.shape[3]
            area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
            noise_scale = torch.sqrt(area / self.base_area)
            batch_latents = batch_latents * noise_scale

            batch_tag_weights = tag_weights[start_idx:end_idx].view(-1,1,1,1)

            # Prepare text embeddings
            base_hidden = text_embeds["base_text_embeds"][start_idx:end_idx].squeeze(1).clone()
            base_pooled = text_embeds["base_pooled_embeds"][start_idx:end_idx].squeeze(1).clone()

            # Project text embeddings
            base_hidden_float32 = base_hidden.to(dtype=torch.float32)
            batch_size, seq_len, _ = base_hidden_float32.shape
            encoder_hidden_states = self.hidden_proj(
                base_hidden_float32.view(-1, 768)
            ).view(batch_size, seq_len, -1)

            # Sample noise and timesteps
            noise = torch.randn_like(batch_latents)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_latents.size(0),), 
                device=self.device
            ).long()

            sigmas = self.sigmas[timesteps]

            # Add noise
            noisy_latents = batch_latents + sigmas.view(-1,1,1,1)*noise

            # Compute v_target
            v_target = self.scheduler.get_velocity(batch_latents, noise, timesteps)

            # Generate time_ids
            time_ids = self._get_add_time_ids(batch_latents).clone()

            # Karras scaling
            c_skip, c_out, c_in = self.get_karras_scalings(sigmas)

            # Ensure input tensors have gradients enabled
            scaled_input = (c_in.view(-1,1,1,1)*noisy_latents).clone().requires_grad_(True)
            encoder_hidden_states = encoder_hidden_states.clone().requires_grad_(True)
            timesteps = timesteps.clone()

            # Forward pass with autocast
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                F_out = self.model(
                    scaled_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": base_pooled.clone(),
                        "time_ids": time_ids
                    }
                ).sample

                D_out = c_skip.view(-1,1,1,1)*noisy_latents + c_out.view(-1,1,1,1)*F_out

                loss_per_sample = F.mse_loss(D_out.float(), v_target.float(), reduction='none')
                loss_per_sample = loss_per_sample.mean(dim=[1,2,3])

                snr_weights = self.get_minsnr_weights(timesteps)

                loss_per_sample = loss_per_sample * batch_tag_weights.squeeze() * snr_weights

                loss = loss_per_sample.mean() / self.gradient_accumulation_steps

            loss.backward()
            
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value

            total_v_pred = D_out.detach() if total_v_pred is None else torch.cat([total_v_pred, D_out.detach()], dim=0)

        avg_loss = running_loss / self.gradient_accumulation_steps
        return total_loss, batch_latents, total_v_pred, timesteps, avg_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> float:
        self.current_epoch = epoch
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            
            images, text_embeds, tag_weights = batch
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, pred_images, v_pred, timesteps, avg_batch_loss = self.training_step(
                    images, text_embeds, tag_weights
                )
            
            grad_norm = self.compute_grad_norm()
            self.optimizer.step()
            
            if self.accelerator.is_main_process:
                wandb.log({
                    'grad/norm': grad_norm,
                    'loss/batch': avg_batch_loss,
                    'loss/running_avg': total_loss / (batch_idx + 1),
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                })
            
            total_loss += avg_batch_loss
            
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch {epoch} [{batch_idx}/{num_batches}]:')
                print(f'  Batch Loss = {avg_batch_loss:.4f}')
                print(f'  Avg Loss = {avg_loss:.4f}')
                print(f'  Grad norm = {grad_norm:.4f}')
                print(f'  Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB')
            
            self.global_step += 1
        
        return total_loss / num_batches

    def compute_grad_norm(self):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def save_checkpoint(self, save_path: str):
        """Save model checkpoint with only modified components in fp16 format"""
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        os.makedirs(save_path, exist_ok=True)

        # Temporarily convert model to float16 for saving
        original_dtype = unwrapped_model.dtype
        unwrapped_model = unwrapped_model.to(torch.float16)

        try:
            unwrapped_model.save_pretrained(
                os.path.join(save_path, "unet"),
                safe_serialization=True
            )
        finally:
            unwrapped_model = unwrapped_model.to(original_dtype)

        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    def load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            print("No training state found, starting from scratch with pretrained weights")

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ) -> DataLoader:
        sampler = AspectBatchSampler(dataset, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )