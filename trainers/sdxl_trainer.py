import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from memory.Manager import MemoryManager
from data.dataset import NovelAIDataset
from data.sampler import AspectBatchSampler
from accelerate import Accelerator
import time
import os
import wandb
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from .training_utils import TrainingProfiler, AutoTuner
from configs.training_config import TrainingConfig
from .progress import TrainProgress
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from memory.coordinator import MemoryCoordinator
from memory.layeroffloading import LayerOffloadStrategy, StaticActivationAllocator


class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        batch_size: int = 1,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        max_vram_usage: float = 0.8,
        gradient_accumulation_steps: int = 4,
        config: Optional[TrainingConfig] = None
    ):
        super().__init__()
        
        # Create default accelerator if none provided
        if accelerator is None:
            self.accelerator = Accelerator()
        else:
            self.accelerator = accelerator
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # SDXL-specific optimizations
        self.use_xformers = False
        try:
            import xformers
            import xformers.ops
            self.use_xformers = True
            print("Using xformers for improved attention efficiency")
        except ImportError:
            print("xformers not available, falling back to default attention")
            
        # Memory optimization flags
        self.use_gradient_checkpointing = True
        
        self.model = model
        # Apply memory optimizations to model
        if self.use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            
        # VAE setup
        self.vae = vae
        # Move VAE to CPU if low on VRAM
        if max_vram_usage < 0.6:
            self.vae = self.vae.to('cpu')
            print("Moving VAE to CPU to save VRAM")
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize memory management components
        self.memory_coordinator = MemoryCoordinator(
            max_vram_usage=max_vram_usage,
            enable_quantization=True,
            enable_efficient_attention=True,
            device=device,
            strategy=LayerOffloadStrategy.DYNAMIC if max_vram_usage < 0.7 else LayerOffloadStrategy.STATIC
        )
        
        # Setup model with memory coordinator
        self.memory_coordinator.setup_model(self.model)
        
        # Get layer conductor from coordinator
        self.layer_conductor = self.memory_coordinator.layer_conductor
        
        # Initialize activation allocator with proper device placement
        self.activation_allocator = StaticActivationAllocator(self.model)
        self.activation_allocator.allocate_buffers(device)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(
            enabled=True,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100
        )
        
        # Setup profiler and auto-tuner
        self.profiler = TrainingProfiler()
        self.auto_tuner = AutoTuner(
            initial_params={
                'batch_size': batch_size,
                'grad_accum': gradient_accumulation_steps
            },
            config=config,
            device=device
        )
        
        # Initialize progress tracking
        self.progress = TrainProgress()
        self.current_step = 0
        self.batch_size = batch_size

        # Load checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Initialize memory manager (existing code)
        self.memory_manager = MemoryManager(max_vram_usage=max_vram_usage)
        self.memory_manager.setup_memory_management(model, device)
        
        # Initialize profiler and auto-tuner
        self.profiler = TrainingProfiler(
            window_size=50,
            config=config,
            device=self.device
        )
        self.auto_tuner = AutoTuner(
            initial_params={
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            config=config,
            min_samples=50,
            device=self.device
        )
        
        # Add memory monitoring to profiler
        if hasattr(self.memory_manager, 'get_current_usage'):
            self.profiler.add_memory_callback(self.memory_manager.get_current_usage)
        else:
            print("Warning: Memory manager does not support current usage tracking")
        
        # Add projection layer for CLIP embeddings with efficient memory layout
        self.hidden_proj = nn.Linear(768, model.config.cross_attention_dim).to(
            device=device, 
            dtype=torch.bfloat16  # Use bfloat16 for better training stability
        )
        
        # Pre-allocate tensors for time embeddings with optimal dtype
        self.register_buffer('base_area', torch.tensor(1024 * 1024, dtype=torch.float32))
        self.register_buffer('aesthetic_score', torch.tensor(6.0, dtype=torch.bfloat16))
        self.register_buffer('crop_score', torch.tensor(3.0, dtype=torch.bfloat16))
        self.register_buffer('zero_score', torch.tensor(0.0, dtype=torch.bfloat16))
        
        # Update ZTSNR parameters for SDXL
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = 20000.0
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas with optimal memory layout
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Replace individual progress tracking with TrainProgress
        self.progress = TrainProgress()
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        # Initialize mixed precision scaler
        self.scaler = GradScaler()
        self.warmup_steps = 100
        
        # Centralize CUDA settings
        self._setup_cuda_optimizations()
        
    def _setup_cuda_optimizations(self):
        if not torch.cuda.is_available():
            return
        
        device_cap = torch.cuda.get_device_capability(self.device)
        
        # Enable TF32 for Ampere+ GPUs
        if device_cap[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable flash attention if supported
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
    def _setup_vae(self):
        """Setup VAE with memory-efficient configuration"""
        if self.memory_coordinator.should_offload_vae():
            self.vae = self.vae.to('cpu')
            self.use_cpu_vae = True
            print("Moving VAE to CPU to optimize memory usage")
        else:
            self.vae = self.vae.to(self.device)
            self.use_cpu_vae = False
            print("Keeping VAE on GPU for faster processing")
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        added_cond_kwargs: Dict = None,
        current_layer: str = None,
        phase: str = 'forward'
    ) -> torch.Tensor:
        """Forward pass with memory-efficient layer handling"""
        # Handle layer memory management if needed
        if current_layer:
            self.memory_manager.before_layer_computation(current_layer, phase)
        
        try:
            # Clear cache before forward pass
            torch.cuda.empty_cache()
            
            # Call UNet's forward pass with gradient checkpointing
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output = self.model(
                    sample=x,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
            
            # Clear unnecessary tensors
            del x
            torch.cuda.empty_cache()
            
            return output
            
        finally:
            if current_layer:
                self.memory_manager.after_layer_computation(current_layer)
                torch.cuda.empty_cache()

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized training step with improved memory management and performance"""
        
        # Pre-allocate device tensors with non-blocking transfer
        images = images.to(self.device, non_blocking=True)
        text_embeds = {k: v.to(self.device, non_blocking=True) for k, v in text_embeds.items()}
        tag_weights = tag_weights.to(self.device, non_blocking=True)

        batch_size = images.shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps

        # Pre-allocate accumulation tensors
        total_loss = 0.0
        total_v_pred = None

        # Zero gradients with optimal memory usage
        self.optimizer.zero_grad(set_to_none=True)

        # Activate required layer groups
        self.memory_coordinator.activate_layer_group("encoder")

        for i in range(self.gradient_accumulation_steps):
            # Get micro-batch slice indices
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size

            # Process micro-batch with optimized memory handling
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Extract micro-batch with contiguous memory
                batch_images = images[start_idx:end_idx].contiguous()
                batch_text_embeds = {k: v[start_idx:end_idx].contiguous() 
                                   for k, v in text_embeds.items()}
                batch_tag_weights = tag_weights[start_idx:end_idx].contiguous()

                # Reshape batch_images maintaining exact format
                batch_images = batch_images.squeeze(1)[:, :3]  # [B, 3, H, W]

                # Compute area scaling factor once per batch
                area = torch.tensor(
                    batch_images.shape[2] * batch_images.shape[3],
                    device=self.device, dtype=torch.float32
                )
                noise_scale = (area / self.base_area).sqrt()

                # VAE encoding with memory optimization
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    batch_latents = self.vae.encode(batch_images).latent_dist.sample()
                    batch_latents = batch_latents * self.vae.config.scaling_factor * noise_scale
                    del batch_images
                
                # Generate noise and timesteps efficiently
                noise = torch.randn_like(batch_latents, device=self.device)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (micro_batch_size,), device=self.device
                )
                
                # Get preconditioner factors
                sigmas = self.sigmas[timesteps]
                c_skip, c_out, c_in = self.get_karras_scalings(sigmas)
                
                # Compute noisy latents and scale input
                noisy_latents = batch_latents + sigmas.view(-1, 1, 1, 1) * noise
                scaled_input = c_in.view(-1, 1, 1, 1) * noisy_latents

                # Process embeddings with optimal memory layout
                base_hidden = batch_text_embeds["base_text_embeds"].squeeze(1)  # [B, 77, 768]
                base_pooled = batch_text_embeds["base_pooled_embeds"].squeeze(1)  # [B, 768]
                
                # Project embeddings efficiently
                encoder_hidden_states = self.hidden_proj(
                    base_hidden.reshape(-1, 768)
                ).view(micro_batch_size, base_hidden.shape[1], -1)

                # Get time embeddings
                time_ids = self._get_add_time_ids(batch_latents)
                
                # Prepare condition kwargs
                added_cond_kwargs = {
                    "text_embeds": base_pooled,
                    "time_ids": time_ids
                }

                # Forward pass with optimized memory management
                v_prediction = self._forward_with_cache(
                    scaled_input, timesteps, encoder_hidden_states, 
                    added_cond_kwargs
                )

                # Compute model prediction
                model_pred = (
                    c_skip.view(-1, 1, 1, 1) * noisy_latents + 
                    c_out.view(-1, 1, 1, 1) * v_prediction
                )

                # Get target and compute loss
                v_pred_target = self.scheduler.get_velocity(batch_latents, noise, timesteps)
                
                # Compute loss with tag weights
                batch_tag_weights = batch_tag_weights.view(-1, 1, 1, 1)
                loss = F.mse_loss(
                    model_pred.float(), 
                    v_pred_target.float(), 
                    reduction="none"
                )
                loss = (loss.mean(dim=[1,2,3]) * batch_tag_weights.squeeze()).mean()
                loss = loss / self.gradient_accumulation_steps

            # Scale loss and backward
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            # Accumulate predictions if needed
            if total_v_pred is None:
                total_v_pred = model_pred.detach()
            else:
                total_v_pred = torch.cat([total_v_pred, model_pred.detach()], dim=0)

            # Clear cache between micro-batches
            torch.cuda.empty_cache()

        # Update weights with gradient scaling
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Deactivate when done
        self.memory_coordinator.deactivate_layer_group("encoder")

        return total_loss, total_v_pred, grad_norm, timesteps

    def _forward_with_cache(self, x, timesteps, encoder_hidden_states, added_cond_kwargs):
        """Optimized forward pass with activation checkpointing"""
        v_prediction = None
        
        # Get model layers more efficiently
        model_layers = [(name, module) for name, module in self.model.named_modules() 
                       if list(module.parameters())]
        
        try:
            for layer_idx, (name, layer) in enumerate(model_layers):
                # Store previous activation if exists
                if v_prediction is not None:
                    self.activation_allocator.store_activation(name, v_prediction)
                
                # Layer computation with memory management
                self.layer_conductor.before_layer(layer_idx, name)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    v_prediction = self.forward(
                        x if layer_idx == 0 else v_prediction,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        current_layer=name
                    )
                
                self.layer_conductor.after_layer(layer_idx, name)
                
            return v_prediction
            

            
        finally:
            # Clean up stored activations
            self.activation_allocator.clear()

    def load_checkpoint(self, path: str):
        """Load checkpoint with progress information"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore progress state
        if 'progress' in checkpoint:
            self.progress = checkpoint['progress']
        
        # Restore profiler and auto-tuner states if available
        if 'profiler_state' in checkpoint:
            self.profiler.metrics = checkpoint['profiler_state']
        if 'auto_tuner_state' in checkpoint:
            self.auto_tuner.current_params = checkpoint['auto_tuner_state']['current_params']
            self.auto_tuner.best_params = checkpoint['auto_tuner_state']['best_params']
            self.auto_tuner.best_throughput = checkpoint['auto_tuner_state']['best_throughput']

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized time_ids computation for H100"""
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

    def get_sigmas(self) -> torch.Tensor:
        """
        Modified to properly handle ZTSNR with correct scaling factors.
        The last timestep should use σ = 20000 as a practical approximation of ∞.
        """
        ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigmas = torch.empty_like(ramp)
        
        # Regular steps (all except last)
        sigmas[:-1] = (min_inv_rho * (1 - ramp[:-1])) ** self.rho
        
        # Set final step to practical infinity (20000) for ZTSNR
        sigmas[-1] = 20000.0  # Practical approximation of infinity
        
        return sigmas.to(self.device)

    def get_karras_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified Karras Preconditioner scaling factors for v-prediction with ZTSNR support.
        For high σ (approximating ∞):
            cskip(σ) = σ²_data / (σ² + σ²_data) ≈ 0
            cout(σ) = -σ·σ_data / √(σ² + σ²_data) ≈ -σ_data
            cin(σ) = 1/√(σ² + σ²_data)
        """
        sigma_sq = sigma * sigma
        sigma_data_sq = self.sigma_data * self.sigma_data
        denominator = sigma_data_sq + sigma_sq
        denominator_sqrt = torch.sqrt(denominator)
        
        # Compute scaling factors according to v-prediction formulation
        c_skip = sigma_data_sq / denominator
        c_out = -sigma * self.sigma_data / denominator_sqrt
        c_in = 1.0 / denominator_sqrt
        
        return c_skip, c_out, c_in


    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-Noise Ratio as defined in the paper for given sigma:
        SNR(σ) = (σ_data / σ)²
        """
        return (self.sigma_data / sigma)**2

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute MinSNR weights as described in the paper.
        Uses alphas from the scheduler:
        snr_t = alpha_t / (1 - alpha_t), and then clamp snr_t by min_snr_gamma.
        
        weights = min(snr_t, min_snr_gamma)
        """
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]
        snr_t = alpha_t / (1 - alpha_t)
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr_t, min_snr)
        return weights.float()

    def _prepare_static_input(self, batch):
        """Prepare static input tensors for CUDA graph capture"""
        # Unpack batch data (assuming tuple format from NovelAIDataset)
        images, text_embeds, tag_weights = batch
        
        # Move tensors to device
        images = images.to(self.device, non_blocking=True)
        if isinstance(text_embeds, dict):
            text_embeds = {k: v.to(self.device, non_blocking=True) 
                          for k, v in text_embeds.items()}
        else:
            text_embeds = text_embeds.to(self.device, non_blocking=True)
        tag_weights = tag_weights.to(self.device, non_blocking=True)
        
        return {
            'images': images,
            'text_embeds': text_embeds,
            'tag_weights': tag_weights
        }

    def _update_static_input(self, static_input, batch):
        """Update static input tensors with new batch data"""
        # Unpack batch data (assuming tuple format from NovelAIDataset)
        images, text_embeds, tag_weights = batch
        
        # Update tensors
        static_input['images'].copy_(images.to(self.device, non_blocking=True))
        if isinstance(text_embeds, dict):
            for k, v in text_embeds.items():
                static_input['text_embeds'][k].copy_(v.to(self.device, non_blocking=True))
        else:
            static_input['text_embeds'].copy_(text_embeds.to(self.device, non_blocking=True))
        static_input['tag_weights'].copy_(tag_weights.to(self.device, non_blocking=True))

    def _capture_cuda_graph(self, static_input):
        """Capture CUDA graph for training iteration with proper output handling"""
        # Only capture graph if CUDA is available and version supports it
        if not (torch.cuda.is_available() and 
                hasattr(torch.cuda, 'CUDAGraph') and
                torch.cuda.get_device_capability()[0] >= 7):
            return None
            
        try:
            # Pre-allocate output tensors
            self.static_outputs = {
                'loss': torch.zeros(1, device=self.device),
                'v_pred': torch.zeros_like(static_input['images']),
                'grad_norm': torch.zeros(1, device=self.device),
                'timesteps': torch.zeros(static_input['images'].shape[0], 
                                       dtype=torch.long, device=self.device)
            }
            
            # Ensure all inputs are contiguous and warmed up
            for k, v in static_input.items():
                if isinstance(v, torch.Tensor):
                    static_input[k] = v.contiguous()
                elif isinstance(v, dict):
                    static_input[k] = {
                        sub_k: sub_v.contiguous() 
                        for sub_k, sub_v in v.items()
                    }
            
            # Warmup before capture
            for _ in range(3):
                loss, v_pred, grad_norm, timesteps = self.training_step(
                    static_input['images'],
                    static_input['text_embeds'],
                    static_input['tag_weights']
                )
            
            torch.cuda.synchronize()
            
            # Start graph capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(s):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    loss, v_pred, grad_norm, timesteps = self.training_step(
                        static_input['images'],
                        static_input['text_embeds'],
                        static_input['tag_weights']
                    )
                    
                    # Copy outputs to pre-allocated tensors
                    if isinstance(loss, torch.Tensor):
                        self.static_outputs['loss'].copy_(loss.detach())
                    self.static_outputs['v_pred'].copy_(v_pred.detach())
                    if isinstance(grad_norm, torch.Tensor):
                        self.static_outputs['grad_norm'].copy_(grad_norm.detach())
                    self.static_outputs['timesteps'].copy_(timesteps)
            
            torch.cuda.current_stream().wait_stream(s)
            
            # Store graph and return
            self.cuda_graph = graph
            return graph
            
        except RuntimeError as e:
            print(f"Warning: Failed to capture CUDA graph: {str(e)}")
            # Clean up on failure
            self.static_outputs = None
            self.cuda_graph = None
            return None
            
    def _get_graph_outputs(self):
        """Retrieve outputs from last graph execution"""
        if not hasattr(self, 'static_outputs'):
            return None
            
        return (
            self.static_outputs['loss'].item(),
            self.static_outputs['v_pred'],
            self.static_outputs['grad_norm'].item(),
            self.static_outputs['timesteps']
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Optimized training loop with CUDA graph support"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        # Enable tensor cores and cudnn benchmarking
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Initialize CUDA graph components
        static_input = None
        use_cuda_graphs = (torch.cuda.is_available() and 
                          hasattr(torch.cuda, 'CUDAGraph') and
                          torch.cuda.get_device_capability()[0] >= 7)
        
        # Training loop with progress bar
        with tqdm.tqdm(dataloader, desc=f'Epoch {epoch}', total=num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Handle CUDA graph execution
                    if use_cuda_graphs:
                        if static_input is None:
                            # First iteration: prepare and capture
                            static_input = self._prepare_static_input(batch)
                            if self._capture_cuda_graph(static_input) is None:
                                use_cuda_graphs = False
                                print("Falling back to regular training without CUDA graphs")
                        else:
                            # Subsequent iterations: update and replay
                            self._update_static_input(static_input, batch)
                            self.cuda_graph.replay()
                    
                    # Get outputs either from graph or direct training
                    if use_cuda_graphs:
                        loss, v_pred, grad_norm, timesteps = self._get_graph_outputs()
                    else:
                        # Regular training step without CUDA graphs
                        images, text_embeds, tag_weights = batch
                        loss, v_pred, grad_norm, timesteps = self.training_step(
                            images,
                            text_embeds,
                            tag_weights
                        )
                    
                    # Update metrics
                    if isinstance(loss, torch.Tensor):
                        loss_value = loss.item()
                    else:
                        loss_value = loss
                    total_loss += loss_value
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss_value:.4f}',
                        'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                        'grad_norm': f'{grad_norm:.4f}' if isinstance(grad_norm, float) else 'N/A',
                        'cuda_graphs': 'enabled' if use_cuda_graphs else 'disabled'
                    })
                    
                    # Log metrics if available
                    if hasattr(self, 'log_detailed_metrics'):
                        self.log_detailed_metrics(loss, v_pred, grad_norm, timesteps)
                    
                    # Auto-tune if needed
                    if self.auto_tuner and batch_idx > 0 and batch_idx % 50 == 0:
                        self.auto_tuner.update(total_loss / (batch_idx + 1))
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        print(f"WARNING: out of memory on batch {batch_idx}. Skipping batch.")
                        if static_input is not None:
                            del static_input
                            static_input = None
                        if hasattr(self, 'static_outputs'):
                            del self.static_outputs
                        if hasattr(self, 'cuda_graph'):
                            del self.cuda_graph
                        use_cuda_graphs = False
                        continue
                    else:
                        raise e
        
        return total_loss / num_batches

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True
    ) -> DataLoader:
        """Create optimized dataloader for training"""
        if batch_size is None:
            batch_size = 1  # Default batch size if not specified
        
        # Create sampler that groups by latent dimensions
        sampler = AspectBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # Create dataloader with optimized settings
        return DataLoader(
            dataset,
            batch_sampler=sampler,  # Use custom sampler for dimension-matched batches
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator().manual_seed(42),  # Reproducible shuffling
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),  # Seed workers
        )

    def save_checkpoint(self, path: str):
        """Save checkpoint with progress information"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'progress': self.progress,  # Save progress state
            'profiler_state': self.profiler.metrics,
            'auto_tuner_state': {
                'current_params': self.auto_tuner.current_params,
                'best_params': self.auto_tuner.best_params,
                'best_throughput': self.auto_tuner.best_throughput
            }
        }
        torch.save(checkpoint, path)

    def compute_grad_norm(self):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def log_detailed_metrics(self,
                            loss: torch.Tensor,
                            v_pred: torch.Tensor,
                            grad_norm: float,
                            timesteps: torch.Tensor):
        """Log detailed training metrics to W&B"""
        # Safely check if we should log
        should_log = hasattr(self.accelerator, 'is_main_process') and \
                     self.accelerator.is_main_process and \
                     wandb.run is not None
                 
        if not should_log:
            return
        
        # Compute v-prediction statistics
        v_pred_mean = v_pred.mean().item()
        v_pred_std = v_pred.std().item()
        v_pred_min = v_pred.min().item()
        v_pred_max = v_pred.max().item()
        
        # Log detailed metrics
        wandb.log({
            'loss/total': loss.item(),
            'v_pred/mean': v_pred_mean,
            'v_pred/std': v_pred_std,
            'v_pred/min': v_pred_min,
            'v_pred/max': v_pred_max,
            'grad/norm': grad_norm,
            'timesteps/mean': timesteps.float().mean().item(),
            'timesteps/std': timesteps.float().std().item()
        })

    def _accumulate_gradients(self, loss, batch_size):
        scaled_loss = loss / (self.gradient_accumulation_steps * batch_size)
        self.scaler.scale(scaled_loss).backward()
        
        if self.gradient_accumulation_steps > 1:
            # Synchronize gradients across devices if needed
            if dist.is_initialized():
                self.model.require_backward_grad_sync = (
                    self.current_step % self.gradient_accumulation_steps == 0
                )

    def _handle_oom_error(self, batch_idx):
        try:
            torch.cuda.empty_cache()
            if hasattr(self, 'cuda_graph'):
                del self.cuda_graph
            if hasattr(self, 'static_input'):
                del self.static_input
            
            # Try to recover by reducing batch size
            if self.batch_size > 1:
                self.batch_size //= 2
                print(f"Reducing batch size to {self.batch_size}")
                return True
            return False
        except Exception as e:
            print(f"Failed to recover from OOM: {e}")
            return False
