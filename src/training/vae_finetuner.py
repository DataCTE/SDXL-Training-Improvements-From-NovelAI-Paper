"""Ultra-optimized VAE finetuner with GPU acceleration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from contextlib import nullcontext
import weakref
from pathlib import Path
from src.utils.progress import ProgressTracker
from src.config.args import VAEConfig
from src.utils.vae_utils import normalize_vae_latents
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """Enhanced perceptual loss with NAI improvements."""
    
    def __init__(self, resize=True):
        super().__init__()
        from torchvision.models import vgg16
        
        vgg = vgg16(pretrained=True).eval()
        # Use more VGG layers for better feature matching
        self.slice1 = nn.Sequential(*vgg.features[:4]).eval()
        self.slice2 = nn.Sequential(*vgg.features[4:9]).eval()
        self.slice3 = nn.Sequential(*vgg.features[9:16]).eval()
        self.slice4 = nn.Sequential(*vgg.features[16:23]).eval()  # Added deeper layer
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.resize = T.Resize(224, interpolation=T.InterpolationMode.BICUBIC) if resize else None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced perceptual loss between input and target images."""
        if self.resize:
            x = self.resize(x)
            target = self.resize(target)
            
        x = normalize_vae_latents(x)  # Use NAI normalization
        target = normalize_vae_latents(target)
        
        loss = 0.0
        for slice_model in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x_feat = slice_model(x)
            target_feat = slice_model(target)
            loss += F.mse_loss(x_feat, target_feat)
            
        return loss
class CachedTensor:
    """Memory-efficient tensor cache with weak references."""
    
    def __init__(self, size: int):
        self.size = size
        self.cache = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, creator) -> torch.Tensor:
        """Get tensor from cache or create new one."""
        if key in self.cache:
            self._hits += 1
            return self.cache[key]()  # Dereference weakref
            
        self._misses += 1
        tensor = creator()
        if len(self.cache) >= self.size:
            # Evict oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = weakref.ref(tensor)
        return tensor
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

class VAEFinetuner:
    """Optimized VAE finetuner with GPU acceleration."""
    
    def __init__(
        self,
        vae: nn.Module,
        config: VAEConfig,
        train_dataloader: DataLoader,
        device: torch.device
    ):
        """Initialize VAE finetuner.
        
        Args:
            vae: VAE model to finetune
            config: VAE training configuration
            train_dataloader: Training data loader
            device: Target device for training
        """
        self.vae = vae.to(device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.device = device
        
        # Initialize perceptual loss if configured
        self.perceptual_loss = None
        if config.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(resize=True).to(device)
        
        # Initialize optimizer with config parameters
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0,
            betas=(config.adam_beta1 if hasattr(config, 'adam_beta1') else 0.9,
                  config.adam_beta2 if hasattr(config, 'adam_beta2') else 0.999),
            eps=config.adam_epsilon if hasattr(config, 'adam_epsilon') else 1e-8
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None
        
        # Setup progress tracking
        self.progress = ProgressTracker(
            description="VAE Training",
            total_steps=config.num_epochs * len(train_dataloader),
            log_steps=10,
            save_steps=100,
            eval_steps=100
        )
        
        # Setup CUDA graphs if enabled
        self._cuda_graphs = {}
        if config.enable_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()
        
        # Initialize tensor cache
        self.tensor_cache = CachedTensor(config.cache_size)
        
        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
        
        # Track training state
        self.step = 0
        self._warmup_steps = config.num_warmup_steps

    def _setup_cuda_graphs(self) -> None:
        """Initialize CUDA graphs for common operations."""
        try:
            static_input = torch.randn(
                self.config.batch_size, 3, 256, 256,
                device=self.device
            )
            
            # Capture forward graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # Warmup
                    self.vae(static_input)
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = self.vae(static_input)
            
            self._cuda_graphs["forward"] = (g, static_input, static_output)
            
            torch.cuda.current_stream().wait_stream(s)
            
        except Exception as e:
            logger.warning(f"Failed to setup CUDA graphs: {e}")
            self._cuda_graphs.clear()
    
    @torch.no_grad()
    def _run_cuda_graph(
        self,
        input_tensor: torch.Tensor,
        graph_key: str
    ) -> torch.Tensor:
        """Execute cached CUDA graph."""
        if graph_key not in self._cuda_graphs:
            return None
            
        g, static_in, static_out = self._cuda_graphs[graph_key]
        static_in.copy_(input_tensor)
        g.replay()
        return static_out.clone()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute optimized training step."""
        try:
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) 
                    for k, v in batch.items()}
            
            # Determine precision context
            if self.config.mixed_precision == "fp16":
                ctx = autocast(dtype=torch.float16)
            elif self.config.mixed_precision == "bf16":
                ctx = autocast(dtype=torch.bfloat16)
            else:
                ctx = nullcontext()
            
            # Forward pass with mixed precision
            with ctx:
                # Try CUDA graph first
                output = None
                if self._cuda_graphs and self.step > self._warmup_steps:
                    output = self._run_cuda_graph(
                        batch["pixel_values"],
                        "forward"
                    )
                
                if output is None:  # Fall back to regular forward
                    output = self.vae(batch["pixel_values"])
                
                # Apply NAI normalization to latents
                normalized_latents = normalize_vae_latents(output.latent_dist.sample())
                
                # Calculate reconstruction loss on normalized latents
                recon_loss = F.mse_loss(
                    normalized_latents,
                    normalize_vae_latents(batch["pixel_values"]),
                    reduction="mean"
                )
                
                # Add KL divergence loss if configured
                if self.config.kl_weight > 0:
                    kl_loss = output.latent_dist.kl().mean()
                    loss = recon_loss + self.config.kl_weight * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    loss = recon_loss
                
                # Add perceptual loss if configured
                if self.config.perceptual_weight > 0:
                    if not hasattr(self, 'perceptual_loss'):
                        # Initialize perceptual loss module once
                        self.perceptual_loss = PerceptualLoss(resize=True).to(self.device)
                    
                    # Calculate perceptual loss using forward method
                    p_loss = self.perceptual_loss(
                        output.sample, 
                        batch["pixel_values"]
                    )
                    loss = loss + self.config.perceptual_weight * p_loss
            
            # Backward pass with automatic mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if self.config.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.vae.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                loss.backward()
                
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.vae.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update state
            self.step += 1
            if self.step <= self.config.num_warmup_steps:
                self._warmup_steps = self.step
            
            # Gather metrics
            metrics = {
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "cache_hit_rate": self.tensor_cache.hit_rate,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            
            return loss, metrics
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise

    def train(self) -> Dict[str, List[float]]:
        """Execute training loop with optimizations."""
        try:
            logger.info(f"Starting VAE training for {self.config.num_epochs} epochs...")
            
            history = defaultdict(list)
            self.vae.train()
            
            for epoch in range(self.config.num_epochs):
                epoch_metrics = defaultdict(float)
                num_batches = len(self.train_dataloader)
                
                # Training loop
                for batch in self.train_dataloader:
                    __, step_metrics = self.train_step(batch)
                    
                    # Update metrics
                    for k, v in step_metrics.items():
                        epoch_metrics[k] += v
                        history[k].append(v)
                    
                    # Log progress
                    if self.progress.should_log(self.step):
                        avg_metrics = {
                            k: v / (self.step % num_batches + 1)
                            for k, v in epoch_metrics.items()
                        }
                        logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} "
                            f"Step {self.step}: {avg_metrics}"
                        )
                    
                    # Save checkpoint if configured
                    if self.progress.should_save(self.step):
                        save_path = Path(self.config.output_dir) / f"vae_step_{self.step}.pt"
                        self.save(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Log epoch summary
                avg_epoch_metrics = {
                    k: v / num_batches for k, v in epoch_metrics.items()
                }
                logger.info(
                    f"Epoch {epoch+1} completed: {avg_epoch_metrics}"
                )
            
            logger.info("Training completed successfully!")
            return dict(history)
            
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            raise
    
    def save(self, path: str) -> None:
        """Save model state with error handling."""
        try:
            torch.save({
                "model_state": self.vae.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config,
            }, path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device
    ) -> "VAEFinetuner":
        """Load model state with error handling."""
        try:
            checkpoint = torch.load(path, map_location=device)
            config = checkpoint["config"]
            
            finetuner = cls(
                vae=None,  # User must provide VAE model
                config=config,
                train_dataloader=None,  # User must provide train_dataloader
                device=device
            )
            
            finetuner.vae.load_state_dict(checkpoint["model_state"])
            finetuner.optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
            finetuner.step = checkpoint["step"]
            
            return finetuner
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

def setup_vae_finetuner(
    vae: nn.Module,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    mixed_precision: str = "fp16",
    enable_cuda_graphs: bool = True,
    cache_size: int = 32,
    num_warmup_steps: int = 100,
    device: Optional[torch.device] = None
) -> VAEFinetuner:
    """
    Factory function to create and configure a VAEFinetuner instance.
    
    Args:
        vae: The VAE model to finetune
        learning_rate: Learning rate for optimization
        batch_size: Training batch size
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: Mixed precision mode ("no", "fp16", or "bf16")
        enable_cuda_graphs: Whether to use CUDA graphs for optimization
        cache_size: Size of tensor cache
        num_warmup_steps: Number of warmup steps before CUDA graphs
        device: Target device for training
        
    Returns:
        Configured VAEFinetuner instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    config = VAEConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        mixed_precision=mixed_precision,
        enable_cuda_graphs=enable_cuda_graphs,
        cache_size=cache_size,
        num_warmup_steps=num_warmup_steps
    )
    
    return VAEFinetuner(vae=vae, config=config, train_dataloader=None, device=device)

