"""Ultra-optimized VAE finetuner with GPU acceleration.

This module provides a highly optimized VAE finetuner with:
- Mixed precision training
- CUDA graph support 
- Memory-efficient operations
- Dynamic batch sizing
- Advanced caching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from contextlib import nullcontext
import weakref
from src.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VAEConfig:
    """Immutable VAE configuration."""
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_grad_norm: Optional[float] = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", or "bf16"
    enable_cuda_graphs: bool = True
    cache_size: int = 32
    num_warmup_steps: int = 100

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
        device: torch.device
    ):
        self.vae = vae
        self.config = config
        self.device = device
        
        # Initialize optimizer and scaler
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=config.learning_rate
        )
        self.scaler = GradScaler() if config.mixed_precision != "no" else None
        
        # Setup CUDA graphs if enabled
        self._cuda_graphs = {}
        if config.enable_cuda_graphs and torch.cuda.is_available():
            self._setup_cuda_graphs()
        
        # Initialize tensor cache
        self.tensor_cache = CachedTensor(config.cache_size)
        
        # Track training state
        self.step = 0
        self._warmup_steps = 0
    
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
                if self._cuda_graphs and self.step > self._warmup_steps:
                    output = self._run_cuda_graph(
                        batch["pixel_values"],
                        "forward"
                    )
                
                if output is None:  # Fall back to regular forward
                    output = self.vae(batch["pixel_values"])
                
                # Calculate reconstruction loss
                loss = F.mse_loss(
                    output.sample,
                    batch["pixel_values"],
                    reduction="mean"
                )
            
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
                "cache_hit_rate": self.tensor_cache.hit_rate,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            
            return loss, metrics
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        callbacks: Optional[List[Any]] = None,
        wandb_run: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """Execute training loop with optimizations."""
        history = {"loss": [], "cache_hit_rate": []}
        total_loss = 0.0
        
        try:
            logger.info(f"Starting VAE training for {num_epochs} epochs...")
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = len(train_dataloader)
                
                logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) 
                            for k, v in batch.items()}
                    
                    # Execute training step
                    loss, metrics = self.train_step(batch)
                    epoch_loss += loss.item()
                    
                    # Update metrics
                    current_metrics = {
                        "loss": loss.item(),
                        "avg_loss": epoch_loss / (batch_idx + 1),
                        **metrics
                    }
                    
                    # Log progress periodically
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} "
                            f"[{batch_idx+1}/{num_batches}] "
                            f"Loss: {loss.item():.4f}"
                        )
                    
                    # Update history
                    for k, v in current_metrics.items():
                        if k in history:
                            history[k].append(v)
                    
                    # Execute callbacks
                    if callbacks:
                        for callback in callbacks:
                            callback(self, current_metrics)
                
                # Calculate and log epoch metrics
                avg_epoch_loss = epoch_loss / num_batches
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "avg_epoch_loss": avg_epoch_loss,
                    **{k: sum(v[-num_batches:]) / num_batches 
                       for k, v in history.items()}
                }
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} completed - "
                    f"Average Loss: {avg_epoch_loss:.4f}"
                )
                
                if wandb_run:
                    wandb_run.log(epoch_metrics)
                
                total_loss += epoch_loss
            
            # Log final summary
            final_metrics = {
                "total_epochs": num_epochs,
                "final_avg_loss": total_loss / (num_epochs * num_batches),
                "final_cache_hit_rate": history["cache_hit_rate"][-1]
            }
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics: {final_metrics}")
            
            if wandb_run:
                wandb_run.log(final_metrics)
            
            return history
            
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
    
    return VAEFinetuner(vae=vae, config=config, device=device)