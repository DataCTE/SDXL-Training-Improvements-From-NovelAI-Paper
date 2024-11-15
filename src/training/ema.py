import torch
import copy
import math
import logging
from typing import Optional, Union, Dict, Any
from collections import deque
from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)

class EMAModel:
    """
    EMA model that maintains a separate SDXL instance for parameter averaging.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        model_path: str,
        decay: float = 0.9999,
        update_after_step: int = 100,
        device: Optional[torch.device] = None,
        update_every: int = 1,
        use_ema_warmup: bool = True,
        power: float = 2/3,
        min_decay: float = 0.0,
        max_decay: float = 0.9999,
        mixed_precision: str = "bf16",
        jit_compile: bool = False,
        gradient_checkpointing: bool = True,
    ):
        """
        Initialize EMA model by loading a fresh SDXL instance.
        
        Args:
            model: Reference model (used only for initialization)
            model_path: Path to load the SDXL model from
            decay: Base decay rate for exponential moving average
            update_after_step: Start EMA after this many steps
            device: Device to store EMA model on
            update_every: Update every N steps
            use_ema_warmup: Whether to use warmup period for EMA
            power: Power for decay rate schedule during warmup
            min_decay: Minimum decay rate
            max_decay: Maximum decay rate
            mixed_precision: Mixed precision type ("no", "fp16", "bf16")
            jit_compile: Whether to JIT compile the model
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.power = power
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        
        # Load a fresh SDXL pipeline for EMA
        logger.info(f"Loading EMA model from {model_path}")
        try:
            dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype if str(self.device).startswith("cuda") else torch.float32
            )
            self.ema_model = pipeline.unet
            self.ema_model.to(self.device)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
            
            # Enable optimizations
            if hasattr(self.ema_model, 'enable_xformers_memory_efficient_attention'):
                self.ema_model.enable_xformers_memory_efficient_attention()
            if gradient_checkpointing and hasattr(self.ema_model, 'enable_gradient_checkpointing'):
                self.ema_model.enable_gradient_checkpointing()
            
            # JIT compile if requested
            if jit_compile and torch.cuda.is_available():
                logger.info("JIT compiling EMA model...")
                self.ema_model = torch.compile(self.ema_model)
            
            # Initialize EMA weights from the input model
            self.copy_from(model)
            
            # Log configuration
            self._log_config()
            
        except Exception as e:
            logger.error(f"Failed to load EMA model: {str(e)}")
            raise
            
        self.optimization_step = 0
            
    def _get_decay_rate(self) -> float:
        """Calculate the current decay rate with optional warmup."""
        step = max(0, self.optimization_step - self.update_after_step - 1)
        
        if self.use_ema_warmup and step <= self.update_after_step:
            # Warmup phase: gradually increase decay
            decay = min(
                self.max_decay,
                (1 + step / self.update_after_step) ** -self.power
            )
        else:
            # Normal phase: use base decay
            decay = self.decay
            
        return max(self.min_decay, min(decay, self.max_decay))
    
    def step(self, model: torch.nn.Module):
        """
        Update EMA model parameters.
        
        Args:
            model: The current model to take updates from
        """
        self.optimization_step += 1
        
        # Skip update if before start or not on update step
        if (self.optimization_step <= self.update_after_step or 
            self.optimization_step % self.update_every != 0):
            return
            
        decay = self._get_decay_rate()
        
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())
            
            # Update each parameter
            for name, param in model_params.items():
                if name in ema_params:
                    ema_param = ema_params[name]
                    # Ensure parameters are on the same device
                    if param.device != ema_param.device:
                        param = param.to(ema_param.device)
                    # Simple EMA update
                    ema_param.copy_(
                        ema_param * decay + param.data * (1 - decay)
                    )
    
    def copy_to(self, model: torch.nn.Module):
        """Copy EMA parameters to target model."""
        with torch.no_grad():
            model_dict = model.state_dict()
            ema_dict = self.ema_model.state_dict()
            for key in model_dict:
                if key in ema_dict:
                    if model_dict[key].shape == ema_dict[key].shape:
                        model_dict[key].copy_(ema_dict[key])
                    else:
                        logger.warning(
                            f"Shape mismatch for parameter {key}: "
                            f"EMA shape {ema_dict[key].shape}, "
                            f"Model shape {model_dict[key].shape}"
                        )
            model.load_state_dict(model_dict)
            
    def copy_from(self, model: torch.nn.Module):
        """Initialize EMA parameters from model."""
        with torch.no_grad():
            ema_dict = self.ema_model.state_dict()
            model_dict = model.state_dict()
            for key in ema_dict:
                if key in model_dict:
                    if ema_dict[key].shape == model_dict[key].shape:
                        ema_dict[key].copy_(model_dict[key])
                    else:
                        logger.warning(
                            f"Shape mismatch for parameter {key}: "
                            f"EMA shape {ema_dict[key].shape}, "
                            f"Model shape {model_dict[key].shape}"
                        )
            self.ema_model.load_state_dict(ema_dict)
    
    def get_model(self) -> torch.nn.Module:
        """Get the EMA model."""
        return self.ema_model

    def _log_config(self):
        """Log the current configuration"""
        logger.info("\nEMA Model Configuration:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Mixed Precision: {self.mixed_precision}")
        logger.info(f"- Base Decay Rate: {self.decay}")
        logger.info(f"- Update After Step: {self.update_after_step}")
        logger.info(f"- Update Every: {self.update_every}")
        logger.info(f"- EMA Warmup: {self.use_ema_warmup}")
        logger.info(f"- Min/Max Decay: {self.min_decay}/{self.max_decay}")

    def to(self, device=None, dtype=None):
        """Move the EMA model to specified device and dtype"""
        if device is not None:
            self.device = device
            self.ema_model.to(device)
        if dtype is not None:
            self.ema_model.to(dtype=dtype)
        return self
