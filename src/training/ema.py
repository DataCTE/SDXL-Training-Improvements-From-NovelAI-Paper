import torch
import logging
from typing import Optional, Union
from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)

class EMAModel:
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_path: str,
        decay: float = 0.9999,
        update_after_step: int = 100,
        device: Optional[Union[str, torch.device]] = None,
        update_every: int = 1,
        use_ema_warmup: bool = True,
        power: float = 2/3,
        min_decay: float = 0.0,
        max_decay: float = 0.9999,
        mixed_precision: str = "bf16",
        jit_compile: bool = False,
        gradient_checkpointing: bool = True,
    ):
        """Initialize EMA model with enhanced configuration and optimizations."""
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.power = power
        self.min_decay = min_decay
        self.max_decay = max_decay
        
        # Handle device initialization
        if device is None or (isinstance(device, str) and device.lower() == 'auto'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        self.mixed_precision = mixed_precision
        
        # Load and optimize EMA model
        logger.info("Loading EMA model from %s", model_path)
        try:
            self._initialize_ema_model(
                model_path=model_path,
                model=model,
                jit_compile=jit_compile,
                gradient_checkpointing=gradient_checkpointing
            )
            self._log_config()
        except Exception as e:
            logger.error("Failed to load EMA model: %s", str(e))
            raise
            
        self.optimization_step = 0

    def _initialize_ema_model(
        self,
        model_path: str,
        model: torch.nn.Module,
        jit_compile: bool,
        gradient_checkpointing: bool
    ) -> None:
        """Initialize and optimize EMA model with proper configuration."""
        # Determine dtype based on mixed precision setting
        dtype = self._get_dtype()
        
        # Load pipeline with optimized settings
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype if str(self.device).startswith("cuda") else torch.float32
        )
        
        # Setup UNet model
        self.ema_model = pipeline.unet.to(self.device)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Enable memory optimizations
        self._enable_optimizations(gradient_checkpointing)
        
        # JIT compile if requested
        if jit_compile and torch.cuda.is_available():
            logger.info("JIT compiling EMA model...")
            self.ema_model = torch.compile(self.ema_model)
        
        # Initialize weights
        self.copy_from(model)

    @torch.no_grad()
    def _get_dtype(self) -> torch.dtype:
        """Get appropriate dtype based on mixed precision setting."""
        if self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _enable_optimizations(self, gradient_checkpointing: bool) -> None:
        """Enable memory and performance optimizations."""
        if hasattr(self.ema_model, 'enable_xformers_memory_efficient_attention'):
            self.ema_model.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing and hasattr(self.ema_model, 'enable_gradient_checkpointing'):
            self.ema_model.enable_gradient_checkpointing()

    @torch.no_grad()
    def _get_decay_rate(self) -> float:
        """Calculate current decay rate with optimized computation."""
        step = max(0, self.optimization_step - self.update_after_step - 1)
        
        if self.use_ema_warmup and step <= self.update_after_step:
            decay = min(
                self.max_decay,
                (1 + step / self.update_after_step) ** -self.power
            )
        else:
            decay = self.decay
            
        return max(self.min_decay, min(decay, self.max_decay))

    @torch.no_grad()
    def step(self, model: torch.nn.Module) -> None:
        """Optimized EMA update step."""
        self.optimization_step += 1
        
        if (self.optimization_step <= self.update_after_step or 
            self.optimization_step % self.update_every != 0):
            return
            
        decay = self._get_decay_rate()
        
        # Batch update parameters
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name, param in model_params.items():
            if name in ema_params:
                ema_param = ema_params[name]
                if param.device != ema_param.device:
                    param = param.to(ema_param.device)
                ema_param.copy_(
                    ema_param * decay + param.data * (1 - decay)
                )

    def _log_config(self) -> None:
        """Log current configuration with enhanced details."""
        logger.info("\nEMA Model Configuration:")
        logger.info("- Device: %s", self.device)
        logger.info("- Mixed Precision: %s", self.mixed_precision)
        logger.info("- Base Decay Rate: %s", self.decay)
        logger.info("- Update After Step: %s", self.update_after_step)
        logger.info("- Update Every: %s", self.update_every)
        logger.info("- EMA Warmup: %s", self.use_ema_warmup)
        logger.info("- Min/Max Decay: %s/%s", self.min_decay, self.max_decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Optimized parameter copy to target model."""
        model_dict = model.state_dict()
        ema_dict = self.ema_model.state_dict()
        
        for key in model_dict:
            if key in ema_dict:
                if model_dict[key].shape == ema_dict[key].shape:
                    model_dict[key].copy_(ema_dict[key])
                else:
                    logger.warning(
                        "Shape mismatch for parameter %s: "
                        "EMA shape %s, "
                        "Model shape %s",
                        key, ema_dict[key].shape, model_dict[key].shape
                    )
        model.load_state_dict(model_dict)

    @torch.no_grad()
    def copy_from(self, model: torch.nn.Module) -> None:
        """Optimized parameter copy from source model."""
        ema_dict = self.ema_model.state_dict()
        model_dict = model.state_dict()
        
        for key in ema_dict:
            if key in model_dict:
                if ema_dict[key].shape == model_dict[key].shape:
                    ema_dict[key].copy_(model_dict[key])
                else:
                    logger.warning(
                        "Shape mismatch for parameter %s: "
                        "EMA shape %s, "
                        "Model shape %s",
                        key, ema_dict[key].shape, model_dict[key].shape
                    )
        self.ema_model.load_state_dict(ema_dict)

    def get_model(self) -> torch.nn.Module:
        """Get the EMA model."""
        return self.ema_model

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> 'EMAModel':
        """Move EMA model to specified device and dtype."""
        if device is not None:
            self.device = device
            self.ema_model.to(device)
        if dtype is not None:
            self.ema_model.to(dtype=dtype)
        return self