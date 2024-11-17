import torch
import logging
import os
import torch.nn.functional as F
from src.training.optimizers.setup_optimizers import setup_optimizer
from torch.cuda.amp import GradScaler
from collections import deque
from src.training.loss_functions import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class VAELossMeter:
    """Thread-safe loss meter for VAE training."""
    window_size: int = 100
    
    def __post_init__(self):
        self._losses = deque(maxlen=self.window_size)
        self._lock = Lock()
    
    def update(self, loss: float) -> None:
        with self._lock:
            self._losses.append(loss)
    
    def get_average(self) -> float:
        with self._lock:
            if not self._losses:
                return 0.0
            return sum(self._losses) / len(self._losses)
        
    def get_loss_history(self) -> list:
        with self._lock:
            return list(self._losses)
        
    def set_loss_history(self, history: list) -> None:
        with self._lock:
            self._losses = deque(history, maxlen=self.window_size)

@torch.jit.script
def _normalize_latents(
    latents: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled latent normalization"""
    return (latents - means) / stds

@torch.jit.script
def _denormalize_latents(
    latents: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled latent denormalization"""
    return latents * stds + means

@torch.jit.script
def _compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """JIT-compiled KL divergence computation"""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

class VAEFineTuner:
    def __init__(
        self,
        vae,
        device="cuda",
        mixed_precision="no",
        use_amp=False,
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=1e-2,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        use_8bit_adam=False,
        use_channel_scaling=False,
        adaptive_loss_scale=False,
        kl_weight=0.0,
        perceptual_weight=0.0,
        min_snr_gamma=5.0,
        initial_scale_factor=1.0,
        decay=0.9999,
        update_after_step=100,
        model_path=None,
        use_cache=True,
        cache_size=1000,
    ):
        """Initialize VAE finetuner with configuration"""
        self.vae = vae
        self.device = device
        self.mixed_precision = mixed_precision
        self.use_amp = use_amp
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.use_8bit_adam = use_8bit_adam
        self.use_channel_scaling = use_channel_scaling
        self.adaptive_loss_scale = adaptive_loss_scale
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.min_snr_gamma = min_snr_gamma
        self.initial_scale_factor = initial_scale_factor
        self.decay = decay
        self.update_after_step = update_after_step
        self.model_path = model_path
        
        # Initialize perceptual model as None
        self.perceptual_model = None
        
        # Initialize caching system
        self.use_cache = use_cache
        self.cache = {}
        self._cache_size = cache_size
        
        self._init_optimizer()
        self._init_compiled_ops()
        self._init_buffers()
        
        # Initialize and configure VAE model.
        self.vae = self.vae.to(self.device)
        
        # Configure model precision
        if self.use_amp and self.mixed_precision == "fp16":
            self.vae = self.vae.to(dtype=torch.float16)
        elif self.mixed_precision == "bf16":
            self.vae = self.vae.to(dtype=torch.bfloat16)
            
        # Enable optimizations
        if hasattr(self.vae, 'enable_xformers_memory_efficient_attention'):
            self.vae.enable_xformers_memory_efficient_attention()
        if self.gradient_checkpointing and hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()

        # Initialize adaptive scaling
        if self.adaptive_loss_scale:
            self.scale_factor = self.initial_scale_factor
            self.current_step = 0

        self.register_vae_statistics()
        
        # Pre-allocate buffers
        self.loss_meter = VAELossMeter(window_size=100)
        self.latent_count = 0
        self.latent_means = None
        self.latent_m2 = None

    def _init_optimizer(self):
        """Initialize optimizer and related components."""
        self.optimizer = setup_optimizer(
            model=self.vae,
            optimizer_type="adamw",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            use_8bit_optimizer=self.use_8bit_adam
        )
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=1000,
            num_training_steps=10000
        )
        
        self.scaler = GradScaler() if self.use_amp else None

    def _init_compiled_ops(self):
        """Initialize compiled operations"""
        # Pre-compile VGG features extraction
        if self.perceptual_weight > 0:
            self._init_perceptual_model()
            
        # Initialize CUDA streams
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()

    def _init_perceptual_model(self):
        """Initialize VGG16 model for perceptual loss calculation"""
        try:
            import torchvision.models as models
            
            # Initialize VGG16 pretrained model
            vgg16 = models.vgg16(pretrained=True).to(self.device)
            
            # Update the perceptual model with feature layers
            self.perceptual_model = torch.nn.Sequential(
                *list(vgg16.features.children())[:29]
            ).eval()
            
            # Freeze the parameters
            for param in self.perceptual_model.parameters():
                param.requires_grad = False
                
            logger.info("Initialized VGG16 model for perceptual loss")
            
        except Exception as e:
            logger.error("Failed to initialize perceptual model: %s", str(e), exc_info=True)
            raise

    def _init_buffers(self):
        """Pre-allocate reusable buffers"""
        self.loss_buffer = torch.zeros(1, device=self.device)
        if self.perceptual_weight > 0:
            self.perceptual_buffer = torch.zeros(
                3, 224, 224, 
                device=self.device,
                dtype=self.vae.dtype
            )

    def register_vae_statistics(self):
        """Register VAE latent space statistics for normalization."""
        if not hasattr(self.vae, 'latent_means') or not hasattr(self.vae, 'latent_stds'):
            # Initialize default statistics if not present
            latent_size = self.vae.config.latent_channels
            self.vae.register_buffer('latent_means', torch.zeros(latent_size, device=self.device))
            self.vae.register_buffer('latent_stds', torch.ones(latent_size, device=self.device))
            logger.info("Initialized default VAE latent statistics")

    def update_scale_factor(self, loss: torch.Tensor) -> None:
        """Update the adaptive loss scale factor."""
        if self.current_step >= self.update_after_step:
            # Update scale factor using exponential moving average
            with torch.no_grad():
                loss_value = loss.item()
                if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                    target_scale = 1.0 / loss_value if loss_value > 0 else 1.0
                    self.scale_factor = (
                        self.scale_factor * self.decay + 
                        target_scale * (1 - self.decay)
                    )
                    # Clamp to reasonable range
                    self.scale_factor = max(min(self.scale_factor, 1e6), 1e-6)
        self.current_step += 1

    @torch.amp.autocast('cuda')
    def _forward_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """Optimized VAE forward pass"""
        with torch.cuda.stream(self.compute_stream):
            if self.use_channel_scaling:
                latents = _normalize_latents(
                    latents,
                    self.vae.latent_means.view(1, -1, 1, 1),
                    self.vae.latent_stds.view(1, -1, 1, 1)
                )
            return self.vae.decode(latents).sample

    @torch.no_grad()
    def _compute_perceptual(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """Optimized perceptual loss computation"""
        # Resize inputs to 224x224 for VGG
        x_recon_resized = F.interpolate(
            x_recon, 
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        x_target_resized = F.interpolate(
            x_target,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        # Store in buffer for reuse
        self.perceptual_buffer.copy_(x_recon_resized)
        x_recon_features = self.perceptual_model(self.perceptual_buffer)
        
        self.perceptual_buffer.copy_(x_target_resized)
        x_target_features = self.perceptual_model(self.perceptual_buffer)
        return F.mse_loss(x_recon_features, x_target_features)

    def _get_cache_key(self, latents: torch.Tensor) -> str:
        """Generate cache key for latents tensor"""
        # Use tensor hash as cache key
        return str(hash(latents.cpu().numpy().tobytes()))

    def _update_cache(self, key: str, value: torch.Tensor) -> None:
        """Update cache with new value, maintaining size limit"""
        if len(self.cache) >= self._cache_size:
            # Remove oldest item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def training_step(self, batch: dict) -> torch.Tensor:
        """Optimized VAE training step"""
        try:
            # Asynchronous data transfer
            with torch.cuda.stream(self.transfer_stream):
                latents = batch['latents'].to(
                    self.device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                original_images = batch.get('original_images')
                if original_images is not None:
                    original_images = original_images.to(
                        self.device,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )

            # Fast path for cached results
            if self.use_cache:
                cache_key = self._get_cache_key(latents)
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Synchronized compute
            torch.cuda.synchronize()
            
            # Core training step
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Decode latents
                decoded = self._forward_vae(latents)
                
                # Compute losses efficiently
                loss = torch.zeros_like(self.loss_buffer)
                
                # Reconstruction loss (always computed)
                if original_images is not None:
                    recon_loss = F.mse_loss(decoded, original_images)
                    loss += recon_loss
                
                # KL divergence loss (if enabled)
                if self.kl_weight > 0:
                    posterior = self.vae.encode(decoded).latent_dist
                    kl_loss = _compute_kl_loss(
                        posterior.mean,
                        posterior.logvar
                    )
                    loss += kl_loss * self.kl_weight
                
                # Perceptual loss (if enabled)
                if self.perceptual_weight > 0 and original_images is not None:
                    perceptual_loss = self._compute_perceptual(
                        decoded,
                        original_images
                    )
                    loss += perceptual_loss * self.perceptual_weight
                
                # Apply adaptive loss scaling
                if self.adaptive_loss_scale:
                    self.update_scale_factor(loss)
                    loss *= self.scale_factor

            # Optimizer step with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

            # Fast cleanup
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            
            # Update tracking metrics
            self.loss_meter.update(loss.item())
            
            # Cache result if enabled
            if self.use_cache:
                self._update_cache(cache_key, loss.detach())
            
            return loss

        except Exception as e:
            logger.error("Error in training step: %s", str(e), exc_info=True)
            raise

    def save_pretrained(self, save_dir: str):
        """Save VAE model and finetuning state"""
        self.vae.save_pretrained(save_dir)
        
        # Save finetuning state
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scale_factor': self.scale_factor,
            'loss_history': self.loss_meter.get_loss_history()
        }
        torch.save(state, os.path.join(save_dir, 'finetuning_state.pt'))
    
    def load_pretrained(self, load_dir: str):
        """Load VAE model and finetuning state"""
        self.vae.from_pretrained(load_dir)
        
        # Load finetuning state
        state_path = os.path.join(load_dir, 'finetuning_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.scale_factor = state['scale_factor']
            self.loss_meter.set_loss_history(state['loss_history'])

def setup_vae_finetuner(args, models: Dict[str, Any]) -> Optional[VAEFineTuner]:
    """
    Set up VAE finetuner with configuration.
    
    Args:
        args: VAE finetuning configuration arguments
        models: Dictionary containing models including the VAE
        
    Returns:
        Configured VAEFineTuner instance or None if setup fails
        
    Raises:
        ValueError: If VAE model is not found in models dictionary
    """
    try:
        if "vae" not in models:
            raise ValueError("VAE model not found in models dictionary")
            
        vae = models["vae"]
        return VAEFineTuner(
            vae=vae,
            device=args.device if hasattr(args, "device") else "cuda",
            mixed_precision=args.mixed_precision if hasattr(args, "mixed_precision") else "no",
            use_amp=args.use_amp if hasattr(args, "use_amp") else False,
            learning_rate=args.learning_rate if hasattr(args, "learning_rate") else 1e-6,
            adam_beta1=args.adam_beta1 if hasattr(args, "adam_beta1") else 0.9,
            adam_beta2=args.adam_beta2 if hasattr(args, "adam_beta2") else 0.999,
            adam_epsilon=args.adam_epsilon if hasattr(args, "adam_epsilon") else 1e-8,
            weight_decay=args.weight_decay if hasattr(args, "weight_decay") else 1e-2,
            max_grad_norm=args.max_grad_norm if hasattr(args, "max_grad_norm") else 1.0,
            gradient_checkpointing=args.gradient_checkpointing if hasattr(args, "gradient_checkpointing") else False,
            use_8bit_adam=args.use_8bit_adam if hasattr(args, "use_8bit_adam") else False,
            use_channel_scaling=args.use_channel_scaling if hasattr(args, "use_channel_scaling") else False,
            adaptive_loss_scale=args.adaptive_loss_scale if hasattr(args, "adaptive_loss_scale") else False,
            kl_weight=args.kl_weight if hasattr(args, "kl_weight") else 0.0,
            perceptual_weight=args.perceptual_weight if hasattr(args, "perceptual_weight") else 0.0,
            min_snr_gamma=args.min_snr_gamma if hasattr(args, "min_snr_gamma") else 5.0,
            initial_scale_factor=args.initial_scale_factor if hasattr(args, "initial_scale_factor") else 1.0,
            decay=args.decay if hasattr(args, "decay") else 0.9999,
            update_after_step=args.update_after_step if hasattr(args, "update_after_step") else 100,
            model_path=args.model_path if hasattr(args, "model_path") else None,
            use_cache=args.use_cache if hasattr(args, "use_cache") else True,
            cache_size=args.cache_size if hasattr(args, "cache_size") else 1000,
        )
    except Exception as e:
        logger.error(f"Failed to setup VAE finetuner: {str(e)}")
        return None