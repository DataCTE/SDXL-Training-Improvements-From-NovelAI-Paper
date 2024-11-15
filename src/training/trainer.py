import math
import logging
import traceback
from typing import Union, Optional, Any, Dict, List
from dataclasses import dataclass, field
from threading import Lock
import wandb
import torch
import numpy as np
from transformers import Adafactor
from functools import lru_cache
from .vae_finetuner import VAEFineTuner
from data.tag_weighter import TagBasedLossWeighter
from .loss import get_cosine_schedule_with_warmup
from collections import defaultdict
import time
from tqdm import tqdm
from data.dataset import create_dataloader
import os
from .ema import EMAModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _get_optimizer_config(
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float
) -> Dict[str, Any]:
    """Cache optimizer configurations."""
    base_config = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    
    if optimizer_type.lower() == "adamw":
        return {
            **base_config,
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon
        }
    elif optimizer_type.lower() == "adafactor":
        return {
            **base_config,
            "scale_parameter": True,
            "relative_step": False,
            "warmup_init": False
        }
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def setup_optimizer(args, models) -> torch.optim.Optimizer:
    """Set up optimizer with proper configuration and memory optimizations."""
    try:
        # Validate UNet model
        if not hasattr(models["unet"], "parameters"):
            raise ValueError("UNet model is not properly initialized")
            
        # Get trainable parameters with optimized list comprehension
        params_to_optimize = [
            p for p in models["unet"].parameters() 
            if p.requires_grad
        ]
        
        # Add text encoder parameters if needed
        if getattr(args, 'train_text_encoder', False):
            if all(k in models for k in ["text_encoder", "text_encoder_2"]):
                text_params = [
                    p for model_key in ["text_encoder", "text_encoder_2"]
                    for p in models[model_key].parameters()
                    if p.requires_grad
                ]
                params_to_optimize.extend(text_params)
        
        # Validate parameters
        if not params_to_optimize:
            raise ValueError("No trainable parameters found")
            
        num_params = sum(p.numel() for p in params_to_optimize)
        logger.info(f"Setting up optimizer for {num_params:,} parameters")
        
        # Get cached optimizer config
        opt_config = _get_optimizer_config(
            getattr(args, 'optimizer_type', 'adamw'),
            args.learning_rate,
            args.weight_decay,
            args.adam_beta1,
            args.adam_beta2,
            args.adam_epsilon
        )
        
        # Initialize optimizer based on type
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize,
                    **opt_config
                )
                logger.info("Using 8-bit Adam optimizer")
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to regular AdamW")
                args.use_8bit_adam = False
                optimizer = torch.optim.AdamW(
                    params_to_optimize,
                    **opt_config
                )
        elif args.use_adafactor:
            optimizer = Adafactor(
                params_to_optimize,
                lr=opt_config["lr"],
                weight_decay=opt_config["weight_decay"],
                scale_parameter=True,
                relative_step=False,
                warmup_init=False
            )
            logger.info("Using Adafactor optimizer")
        else:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                **opt_config
            )
            logger.info("Using AdamW optimizer")
        
        _log_optimizer_config(args)
        return optimizer
        
    except Exception as e:
        logger.error(f"Failed to setup optimizer: {str(e)}")
        logger.error(f"Models keys available: {list(models.keys())}")
        if "unet" in models:
            logger.error(f"UNet model type: {type(models['unet'])}")
            try:
                logger.error(f"UNet parameters count: {sum(1 for _ in models['unet'].parameters())}")
                logger.error(f"UNet trainable parameters: {sum(1 for p in models['unet'].parameters() if p.requires_grad)}")
            except:
                logger.error("Could not count UNet parameters")
        logger.error(traceback.format_exc())
        raise

@lru_cache(maxsize=32)
def _get_vae_config(
    device: str,
    mixed_precision: str,
    use_amp: bool,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    use_8bit_adam: bool,
    use_channel_scaling: bool = False,
    adaptive_loss_scale: bool = False,
    kl_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    min_snr_gamma: float = 5.0,
    initial_scale_factor: float = 1.0,
    decay: float = 0.9999,
    update_after_step: int = 100,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Cache VAE configuration."""
    return {
        'device': device,
        'mixed_precision': mixed_precision,
        'use_amp': use_amp,
        'learning_rate': learning_rate,
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'adam_epsilon': adam_epsilon,
        'weight_decay': weight_decay,
        'max_grad_norm': max_grad_norm,
        'gradient_checkpointing': gradient_checkpointing,
        'use_8bit_adam': use_8bit_adam,
        'use_channel_scaling': use_channel_scaling,
        'adaptive_loss_scale': adaptive_loss_scale,
        'kl_weight': kl_weight,
        'perceptual_weight': perceptual_weight,
        'min_snr_gamma': min_snr_gamma,
        'initial_scale_factor': initial_scale_factor,
        'decay': decay,
        'update_after_step': update_after_step,
        'model_path': model_path
    }

def setup_vae_finetuner(args, models) -> Optional[VAEFineTuner]:
    """Initialize VAE finetuner with proper configuration."""
    try:
        if not getattr(args, 'finetune_vae', False):
            return None
            
        logger.info("Initializing VAE finetuner")
        vae_config = _get_vae_config(
            device=getattr(args, 'device', 'cuda'),
            mixed_precision=getattr(args, 'mixed_precision', 'no'),
            use_amp=getattr(args, 'use_amp', False),
            learning_rate=getattr(args, 'vae_learning_rate', 1e-6),
            adam_beta1=getattr(args, 'adam_beta1', 0.9),
            adam_beta2=getattr(args, 'adam_beta2', 0.999),
            adam_epsilon=getattr(args, 'adam_epsilon', 1e-8),
            weight_decay=getattr(args, 'weight_decay', 1e-2),
            max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
            gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
            use_8bit_adam=getattr(args, 'use_8bit_adam', False),
            use_channel_scaling=getattr(args, 'vae_use_channel_scaling', False),
            adaptive_loss_scale=getattr(args, 'vae_adaptive_loss_scale', False),
            kl_weight=getattr(args, 'vae_kl_weight', 0.0),
            perceptual_weight=getattr(args, 'vae_perceptual_weight', 0.0),
            min_snr_gamma=getattr(args, 'min_snr_gamma', 5.0),
            initial_scale_factor=getattr(args, 'vae_initial_scale_factor', 1.0),
            decay=getattr(args, 'vae_decay', 0.9999),
            update_after_step=getattr(args, 'vae_update_after_step', 100),
            model_path=getattr(args, 'vae_model_path', None)
        )
        vae_finetuner = VAEFineTuner(
            vae=models["vae"],
            **vae_config
        )
        
        _log_vae_config(args)
        return vae_finetuner
        
    except Exception as e:
        logger.error(f"Failed to setup VAE finetuner: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@lru_cache(maxsize=1)
def _get_ema_config(
    decay: float = 0.9999,
    update_every: int = 10,
    device: Union[str, torch.device] = None,
    model_path: str = None
) -> Dict[str, Any]:
    """Get basic EMA configuration."""
    return {
        'decay': decay,
        'update_every': update_every,
        'device': device,
        'model_path': model_path
    }

def setup_ema(args, model, device=None):
    """Setup EMA model with proper error handling"""
    try:
        # Use provided device or default to CUDA/CPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        decay = getattr(args, 'ema_decay', 0.9999)
        
        # Basic validation
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        
        ema_config = _get_ema_config(
            decay=decay,
            update_every=getattr(args, 'ema_update_every', 10),
            device=device,
            model_path=args.model_path
        )
        
        if args.use_ema:
            logger.info("Creating EMA model...")
            logger.debug(f"EMA config: {ema_config}")  # Add debug logging
            
            ema = EMAModel(
                model,
                **ema_config
            )
            
            logger.info("EMA model created successfully")
            return ema
        return None
        
    except Exception as e:
        logger.error(f"Failed to setup EMA model: {str(e)}")
        logger.debug(f"EMA setup error details: {traceback.format_exc()}")
        raise

def setup_validator(args, models, device, dtype) -> Optional[Any]:
    """Initialize validation components."""
    try:
        if args.skip_validation:
            return None
            
        logger.info("Initializing validator")
        from inference.text_to_image import SDXLInference
        
        validator = SDXLInference(None, device, dtype)
        validator.pipeline = models.get("pipeline")
        
        return validator
        
    except Exception as e:
        logger.error(f"Failed to setup validator: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@lru_cache(maxsize=32)
def _get_tag_weighter_config(
    base_weight: float,
    min_weight: float,
    max_weight: float,
    window_size: int,
    no_cache: bool = False
) -> Dict[str, Any]:
    """Cache tag weighter configuration."""
    return {
        'base_weight': base_weight,
        'min_weight': min_weight,
        'max_weight': max_weight,
        'window_size': window_size,
        'no_cache': no_cache
    }

def setup_tag_weighter(args) -> Optional[Any]:
    """Initialize tag weighting system."""
    try:
        if not getattr(args, 'use_tag_weighting', False):
            return None
            
        logger.info("Initializing tag weighter")
        weighter_config = _get_tag_weighter_config(
            base_weight=getattr(args, 'tag_base_weight', 1.0),
            min_weight=getattr(args, 'min_tag_weight', 0.1),
            max_weight=getattr(args, 'max_tag_weight', 3.0),
            window_size=getattr(args, 'tag_window_size', 100),
            no_cache=getattr(args, 'no_caching', False)
        )
        weighter = TagBasedLossWeighter(config=weighter_config)
        
        return weighter
        
    except Exception as e:
        logger.error(f"Failed to setup tag weighter: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@dataclass
class AverageMeter:
    """Thread-safe average meter with enhanced functionality."""
    name: str
    fmt: str = ':f'
    window_size: Optional[int] = None
    
    def __post_init__(self):
        """Initialize non-pickleable objects after instance creation."""
        self._val: float = 0
        self._sum: float = 0
        self._count: int = 0
        self._avg: float = 0
        self._history: List[float] = []
        # Create lock only when needed, not as a class attribute
        self._lock = None
    
    def _get_lock(self):
        """Lazy initialization of lock object."""
        if self._lock is None:
            self._lock = Lock()
        return self._lock
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._get_lock():
            self._val = 0
            self._sum = 0
            self._count = 0
            self._avg = 0
            self._history.clear()
    
    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """Thread-safe update with support for tensors and arrays."""
        if isinstance(val, (torch.Tensor, np.ndarray)):
            val = float(val.detach().cpu().item() if torch.is_tensor(val) else val.item())
            
        with self._get_lock():
            self._val = val
            self._sum += val * n
            self._count += n
            self._avg = self._sum / self._count
            
            if self.window_size:
                self._history.append(val)
                if len(self._history) > self.window_size:
                    self._history.pop(0)
                self._avg = np.mean(self._history)
    
    @property
    def val(self) -> float:
        """Current value."""
        with self._get_lock():
            return self._val
    
    @property
    def avg(self) -> float:
        """Running average."""
        with self._get_lock():
            return self._avg
    
    @property
    def sum(self) -> float:
        """Total sum."""
        with self._get_lock():
            return self._sum
    
    @property
    def count(self) -> int:
        """Number of updates."""
        with self._get_lock():
            return self._count
            
    def __getstate__(self):
        """Custom state getter for pickling."""
        state = self.__dict__.copy()
        # Don't pickle the lock
        state['_lock'] = None
        return state
    
    def __setstate__(self, state):
        """Custom state setter for unpickling."""
        self.__dict__.update(state)
        # Lock will be recreated when needed via _get_lock()

def _log_optimizer_config(args):
    """Log optimizer configuration details."""
    logger.info(f"Optimizer settings:")
    logger.info(f"- Learning rate: {args.learning_rate}")
    logger.info(f"- Weight decay: {args.weight_decay}")
    if not args.use_adafactor:
        logger.info(f"- Beta1: {args.adam_beta1}")
        logger.info(f"- Beta2: {args.adam_beta2}")
        logger.info(f"- Epsilon: {args.adam_epsilon}")

def _log_vae_config(args):
    """Log VAE configuration details."""
    logger.info(f"VAE Finetuner settings:")
    logger.info(f"- VAE enabled: {args.use_vae}")
    if args.vae_path:
        logger.info(f"- VAE path: {args.vae_path}")
    logger.info(f"- Learning rate: {args.vae_learning_rate}")
    logger.info(f"- Channel scaling: {args.vae_use_channel_scaling}")
    logger.info(f"- Adaptive loss scale: {args.vae_adaptive_loss_scale}")
    logger.info(f"- KL weight: {args.vae_kl_weight}")
    logger.info(f"- Perceptual weight: {args.vae_perceptual_weight}")
    logger.info(f"- Initial scale factor: {args.vae_initial_scale_factor}")
    logger.info(f"- Decay rate: {args.vae_decay}")
    logger.info(f"- Update after step: {args.vae_update_after_step}")

def _log_ema_config(args):
    """Log EMA configuration details."""
    logger.info(f"EMA settings:")
    logger.info(f"- Decay: {args.ema_decay}")
    logger.info(f"- Update after step: {args.ema_update_after_step}")
    logger.info(f"- Update every: {args.ema_update_every}")
    logger.info(f"- Power: {args.ema_power}")
    logger.info(f"- Min/Max decay: {args.ema_min_decay}/{args.ema_max_decay}")

@lru_cache(maxsize=1)
def _get_training_config(
    training_mode: str,
    min_snr_gamma: float,
    sigma_data: float,
    sigma_min: float,
    sigma_max: float,
    scale_method: str,
    scale_factor: float
) -> Dict[str, Any]:
    """Cache training configuration."""
    return {
        "mode": training_mode,
        "min_snr_gamma": min_snr_gamma,
        "sigma_data": sigma_data,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "scale_method": scale_method,
        "scale_factor": scale_factor
    }

def initialize_training_components(args, device, dtype, models):
    """Initialize all training components with proper error handling"""
    components = {}
    
    try:
        # Setup optimizer with validation
        if not models.get("unet"):
            raise ValueError("UNet model not found in models dictionary")
        components["optimizer"] = setup_optimizer(args, models)
        
        # Setup data loader with validation
        required_models = ["vae", "tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"]
        if not all(k in models for k in required_models):
            raise ValueError(f"Missing required models: {[k for k in required_models if k not in models]}")
            
        components["train_dataloader"] = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_caching_latents=args.no_caching,
            all_ar=args.all_ar,
            cache_dir=args.cache_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"]
        )
        
        # Setup learning rate scheduler
        num_training_steps = args.num_epochs * len(components["train_dataloader"])
        components["lr_scheduler"] = get_cosine_schedule_with_warmup(
            optimizer=components["optimizer"],
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Setup optional components with parallel initialization
        optional_components = {
            "ema": (setup_ema, args.use_ema, (args, models["unet"])),
            "tag_weighter": (setup_tag_weighter, args.use_tag_weighting, (args,)),
            "vae_finetuner": (setup_vae_finetuner, args.finetune_vae, (args, models))
        }
        
        components.update({
            name: setup_func(*setup_args) if use_flag else None
            for name, (setup_func, use_flag, setup_args) in optional_components.items()
        })
        
        # Cache and set training configuration
        components["training_config"] = _get_training_config(
            training_mode=getattr(args, 'training_mode', 'v_prediction'),
            min_snr_gamma=getattr(args, 'min_snr_gamma', 5.0),
            sigma_data=getattr(args, 'sigma_data', 1.0),
            sigma_min=getattr(args, 'sigma_min', 0.002),
            sigma_max=getattr(args, 'sigma_max', 80.0),
            scale_method=getattr(args, 'scale_method', 'v'),
            scale_factor=getattr(args, 'scale_factor', 1.0)
        )
        
        # Validate components
        _validate_components(components)
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize training components: {str(e)}")
        logger.error(traceback.format_exc())
        _cleanup_failed_initialization(components)
        raise

def _validate_components(components: Dict[str, Any]) -> None:
    """Validate initialized components."""
    required = ["optimizer", "train_dataloader", "lr_scheduler", "training_config"]
    if not all(k in components for k in required):
        raise ValueError(f"Missing required components: {[k for k in required if k not in components]}")

def _cleanup_failed_initialization(components: Dict[str, Any]) -> None:
    """Clean up resources in case of failed initialization."""
    try:
        # Close data loader if it was created
        if "train_dataloader" in components:
            try:
                components["train_dataloader"].dataset.cleanup()
            except:
                pass
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@torch.no_grad()
def train_epoch(epoch: int, args, models, components, device, dtype, wandb_run, global_step: int) -> Dict[str, float]:
    """Execute single training epoch with proper logging and memory management."""
    try:
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        epoch_metrics = train(args, models, components, device, dtype)
        
        # Log metrics if wandb is enabled
        if wandb_run:
            log_epoch_metrics(wandb_run, epoch_metrics, epoch, global_step)
            log_model_gradients(models["unet"], step=global_step)
            log_memory_stats(step=global_step)
        
        return epoch_metrics
        
    except Exception as e:
        logger.error(f"Error in training epoch {epoch}: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train(args, models, components, device, dtype) -> Dict[str, float]:
    """Execute training steps with proper error handling and logging."""
    # Create new metrics for each training run
    metrics = defaultdict(lambda: AverageMeter(name="default"))
    models["unet"].train()
    
    start_time = time.time()
    data_time = AverageMeter("data_time")
    batch_time = AverageMeter("batch_time")
    
    progress_bar = tqdm(
        components["train_dataloader"],
        desc=f"Training",
        dynamic_ncols=True,
        leave=False
    )
    
    try:
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Update timing
                data_time.update(time.time() - start_time)
                
                # Execute training step
                batch_metrics = train_step(
                    args=args,
                    models=models,
                    components=components,
                    batch=batch,
                    batch_idx=batch_idx,
                    device=device,
                    dtype=dtype
                )
                
                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k].update(v)
                
                # Update progress bar
                progress_bar.set_postfix({
                    k: f"{v.avg:.4f}" for k, v in metrics.items()
                })
                
                # Update timing
                batch_time.update(time.time() - start_time)
                start_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
        return {k: v.avg for k, v in metrics.items()}
        
    finally:
        progress_bar.close()

def log_epoch_metrics(wandb_run, metrics: Dict[str, float], epoch: int, global_step: int) -> None:
    """Log epoch metrics to W&B with proper error handling."""
    try:
        # Prepare metrics for logging
        log_dict = {
            f"train/{k}": v for k, v in metrics.items()
        }
        
        # Add epoch info
        log_dict.update({
            "train/epoch": epoch,
            "train/global_step": global_step
        })
        
        # Log to W&B
        wandb_run.log(log_dict, step=global_step)
        
        # Log to console
        logger.info(
            f"Epoch {epoch} metrics: " + 
            ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
        
    except Exception as e:
        logger.error(f"Failed to log epoch metrics: {str(e)}")
        logger.error(traceback.format_exc())

def log_model_gradients(model: torch.nn.Module, step: int) -> None:
    """Log model gradient statistics to W&B."""
    try:
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Compute gradient statistics
                grad = param.grad.detach()
                grad_dict.update({
                    f"gradients/{name}/mean": grad.mean().item(),
                    f"gradients/{name}/std": grad.std().item(),
                    f"gradients/{name}/norm": grad.norm().item(),
                    f"gradients/{name}/max": grad.max().item(),
                    f"gradients/{name}/min": grad.min().item()
                })
                
                # Log histogram if available
                try:
                    import wandb
                    grad_dict[f"gradients/{name}/hist"] = wandb.Histogram(grad.cpu().numpy())
                except:
                    pass
        
        # Log to W&B
        wandb.log(grad_dict, step=step)
        
    except Exception as e:
        logger.error(f"Failed to log model gradients: {str(e)}")
        logger.error(traceback.format_exc())

@lru_cache(maxsize=1)
def _get_memory_stats_format() -> Dict[str, str]:
    """Cache memory statistics format strings."""
    return {
        "allocated": "Allocated: {:.1f}MB",
        "cached": "Cached: {:.1f}MB",
        "max_allocated": "Max Allocated: {:.1f}MB",
        "max_cached": "Max Cached: {:.1f}MB",
        "active": "Active: {:.1f}MB",
        "inactive": "Inactive: {:.1f}MB",
        "fragmentation": "Fragmentation: {:.1f}"
    }

def log_memory_stats(step: int) -> None:
    """Log CUDA memory statistics to W&B."""
    try:
        if not torch.cuda.is_available():
            return
            
        # Get format strings
        format_strings = _get_memory_stats_format()
            
        # Get memory statistics
        memory_stats = {
            "memory/allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "memory/cached": torch.cuda.memory_reserved() / 1024**2,      # MB
            "memory/max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "memory/max_cached": torch.cuda.max_memory_reserved() / 1024**2
        }
        
        # Get per-device statistics
        for i in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(i)
            memory_stats.update({
                f"memory/device_{i}/active": stats["active_bytes.all.current"] / 1024**2,
                f"memory/device_{i}/inactive": stats["inactive_split_bytes.all.current"] / 1024**2,
                f"memory/device_{i}/fragmentation": stats.get("fragmentation.all.current", 0)
            })
        
        # Log to W&B
        wandb.log(memory_stats, step=step)
        
        # Format and log to console using cached format strings
        formatted_stats = []
        for k, v in memory_stats.items():
            # Get the base stat name without the path
            stat_name = k.split('/')[-1]
            if stat_name in format_strings:
                formatted_stats.append(format_strings[stat_name].format(v))
            else:
                formatted_stats.append(f"{k}: {v:.1f}MB")
        
        logger.debug("Memory stats: " + ", ".join(formatted_stats))
        
    except Exception as e:
        logger.error(f"Failed to log memory stats: {str(e)}")
        logger.error(traceback.format_exc())

def train_step(args, models, components, batch, batch_idx: int, device, dtype) -> Dict[str, float]:
    """Execute single training step with proper error handling."""
    try:
        # Move batch to device
        batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Zero gradients
        components["optimizer"].zero_grad(set_to_none=True)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            # Get loss from model
            loss = models["unet"](
                x_0=batch["latents"],
                sigma=batch["sigmas"],
                text_embeddings=batch["text_embeddings"],
                added_cond_kwargs=batch.get("added_cond_kwargs"),
                sigma_data=args.sigma_data,
                tag_weighter=components.get("tag_weighter"),
                batch_tags=batch.get("tags"),
                min_snr_gamma=args.min_snr_gamma,
                rescale_cfg=args.rescale_cfg,
                rescale_multiplier=args.rescale_multiplier,
                scale_method=args.scale_method,
                use_tag_weighting=args.use_tag_weighting,
                device=device,
                dtype=dtype
            )
            
            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if args.mixed_precision:
            components["scaler"].scale(loss).backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                components["scaler"].unscale_(components["optimizer"])
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
                components["scaler"].step(components["optimizer"])
                components["scaler"].update()
                components["lr_scheduler"].step()
        else:
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
                components["optimizer"].step()
                components["lr_scheduler"].step()
        
        # Update EMA model if enabled
        if components.get("ema") is not None and (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            components["ema"].step(models["unet"])
        
        # Update VAE if enabled
        if components.get("vae_finetuner") is not None:
            vae_metrics = components["vae_finetuner"].train_step(batch)
        else:
            vae_metrics = {}
        
        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "lr": components["optimizer"].param_groups[0]["lr"],
            "grad_norm": get_grad_norm(models["unet"]),
            **vae_metrics
        }
        
        if components.get("ema"):
            metrics["ema_decay"] = components["ema"].cur_decay_value
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in training step: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm with proper error handling."""
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    except Exception as e:
        logger.error(f"Failed to calculate gradient norm: {str(e)}")
        return 0.0

def run_validation(
    args,
    models,
    components,
    device,
    dtype,
    global_step: int
) -> Dict[str, float]:
    """Run validation with proper error handling and metrics tracking."""
    try:
        # Initialize inference pipeline if not already in components
        if "inference" not in components:
            from inference.text_to_image import SDXLInference
            
            components["inference"] = SDXLInference(
                device=device,
                dtype=dtype,
                use_v_prediction=args.training_mode == "v_prediction",
                use_resolution_binning=True,
                use_zero_terminal_snr=args.use_ztsnr,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                sigma_data=args.sigma_data,
                min_snr_gamma=args.min_snr_gamma,
                noise_offset=args.noise_offset
            )
            
            # Set models from training
            components["inference"].pipeline.unet = models["unet"]
            components["inference"].pipeline.vae = models["vae"]
            components["inference"].pipeline.text_encoder = models["text_encoder"]
            components["inference"].pipeline.text_encoder_2 = models["text_encoder_2"]
            components["inference"].pipeline.tokenizer = models["tokenizer"]
            components["inference"].pipeline.tokenizer_2 = models["tokenizer_2"]

        # Set models to eval mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.eval()

        try:
            with torch.no_grad():
                # Run validation using configured prompts
                validation_metrics = components["inference"].run_validation(
                    prompts=args.validation_prompts,
                    output_dir=os.path.join(args.output_dir, f"validation_{global_step}"),
                    log_to_wandb=args.use_wandb,
                    guidance_scale=args.validation_guidance_scale,
                    num_inference_steps=args.validation_num_steps,
                    height=args.validation_height,
                    width=args.validation_width,
                    num_images_per_prompt=args.validation_images_per_prompt,
                    seed=args.validation_seed,
                    rescale_cfg=args.rescale_cfg,
                    scale_method=args.scale_method,
                    rescale_multiplier=args.rescale_multiplier
                )

                # Add EMA metrics if enabled
                if components.get("ema"):
                    validation_metrics["validation/ema_decay"] = components["ema"].cur_decay_value

                # Log validation results
                logger.info(
                    "Validation Results: " +
                    ", ".join(f"{k}: {v:.4f}" for k, v in validation_metrics.items())
                )

                return validation_metrics

        except Exception as e:
            logger.error(f"Validation inference failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    except Exception as e:
        logger.error(f"Validation setup failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

    finally:
        # Set models back to train mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.train()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()