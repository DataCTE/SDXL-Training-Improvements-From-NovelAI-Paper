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
def _get_vae_config(args) -> Dict[str, Any]:
    """Cache VAE configuration."""
    return {
        'device': args.device,
        'mixed_precision': args.mixed_precision,
        'use_amp': args.use_amp,
        'learning_rate': args.vae_learning_rate,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'adam_epsilon': args.adam_epsilon,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'gradient_checkpointing': args.gradient_checkpointing,
        'use_8bit_adam': args.use_8bit_adam,
        'use_channel_scaling': args.vae_use_channel_scaling,
        'adaptive_loss_scale': args.vae_adaptive_loss_scale,
        'kl_weight': args.vae_kl_weight,
        'perceptual_weight': args.vae_perceptual_weight,
        'min_snr_gamma': args.min_snr_gamma,
        'initial_scale_factor': args.vae_initial_scale_factor
    }

def setup_vae_finetuner(args, models) -> Optional[VAEFineTuner]:
    """Initialize VAE finetuner with proper configuration."""
    try:
        if not args.finetune_vae:
            return None
            
        logger.info("Initializing VAE finetuner")
        vae_config = _get_vae_config(args)
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
    device: str = 'auto',
    min_value: float = 0.0
) -> Dict[str, Any]:
    """Get basic EMA configuration."""
    return {
        'decay': decay,
        'update_every': update_every,
        'device': device,
        'min_value': min_value
    }

def _validate_ema_params(decay: float, min_value: float) -> None:
    """Validate EMA parameters."""
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
    if not 0.0 <= min_value <= 1.0:
        raise ValueError(f"EMA min_value must be between 0 and 1, got {min_value}")

def setup_ema(args, model, device=None):
    """Setup EMA model with proper error handling"""
    try:
        # Validate device
        if device is None:
            device = getattr(args, 'ema_device', 'auto')
        if isinstance(device, str) and device != 'auto':
            device = torch.device(device)
            
        decay = getattr(args, 'ema_decay', 0.9999)
        min_value = getattr(args, 'ema_min_value', 0.0)
        
        # Validate parameters
        _validate_ema_params(decay, min_value)
        
        ema_config = _get_ema_config(
            decay=decay,
            update_every=getattr(args, 'ema_update_every', 10),
            device=device,
            min_value=min_value
        )
        
        if args.use_ema:
            logger.info("Creating EMA model...")
            logger.debug(f"EMA config: {ema_config}")  # Add debug logging
            
            ema = EMAModel(
                model,
                **ema_config
            )
            
            # Configure warmup after creation if supported
            if hasattr(ema, 'set_warmup') and getattr(args, 'ema_use_warmup', False):
                ema.set_warmup(
                    warmup_steps=getattr(args, 'ema_warmup_steps', 2000),
                    update_after_step=getattr(args, 'ema_update_after_step', 0)
                )
            
            logger.info("EMA model created successfully")
            return ema
        return None
        
    except Exception as e:
        logger.error(f"Failed to setup EMA: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
def _get_tag_weighter_config(args) -> Dict[str, Any]:
    """Cache tag weighter configuration."""
    return {
        'base_weight': args.tag_base_weight,
        'min_weight': args.tag_min_weight,
        'max_weight': args.tag_max_weight,
        'window_size': args.tag_window_size,
        'no_cache': getattr(args, 'no_caching', False)
    }

def setup_tag_weighter(args) -> Optional[Any]:
    """Initialize tag weighting system."""
    try:
        if not args.use_tag_weighting:
            return None
            
        logger.info("Initializing tag weighter")
        weighter_config = _get_tag_weighter_config(args)
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
    
    _val: float = field(default=0, init=False)
    _sum: float = field(default=0, init=False)
    _count: int = field(default=0, init=False)
    _avg: float = field(default=0, init=False)
    _history: List[float] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._val = 0
            self._sum = 0
            self._count = 0
            self._avg = 0
            self._history.clear()
    
    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """Thread-safe update with support for tensors and arrays."""
        if isinstance(val, (torch.Tensor, np.ndarray)):
            val = float(val.detach().cpu().item() if torch.is_tensor(val) else val.item())
            
        with self._lock:
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
        with self._lock:
            return self._val
    
    @property
    def avg(self) -> float:
        """Running average."""
        with self._lock:
            return self._avg
    
    @property
    def sum(self) -> float:
        """Total sum."""
        with self._lock:
            return self._sum
    
    @property
    def count(self) -> int:
        """Number of updates."""
        with self._lock:
            return self._count

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
    logger.info(f"- Learning rate: {args.vae_learning_rate}")
    logger.info(f"- Channel scaling: {args.vae_use_channel_scaling}")
    logger.info(f"- Adaptive loss scale: {args.vae_adaptive_loss_scale}")
    logger.info(f"- KL weight: {args.vae_kl_weight}")
    logger.info(f"- Perceptual weight: {args.vae_perceptual_weight}")
    logger.info(f"- Initial scale factor: {args.vae_initial_scale_factor}")

def _log_ema_config(args):
    """Log EMA configuration details."""
    logger.info(f"EMA settings:")
    logger.info(f"- Decay: {args.ema_decay}")
    logger.info(f"- Update after step: {args.ema_update_after_step}")
    logger.info(f"- Update every: {args.ema_update_every}")
    logger.info(f"- Power: {args.ema_power}")
    logger.info(f"- Min/Max decay: {args.ema_min_decay}/{args.ema_max_decay}")

@lru_cache(maxsize=32)
def _get_training_config(args) -> Dict[str, Any]:
    """Cache training configuration."""
    return {
        "mode": args.training_mode,
        "min_snr_gamma": args.min_snr_gamma,
        "sigma_data": args.sigma_data,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "scale_method": args.scale_method,
        "scale_factor": args.scale_factor
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
        components["training_config"] = _get_training_config(args)
        
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
def train_epoch(epoch: int, args, models, components, device, dtype, wandb_run, global_step: int):
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

def log_memory_stats(step: int) -> None:
    """Log CUDA memory statistics to W&B."""
    try:
        if not torch.cuda.is_available():
            return
            
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
        
        # Log to console
        logger.debug(
            "Memory stats: " + 
            ", ".join(f"{k}: {v:.1f}MB" for k, v in memory_stats.items())
        )
        
    except Exception as e:
        logger.error(f"Failed to log memory stats: {str(e)}")
        logger.error(traceback.format_exc())

@lru_cache(maxsize=1)
def _get_memory_stats_format() -> Dict[str, str]:
    """Cache memory statistics format strings."""
    return {
        "allocated": "Allocated: {:.1f}MB",
        "cached": "Cached: {:.1f}MB",
        "max_allocated": "Max Allocated: {:.1f}MB",
        "max_cached": "Max Cached: {:.1f}MB"
    }

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