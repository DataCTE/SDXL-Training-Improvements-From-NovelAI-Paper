import os
import math
import time
import logging
import traceback
from collections import defaultdict
from typing import Union, Optional, Any, Tuple, Dict
from dataclasses import dataclass, field
from threading import Lock

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import Adafactor

from .vae_finetuner import VAEFineTuner
from .ema import EMAModel
from data.tag_weighter import TagWeighter

logger = logging.getLogger(__name__)

def setup_optimizer(args, models) -> torch.optim.Optimizer:
    """Set up optimizer with proper configuration and memory optimizations."""
    try:
        params_to_optimize = models["unet"].parameters()
        
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon,
                    weight_decay=args.weight_decay
                )
                logger.info("Using 8-bit Adam optimizer")
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to regular AdamW")
                args.use_8bit_adam = False
        
        if args.use_adafactor:
            optimizer = Adafactor(
                params_to_optimize,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False
            )
            logger.info("Using Adafactor optimizer")
        else:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay
            )
            logger.info("Using AdamW optimizer")
        
        _log_optimizer_config(args)
        return optimizer
        
    except Exception as e:
        logger.error(f"Failed to setup optimizer: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_vae_finetuner(args, models) -> Optional[VAEFineTuner]:
    """Initialize VAE finetuner with proper configuration."""
    try:
        if not args.finetune_vae:
            return None
            
        logger.info("Initializing VAE finetuner")
        vae_finetuner = VAEFineTuner(
            vae=models["vae"],
            device=args.device,
            mixed_precision=args.mixed_precision,
            use_amp=args.use_amp,
            learning_rate=args.vae_learning_rate,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            use_8bit_adam=args.use_8bit_adam,
            use_channel_scaling=args.vae_use_channel_scaling,
            adaptive_loss_scale=args.vae_adaptive_loss_scale,
            kl_weight=args.vae_kl_weight,
            perceptual_weight=args.vae_perceptual_weight,
            min_snr_gamma=args.min_snr_gamma,
            initial_scale_factor=args.vae_initial_scale_factor
        )
        
        _log_vae_config(args)
        return vae_finetuner
        
    except Exception as e:
        logger.error(f"Failed to setup VAE finetuner: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_ema(args, models, device) -> Optional[Any]:
    """Initialize EMA model with proper configuration."""
    try:
        if not args.use_ema:
            return None
            
        logger.info("Initializing EMA model")
        from .ema import EMAModel
        
        ema = EMAModel(
            model=models["unet"],
            model_path=args.model_path,
            device=device,
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            update_every=args.ema_update_every,
            power=args.ema_power,
            min_decay=args.ema_min_decay,
            max_decay=args.ema_max_decay,
            mixed_precision=args.mixed_precision,
            jit_compile=args.enable_compile,
            gradient_checkpointing=args.gradient_checkpointing
        )
        
        _log_ema_config(args)
        return ema
        
    except Exception as e:
        logger.error(f"Failed to setup EMA: {str(e)}")
        logger.error(traceback.format_exc())
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

def setup_tag_weighter(args) -> Optional[Any]:
    """Initialize tag weighting system."""
    try:
        if not args.use_tag_weighting:
            return None
            
        logger.info("Initializing tag weighter")
        from data.tag_weighter import TagWeighter
        
        weighter = TagWeighter(
            base_weight=args.tag_base_weight,
            min_weight=args.tag_min_weight,
            max_weight=args.tag_max_weight,
            window_size=args.tag_window_size
        )
        
        return weighter
        
    except Exception as e:
        logger.error(f"Failed to setup tag weighter: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
    _history: list = field(default_factory=list, init=False)
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

