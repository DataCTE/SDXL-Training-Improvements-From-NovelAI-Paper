import logging
import sys
import os
from pathlib import Path
import torch
import wandb
import time
import traceback
import numpy as np
from typing import Dict, Any, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Configure logging settings for the application
    
    Args:
        log_dir (str, optional): Directory to save log files
        log_level: Logging level (default: INFO)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_dir is specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / "training.log",
            mode='a'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def log_system_info():
    """Log system and environment information"""
    logger = logging.getLogger(__name__)
    
    # PyTorch version and CUDA info
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU device: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.2fGB", torch.cuda.get_device_properties(0).total_memory / 1e9)

def log_training_setup(args, models, train_components):
    """
    Log training configuration and setup details
    
    Args:
        args: Training arguments
        models: Dictionary of model components
        train_components: Dictionary of training components
    """
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info("\n=== Training Configuration ===")
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)
    
    # Log model information
    logger.info("\n=== Model Information ===")
    for name, model in models.items():
        if hasattr(model, 'parameters'):
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("%s:", name)
            logger.info("  Total parameters: %d", num_params)
            logger.info("  Trainable parameters: %d", trainable_params)
    
    # Log dataset information
    if 'dataset' in train_components:
        logger.info("\n=== Dataset Information ===")
        logger.info("Number of training samples: %d", len(train_components['dataset']))
        logger.info("Batch size: %d", args.batch_size)
        logger.info("Steps per epoch: %d", len(train_components['train_dataloader']))

def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return "GPU Memory: %.2fGB allocated, %.2fGB reserved" % (allocated, reserved)
    return "GPU not available"

def setup_wandb(args):
    """
    Initialize Weights & Biases logging with optimized configuration
    
    Args:
        args: Training arguments containing W&B configuration
        
    Returns:
        wandb.Run or None: Initialized W&B run
    """
    if not args.use_wandb:
        return None
        
    try:
        # Use more robust wandb initialization with additional configuration
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"sdxl_training_%d" % time.time(),
            config=vars(args),
            resume='allow',  # Allow resuming previous runs
            save_code=True,  # Save code snapshot for reproducibility
            tags=['sdxl', 'training'],
            # Reduce network overhead
            settings=wandb.Settings(
                _disable_stats=True,  # Disable resource-intensive stats collection
                _save_requirements=False  # Disable saving requirements
            )
        )
        
        # Define loss metrics
        wandb.define_metric("loss/*", summary="min", step_metric="global_step")
        wandb.define_metric("train/loss", summary="min", step_metric="global_step")
        wandb.define_metric("val/loss", summary="min", step_metric="global_step")
        
        # Define learning rate metrics
        wandb.define_metric("lr/*", summary="last", step_metric="global_step")
        wandb.define_metric("train/lr", summary="last", step_metric="global_step")
        
        # Define epoch metrics
        wandb.define_metric("epoch", summary="max")
        wandb.define_metric("epoch/*", step_metric="epoch")
        
        # Define validation metrics
        wandb.define_metric("validation/*", summary="last", step_metric="epoch")
        
        # Define performance metrics
        wandb.define_metric("performance/*", summary="mean", step_metric="global_step")
        wandb.define_metric("memory/*", summary="max", step_metric="global_step")
        
        # Define gradient metrics
        wandb.define_metric("gradients/*", summary="mean", step_metric="global_step")
        
        logger.info("Initialized W&B run: %s", run.name)
        return run
        
    except Exception as e:
        logger.error("Failed to initialize W&B: %s", str(e))
        return None

def log_metrics_batch(metrics_dict, step=None):
    """
    Efficiently log metrics in batches to reduce API calls
    
    Args:
        metrics_dict (dict): Dictionary of metrics to log
        step (int, optional): Global training step
    """
    if not wandb.run:
        return
        
    try:
        wandb.log(metrics_dict, step=step)
    except Exception as e:
        logger.warning("Failed to log metrics batch: %s", str(e))

def log_model_gradients(model, step):
    """
    Log model gradient statistics
    
    Args:
        model: PyTorch model
        step (int): Global training step
    """
    if not wandb.run:
        return
        
    try:
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict["gradients/%s/mean" % name] = param.grad.mean().item()
                grad_dict["gradients/%s/std" % name] = param.grad.std().item()
                grad_dict["gradients/%s/norm" % name] = param.grad.norm().item()
        
        if grad_dict:
            wandb.log(grad_dict, step=step)
    except Exception as e:
        logger.warning("Failed to log model gradients: %s", str(e))

def log_memory_stats(step):
    """
    Log GPU memory statistics
    
    Args:
        step (int): Global training step
    """
    if not wandb.run or not torch.cuda.is_available():
        return
        
    try:
        memory_stats = {
            "memory/allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "memory/reserved": torch.cuda.memory_reserved() / 1024**2,    # MB
            "memory/max_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
        }
        wandb.log(memory_stats, step=step)
    except Exception as e:
        logger.warning("Failed to log memory stats: %s", str(e))

def cleanup_wandb(run):
    """
    Cleanup W&B run with enhanced error handling
    
    Args:
        run: W&B run to cleanup
    """
    if run is not None:
        try:
            # Log system info before closing
            run.log({
                "system/final_gpu_memory": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "system/peak_gpu_memory": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            })
            run.finish(quiet=True)  # Suppress verbose output
        except Exception as e:
            logger.error("Error closing W&B run: %s", str(e))

def log_epoch_metrics(wandb_run, metrics: Dict[str, float], epoch: int, global_step: int) -> None:
    """Log epoch metrics to W&B with proper error handling."""
    try:
        # Prepare metrics for logging
        log_dict = {"train/%s" % k: v for k, v in metrics.items()}
        log_dict.update({"train/epoch": epoch, "train/global_step": global_step})
        
        # Log to W&B
        wandb_run.log(log_dict, step=global_step)
        
        # Log to console
        logger.info("Epoch %d metrics: %s", epoch, ", ".join("%s: %.6f" % (k, v) for k, v in metrics.items()))
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to log epoch metrics: %s", str(e))
        logger.error(traceback.format_exc())

def log_model_gradients(model: torch.nn.Module, step: int) -> None:
    """Log model gradient statistics to W&B."""
    try:
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_dict.update({
                    "gradients/%s/mean" % name: grad.mean().item(),
                    "gradients/%s/std" % name: grad.std().item(),
                    "gradients/%s/norm" % name: grad.norm().item(),
                    "gradients/%s/max" % name: grad.max().item(),
                    "gradients/%s/min" % name: grad.min().item(),
                })

                try:
                    grad_dict["gradients/%s/hist" % name] = wandb.Histogram(
                        grad.cpu().numpy()
                    )
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Failed to create histogram for %s: %s", name, str(e))

        wandb.log(grad_dict, step=step)
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to log model gradients: %s", str(e))
        logger.error(traceback.format_exc())

@lru_cache(maxsize=1)
def _get_memory_stats_format() -> Dict[str, str]:
    """Cache memory statistics format strings."""
    return {
        "allocated": "Allocated: %.1fMB",
        "cached": "Cached: %.1fMB",
        "max_allocated": "Max Allocated: %.1fMB",
        "max_cached": "Max Cached: %.1fMB",
        "active": "Active: %.1fMB",
        "inactive": "Inactive: %.1fMB",
        "fragmentation": "Fragmentation: %.1f",
    }

def log_memory_stats(step: int) -> None:
    """Log CUDA memory statistics to W&B."""
    try:
        if not torch.cuda.is_available():
            return

        format_strings = _get_memory_stats_format()
        memory_stats = {
            "memory/allocated": torch.cuda.memory_allocated() / 1024**2,
            "memory/cached": torch.cuda.memory_reserved() / 1024**2,
            "memory/max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "memory/max_cached": torch.cuda.max_memory_reserved() / 1024**2,
        }

        for i in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(i)
            memory_stats.update({
                "memory/device_%d/active" % i: stats["active_bytes.all.current"] / 1024**2,
                "memory/device_%d/inactive" % i: stats["inactive_split_bytes.all.current"] / 1024**2,
                "memory/device_%d/fragmentation" % i: stats.get("fragmentation.all.current", 0),
            })

        wandb.log(memory_stats, step=step)

        formatted_stats = []
        for k, v in memory_stats.items():
            stat_name = k.split("/")[-1]
            if stat_name in format_strings:
                formatted_stats.append(format_strings[stat_name] % v)
            else:
                formatted_stats.append("%s: %.1fMB" % (k, v))

        logger.debug("Memory stats: %s", ", ".join(formatted_stats))
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to log memory stats: %s", str(e))
        logger.error(traceback.format_exc())

def log_optimizer_config(args) -> None:
    """Log optimizer configuration details."""
    logger.info("Optimizer config:")
    logger.info("  Type: %s", args.optimizer_type)
    logger.info("  Learning rate: %f", args.learning_rate)
    logger.info("  Weight decay: %f", args.weight_decay)
    logger.info("  Adam beta1: %f", args.adam_beta1)
    logger.info("  Adam beta2: %f", args.adam_beta2)
    logger.info("  Adam epsilon: %f", args.adam_epsilon)

def log_vae_config(args) -> None:
    """Log VAE configuration details."""
    logger.info("VAE config:")
    logger.info("  Learning rate: %f", args.vae_learning_rate)
    logger.info("  Weight decay: %f", args.vae_weight_decay)
    logger.info("  Adam beta1: %f", args.vae_adam_beta1)
    logger.info("  Adam beta2: %f", args.vae_adam_beta2)
    logger.info("  Adam epsilon: %f", args.vae_adam_epsilon)
    logger.info("  Max grad norm: %f", args.vae_max_grad_norm)
    logger.info("  Gradient checkpointing: %s", args.vae_gradient_checkpointing)

def log_ema_config(args) -> None:
    """Log EMA configuration details."""
    logger.info("EMA config:")
    logger.info("  Decay: %f", args.ema_decay)
    logger.info("  Update every: %d", args.ema_update_every)
