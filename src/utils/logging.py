"""Logging utilities for SDXL training.

This module provides utilities for setting up logging and tracking training progress.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import time
from datetime import datetime
import torch
import wandb
import numpy as np
from collections import defaultdict

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(levelname)s - %(name)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    enable_console: bool = True,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> None:
    """Configure logging settings for the application.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (default: INFO)
        enable_console: Whether to enable console output
        log_format: Format string for log messages
    """
    # Create formatter
    file_formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with colored output
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)
    
    # Add file handler if log_dir is specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Create a symlink to latest log
        latest_log = log_dir / "latest.log"
        if latest_log.exists():
            latest_log.unlink()
        latest_log.symlink_to(log_file.name)

def log_training_setup(
    args: Any,
    models: Dict[str, Any],
    train_components: Dict[str, Any],
) -> None:
    """Log training configuration and setup details.
    
    Args:
        args: Training arguments
        models: Dictionary of model components
        train_components: Dictionary of training components
    """
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info("=== Training Configuration ===")
    config_dict = {k: str(v) for k, v in vars(args).items()}
    logger.info("Arguments:\n%s", json.dumps(config_dict, indent=2))
    
    # Log model information
    logger.info("\n=== Model Information ===")
    for name, model in models.items():
        if hasattr(model, "parameters"):
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("%s:", name)
            logger.info("  Total parameters: %d", num_params)
            logger.info("  Trainable parameters: %d", trainable_params)
    
    # Log dataset information
    if "dataset" in train_components:
        logger.info("\n=== Dataset Information ===")
        logger.info("Number of training samples: %d", len(train_components["dataset"]))
        logger.info("Batch size: %d", args.batch_size)
        logger.info("Steps per epoch: %d", len(train_components["train_dataloader"]))
        
    # Log optimizer information
    if "optimizer" in train_components:
        logger.info("\n=== Optimizer Information ===")
        optimizer = train_components["optimizer"]
        logger.info("Optimizer: %s", optimizer.__class__.__name__)
        logger.info("Learning rate: %f", args.optimizer.learning_rate)
        
    # Log system information
    logger.info("\n=== System Information ===")
    logger.info("Mixed precision: %s", args.mixed_precision)
    logger.info("Gradient checkpointing: %s", args.gradient_checkpointing)
    logger.info("Number of workers: %d", args.num_workers)

class DetailedLogger:
    """Enhanced logger with comprehensive value tracking."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        use_wandb: bool = False,
        log_frequency: int = 1,
        detailed_grad_logging: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different log files for different types of data
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.gradients_file = self.log_dir / "gradients.jsonl"
        self.model_states_file = self.log_dir / "model_states.jsonl"
        self.training_steps_file = self.log_dir / "training_steps.jsonl"
        
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        self.use_wandb = use_wandb
        self.log_frequency = log_frequency
        self.detailed_grad_logging = detailed_grad_logging
        
        # Statistics tracking
        self.step_times = []
        self.loss_history = defaultdict(list)
        self.grad_norms = defaultdict(list)
        
    def log_forward_pass(
        self,
        loss: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> None:
        """Log detailed forward pass information."""
        if step % self.log_frequency != 0:
            return
            
        forward_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": loss.item(),
            "loss_components": self._extract_loss_components(loss),
            "model_output_stats": self._get_tensor_stats(model_outputs),
            "batch_stats": self._get_tensor_stats(batch),
            "memory_stats": self._get_memory_stats()
        }
        
        # Log to file
        with open(self.training_steps_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(forward_data) + "\n")
            
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "forward_pass": forward_data,
                **{f"loss/{k}": v for k, v in forward_data["loss_components"].items()},
                **{f"memory/{k}": v for k, v in forward_data["memory_stats"].items()}
            }, step=step)

    def log_backward_pass(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_norm: float,
        step: int
    ) -> None:
        """Log detailed gradient and optimizer information."""
        if not self.detailed_grad_logging or step % self.log_frequency != 0:
            return
            
        grad_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "total_grad_norm": grad_norm,
            "parameter_grad_norms": self._get_parameter_grad_stats(model),
            "optimizer_stats": self._get_optimizer_stats(optimizer),
            "memory_stats": self._get_memory_stats()
        }
        
        # Log to file
        with open(self.gradients_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(grad_data) + "\n")
            
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "backward_pass": grad_data,
                "grad_norm": grad_norm,
                **{f"grads/{k}": v for k, v in grad_data["parameter_grad_norms"].items()}
            }, step=step)

    def log_training_step(
        self,
        loss: float,
        metrics: Dict[str, float],
        learning_rate: float,
        step: int,
        epoch: int,
        batch_idx: int,
        step_time: float
    ) -> None:
        """Log comprehensive training step information."""
        step_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "loss": loss,
            "metrics": metrics,
            "learning_rate": learning_rate,
            "step_time": step_time,
            "steps_per_second": 1.0 / step_time if step_time > 0 else 0,
            "memory_stats": self._get_memory_stats(),
            "time_breakdown": self._get_time_breakdown()
        }
        
        # Update statistics
        self.step_times.append(step_time)
        self.loss_history["total"].append(loss)
        
        # Log to file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_data) + "\n")
            
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "training": step_data,
                "loss": loss,
                "learning_rate": learning_rate,
                "step_time": step_time,
                **metrics
            }, step=step)

    def log_model_state(
        self,
        model: torch.nn.Module,
        epoch: int,
        step: int
    ) -> None:
        """Log detailed model state information."""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "parameter_stats": self._get_parameter_stats(model),
            "memory_stats": self._get_memory_stats(),
            "training_stats": self._get_training_stats()
        }
        
        # Log to file
        with open(self.model_states_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(state_data) + "\n")
            
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "model_state": state_data,
                **{f"params/{k}": v for k, v in state_data["parameter_stats"].items()}
            }, step=step)

    def _get_tensor_stats(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Get detailed statistics for tensors."""
        stats = {}
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                stats[name] = {
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item(),
                    "min": tensor.min().item(),
                    "max": tensor.max().item(),
                    "norm": tensor.norm().item(),
                    "shape": list(tensor.shape)
                }
        return stats

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get detailed GPU memory statistics."""
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "cached": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "max_cached": torch.cuda.max_memory_reserved() / 1024**2
        }

    def _get_parameter_grad_stats(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """Get detailed gradient statistics for model parameters."""
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats[name] = {
                    "grad_mean": param.grad.mean().item(),
                    "grad_std": param.grad.std().item(),
                    "grad_norm": param.grad.norm().item(),
                    "param_norm": param.norm().item(),
                    "grad_param_ratio": (param.grad.norm() / param.norm()).item()
                }
        return stats

    def _get_optimizer_stats(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Get detailed optimizer statistics."""
        stats = {
            "optimizer_type": optimizer.__class__.__name__,
            "param_groups": []
        }
        
        for group in optimizer.param_groups:
            group_stats = {
                "lr": group["lr"],
                "momentum": group.get("momentum", 0),
                "weight_decay": group.get("weight_decay", 0),
                "params_count": len(group["params"])
            }
            stats["param_groups"].append(group_stats)
            
        return stats

    def _get_training_stats(self) -> Dict[str, Any]:
        """Get accumulated training statistics."""
        return {
            "avg_step_time": np.mean(self.step_times[-100:]),
            "loss_moving_avg": np.mean(self.loss_history["total"][-100:]),
            "loss_std": np.std(self.loss_history["total"][-100:]),
            "grad_norm_moving_avg": {
                k: np.mean(v[-100:]) for k, v in self.grad_norms.items()
            }
        }

    def _extract_loss_components(self, loss: torch.Tensor) -> Dict[str, float]:
        """Extract individual loss components if available."""
        if hasattr(loss, "components"):
            return {k: v.item() for k, v in loss.components.items()}
        return {"total": loss.item()}

    def _get_parameter_stats(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """Get detailed statistics for model parameters.
        
        Args:
            model: The PyTorch model to analyze
            
        Returns:
            Dictionary containing parameter statistics
        """
        stats = {}
        for name, param in model.named_parameters():
            if param is not None:
                stats[name] = {
                    "mean": param.mean().item(),
                    "std": param.std().item(),
                    "min": param.min().item(),
                    "max": param.max().item(),
                    "norm": param.norm().item(),
                    "shape": list(param.shape),
                    "requires_grad": param.requires_grad,
                    "dtype": str(param.dtype),
                    "device": str(param.device)
                }
                
                # Add gradient statistics if available
                if param.grad is not None:
                    stats[name].update({
                        "grad_mean": param.grad.mean().item(),
                        "grad_std": param.grad.std().item(),
                        "grad_norm": param.grad.norm().item(),
                        "grad_param_ratio": (param.grad.norm() / param.norm()).item()
                    })
                
                # Add memory stats for large tensors
                if param.numel() > 1e6:  # Only for parameters with >1M elements
                    stats[name]["memory_mb"] = param.numel() * param.element_size() / (1024 * 1024)
        
        # Add global statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats["__global__"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "memory_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        return stats

    def _get_time_breakdown(self) -> Dict[str, float]:
        """Get detailed timing breakdown."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        return {
            "elapsed_total": elapsed,
            "elapsed_hours": elapsed / 3600,
            "steps_per_sec": len(self.step_times) / elapsed if self.step_times else 0,
            "avg_step_time": np.mean(self.step_times[-100:]) if self.step_times else 0,
            "std_step_time": np.std(self.step_times[-100:]) if self.step_times else 0
        }

class SDXLTrainingLogger(DetailedLogger):
    """Specialized logger for SDXL training with simplified and detailed views."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        use_wandb: bool = False,
        log_frequency: int = 1,
        window_size: int = 100
    ):
        super().__init__(log_dir, use_wandb, log_frequency)
        self.window_size = window_size
        self.metrics_history = {
            'loss': {
                'total': [],
                'v_pred': [],
                'simple': [],
                'moving_avg': [],
            },
            'v_pred': {
                'values': [],
                'targets': [],
                'errors': [],
                'moving_avg': []
            },
            'learning_rate': [],
            'grad_norms': [],
            'step_times': []
        }
        
        if use_wandb:
            self._setup_wandb_logging()
        
        # Initialize log files with proper encoding
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.gradients_file = self.log_dir / "gradients.jsonl"
        self.model_states_file = self.log_dir / "model_states.jsonl"
        self.training_steps_file = self.log_dir / "training_steps.jsonl"
        
        # Create files with headers if they don't exist
        for file_path in [self.metrics_file, self.gradients_file, 
                         self.model_states_file, self.training_steps_file]:
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")

    def _setup_wandb_logging(self):
        """Configure wandb logging with custom panels."""
        if not self.use_wandb:
            return
            
        wandb.define_metric("loss/total", summary="min")
        wandb.define_metric("loss/v_pred", summary="min")
        wandb.define_metric("loss/moving_avg", summary="min")
        
        # Create custom wandb panels
        wandb.run.log({
            "training_progress": wandb.Table(
                columns=["epoch", "step", "loss", "v_pred_loss", "lr"]
            )
        })
    
    def log_training_step(
        self,
        loss: torch.Tensor,
        metrics: Dict[str, float],
        learning_rate: float,
        step: int,
        epoch: int,
        batch_idx: int,
        step_time: float,
        v_pred_loss: Optional[torch.Tensor] = None,
        v_pred_values: Optional[torch.Tensor] = None,
        v_pred_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        """Log training step with both detailed and simplified metrics.
        
        Args:
            loss: Current training loss
            metrics: Dictionary of additional metrics
            learning_rate: Current learning rate
            step: Global step count
            epoch: Current epoch
            batch_idx: Current batch index
            step_time: Time taken for step
            v_pred_loss: Optional v-prediction loss
            v_pred_values: Optional v-prediction values
            v_pred_targets: Optional v-prediction targets
        """
        # Update history with v-prediction metrics
        self._update_history(loss, v_pred_loss, v_pred_values, v_pred_targets, learning_rate)
        moving_avgs = self._calculate_moving_averages()
        
        # Combine base metrics with v-prediction metrics
        combined_metrics = {
            "loss/total": loss.item(),
            "loss/moving_avg": moving_avgs['loss'],
            "learning_rate": learning_rate,
            "step_time": step_time,
            "steps_per_second": 1.0 / step_time if step_time > 0 else 0,
            **metrics
        }
        
        # Add v-prediction metrics if available
        if v_pred_loss is not None:
            combined_metrics.update({
                "v_pred/loss": v_pred_loss.item(),
                "v_pred/moving_avg": moving_avgs['v_pred'],
                "v_pred/error": moving_avgs['v_pred_error']
            })
        
        # Log to wandb with organized sections
        if self.use_wandb:
            wandb.log({
                # Main metrics panel
                "training": {
                    "loss": loss.item(),
                    "learning_rate": learning_rate,
                    "step_time": step_time,
                    **metrics
                },
                
                # V-prediction panel
                "v_prediction": {
                    "loss": v_pred_loss.item() if v_pred_loss is not None else 0.0,
                    "moving_avg": moving_avgs['v_pred'],
                    "error": moving_avgs['v_pred_error']
                } if v_pred_loss is not None else None,
                
                # Distributions
                "distributions": {
                    "loss_hist": wandb.Histogram(self.metrics_history['loss']['total'][-self.window_size:]),
                    "v_pred_hist": wandb.Histogram(self.metrics_history['v_pred']['values'][-self.window_size:])
                    if v_pred_values is not None else None
                },
                
                # Progress tracking
                "progress": {
                    "epoch": epoch,
                    "step": step,
                    "batch": batch_idx
                }
            }, step=step)
            
            # Update progress table
            wandb.run.log({
                "training_progress": wandb.Table(
                    data=[[
                        epoch,
                        step,
                        loss.item(),
                        v_pred_loss.item() if v_pred_loss is not None else 0.0,
                        learning_rate
                    ]],
                    columns=["epoch", "step", "loss", "v_pred_loss", "lr"]
                )
            })
        
        # Call parent class method with combined metrics
        super().log_training_step(
            loss=loss.item(),
            metrics=combined_metrics,
            learning_rate=learning_rate,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            step_time=step_time
        )
    
    def _update_history(self, loss, v_pred_loss, v_pred_values, v_pred_targets, lr):
        """Update metric history with new values."""
        self.metrics_history['loss']['total'].append(loss.item())
        self.metrics_history['learning_rate'].append(lr)
        
        if v_pred_loss is not None:
            self.metrics_history['loss']['v_pred'].append(v_pred_loss.item())
        
        if v_pred_values is not None and v_pred_targets is not None:
            self.metrics_history['v_pred']['values'].append(v_pred_values.mean().item())
            self.metrics_history['v_pred']['targets'].append(v_pred_targets.mean().item())
            self.metrics_history['v_pred']['errors'].append(
                torch.nn.functional.mse_loss(v_pred_values, v_pred_targets).item()
            )
    
    def _calculate_moving_averages(self) -> Dict[str, float]:
        """Calculate moving averages for all tracked metrics."""
        window = self.window_size
        return {
            'loss': np.mean(self.metrics_history['loss']['total'][-window:]),
            'v_pred': np.mean(self.metrics_history['loss']['v_pred'][-window:])
                     if self.metrics_history['loss']['v_pred'] else 0.0,
            'v_pred_error': np.mean(self.metrics_history['v_pred']['errors'][-window:])
                          if self.metrics_history['v_pred']['errors'] else 0.0,
            'grad_norm': np.mean(self.metrics_history['grad_norms'][-window:])
                        if self.metrics_history['grad_norms'] else 0.0,
            'step_time': np.mean(self.metrics_history['step_times'][-window:])
                        if self.metrics_history['step_times'] else 0.0
        }
