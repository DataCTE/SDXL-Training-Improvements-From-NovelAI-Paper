"""Logging utilities for SDXL training.

This module provides utilities for setting up logging and tracking training progress.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time
from datetime import datetime

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

class TrainingLogger:
    """Logger for tracking training progress and metrics."""
    
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        # Add timestamp and step
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "elapsed_time": time.time() - self.start_time,
            **metrics
        }
        
        # Write to metrics file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to console
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info("Step %d: %s", step, metrics_str)
    
    def log_validation_results(
        self,
        metrics: Dict[str, float],
        epoch: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Log validation results.
        
        Args:
            metrics: Dictionary of validation metrics
            epoch: Current epoch number
            save_path: Optional path where validation artifacts were saved
        """
        self.logger.info("=== Validation Results (Epoch %d) ===", epoch)
        for name, value in metrics.items():
            self.logger.info("%s: %.4f", name, value)
        
        if save_path:
            self.logger.info("Validation artifacts saved to: %s", save_path)
            
    def log_checkpoint(self, checkpoint_path: str, epoch: int) -> None:
        """Log checkpoint saving.
        
        Args:
            checkpoint_path: Path where checkpoint was saved
            epoch: Current epoch number
        """
        self.logger.info("Saved checkpoint for epoch %d to: %s", epoch, checkpoint_path)
