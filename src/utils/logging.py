import logging
import sys
import os
from pathlib import Path
import torch
import wandb

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
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

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
        logger.info(f"{key}: {value}")
    
    # Log model information
    logger.info("\n=== Model Information ===")
    for name, model in models.items():
        if hasattr(model, 'parameters'):
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"{name}:")
            logger.info(f"  Total parameters: {num_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Log dataset information
    if 'dataset' in train_components:
        logger.info("\n=== Dataset Information ===")
        logger.info(f"Number of training samples: {len(train_components['dataset'])}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Steps per epoch: {len(train_components['train_dataloader'])}")

def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "GPU not available"

def setup_wandb(args):
    """
    Initialize Weights & Biases logging
    
    Args:
        args: Training arguments containing W&B configuration
        
    Returns:
        wandb.Run or None: Initialized W&B run
    """
    if not args.use_wandb:
        return None
        
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            resume=True
        )
        logging.info(f"Initialized W&B run: {run.name}")
        return run
        
    except Exception as e:
        logging.error(f"Failed to initialize W&B: {str(e)}")
        return None

def cleanup_wandb(run):
    """
    Cleanup W&B run
    
    Args:
        run: W&B run to cleanup
    """
    if run is not None:
        try:
            run.finish()
        except Exception as e:
            logging.error(f"Error closing W&B run: {str(e)}")
