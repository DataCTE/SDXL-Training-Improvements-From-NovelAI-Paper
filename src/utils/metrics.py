import torch
import logging
import os
import sys
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

logger = logging.getLogger(__name__)

def setup_logging(
    config: Config,
    logs_dir: str, 
    is_main_process: bool = True
) -> None:
    """Setup logging configuration.
    
    Args:
        config: Training configuration
        logs_dir: Directory to store log files
        is_main_process: Whether this is the main process
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO if is_main_process else logging.WARNING,
            format=log_format,
            handlers=[
                # Console handler
                logging.StreamHandler(sys.stdout),
                # File handler
                logging.FileHandler(os.path.join(logs_dir, 'training.log'))
            ] if is_main_process else []
        )
        
        # Suppress unnecessary warnings
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        
        if is_main_process:
            logger.info(f"Logging setup complete. Logs will be saved to: {logs_dir}")
            
            # Initialize wandb if enabled
            if config.training.use_wandb:
                try:
                    import wandb
                    if not wandb.run:
                        wandb.init(
                            project=config.training.wandb_project,
                            name=config.training.wandb_run_name,
                            tags=config.training.wandb_tags,
                            config=config
                        )
                    logger.info("Wandb initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing wandb: {e}")
            
    except Exception as e:
        print(f"Error setting up logging: {str(e)}", file=sys.stderr)
        raise

def log_system_info() -> None:
    """Log system information including hardware, OS, and Python/PyTorch versions."""
    try:
        # System info
        logger.info("System Information:")
        logger.info(f"OS: {platform.system()} {platform.version()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_threads} threads")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
        
        # GPU info
        if torch.cuda.is_available():
            logger.info("GPU Information:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / (1024**3):.1f}GB")
                logger.info(f"  Compute capability: {props.major}.{props.minor}")
                logger.info(f"  Multi-processor count: {props.multi_processor_count}")
                
                # Current memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"  Memory usage: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        else:
            logger.warning("No GPU available - running in CPU mode")
            
        # CUDA environment
        if torch.cuda.is_available():
            logger.info("CUDA Environment:")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            logger.info(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
            logger.info(f"TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}")
            
            # Check for tensor cores
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                has_tensor_cores = props.major >= 7
                logger.info(f"GPU {i} tensor cores: {'Available' if has_tensor_cores else 'Not available'}")
                
        # Additional PyTorch info
        logger.info("PyTorch Configuration:")
        logger.info(f"Number of threads: {torch.get_num_threads()}")
        logger.info(f"Default dtype: {torch.get_default_dtype()}")
        
    except Exception as e:
        logger.error(f"Error logging system information: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
def compute_grad_norm(model: torch.nn.Module, grad_norm_buffer: torch.Tensor) -> float:
    """Compute gradient norm efficiently using pre-allocated buffer."""
    try:
        # Get all gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
            
        # Compute norms efficiently
        grad_norm_buffer[:len(grads)].copy_(torch.tensor([g.norm().item() for g in grads]))
        return grad_norm_buffer[:len(grads)].norm().item()
        
    except Exception as e:
        logger.error(f"Error computing gradient norm: {str(e)}")
        return 0.0

def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    is_main_process: bool,
    use_wandb: bool,
    step_type: str = "step"
) -> None:
    """Log metrics to console and wandb if enabled.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        is_main_process: Whether this is the main process
        use_wandb: Whether to use wandb
        step_type: Type of step (e.g. "step", "epoch")
    """
    if not is_main_process:
        return
        
    try:
        # Log to console
        metric_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items()])
        logger.info(f"{step_type.capitalize()} {step}: {metric_str}")
        
        # Log to wandb if enabled (assume already initialized)
        if use_wandb:
            import wandb
            wandb.log(metrics, step=step)
            
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")

def cleanup_logging(is_main_process: bool = True) -> None:
    """Cleanup logging resources.
    
    Args:
        is_main_process: Whether this is the main process
    """
    if is_main_process:
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except Exception as e:
            logger.error(f"Error cleaning up wandb: {e}")