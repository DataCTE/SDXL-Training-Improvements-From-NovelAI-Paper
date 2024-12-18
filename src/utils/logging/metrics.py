import torch
import logging
import os
import sys
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
from src.config.config import Config
import sys


try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)

def setup_logging(
    config: Config,
    logs_dir: str, 
    is_main_process: bool = True,
    log_level: str = "INFO"
) -> None:
    """Setup logging configuration with more detailed formatting and handlers."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging format with more detail
        log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        if is_main_process:
            # Console handler with color formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            root_logger.addHandler(console_handler)
            
            # File handlers
            # Main log file
            main_log = logs_dir / 'training.log'
            file_handler = logging.FileHandler(main_log)
            file_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            root_logger.addHandler(file_handler)
            
            # Debug log file
            debug_log = logs_dir / 'debug.log'
            debug_handler = logging.FileHandler(debug_log)
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            root_logger.addHandler(debug_handler)
            
            # Error log file
            error_log = logs_dir / 'error.log'
            error_handler = logging.FileHandler(error_log)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            root_logger.addHandler(error_handler)
            
            logger.info(f"Logging setup complete. Logs will be saved to: {logs_dir}")
            logger.info(f"Log files created:\n"
                       f"- Main log: {main_log}\n"
                       f"- Debug log: {debug_log}\n"
                       f"- Error log: {error_log}")
            
            # Initialize wandb if enabled
            if config.training.use_wandb:
                try:
                    import wandb
                    if not wandb.run:
                        wandb.init(
                            project=config.training.wandb_project,
                            name=config.training.wandb_run_name,
                            tags=config.training.wandb_tags,
                            config=config,
                            settings=wandb.Settings(start_method="thread")
                        )
                    logger.info("Wandb initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing wandb: {e}")
            
        # Suppress unnecessary warnings
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        
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
    step: Optional[int] = None,
    step_type: str = "global",
    is_main_process: bool = True,
    use_wandb: bool = True
) -> None:
    """Log metrics with proper error handling."""
    try:
        if not is_main_process:
            return

        # Log to console
        if step is not None:
            logger.info(f"Step {step} ({step_type}) metrics:")
        else:
            logger.info(f"Metrics ({step_type}):")
            
        for key, value in metrics.items():
            logger.info(f"- {key}: {value}")
            
        # Log to wandb if enabled
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error logging to wandb: {e}")
                
    except Exception as e:
        log_error_with_context(e, "Error logging metrics")

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

def log_batch_metrics(
    metrics: Dict[str, Any],
    step: int,
    phase: str = "train",
    is_main_process: bool = True,
    use_wandb: bool = False,
    log_interval: int = 10
) -> None:
    """Log detailed batch metrics with better formatting."""
    if not is_main_process or step % log_interval != 0:
        return
        
    try:
        # Format metrics string
        metric_lines = [f"Step {step} ({phase}):"]
        for k, v in metrics.items():
            if isinstance(v, float):
                metric_lines.append(f"  {k}: {v:.4f}")
            else:
                metric_lines.append(f"  {k}: {v}")
                
        # Log to console/file
        logger.info("\n".join(metric_lines))
        
        # Log to wandb
        if use_wandb:
            import wandb
            wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
            
    except Exception as e:
        logger.error(f"Error logging batch metrics: {e}")

def log_system_metrics(
    prefix: str = "",
    include_gpu: bool = True,
    include_memory: bool = True
) -> Dict[str, Any]:
    """Log detailed system metrics."""
    metrics = {}
    
    try:
        if include_memory:
            process = psutil.Process()
            metrics.update({
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss_gb": process.memory_info().rss / (1024**3)
            })
            
        if include_gpu and torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
            })
            
            # Add NVML metrics if available
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics.update({
                        "gpu_memory_total_gb": info.total / (1024**3),
                        "gpu_memory_used_gb": info.used / (1024**3),
                        "gpu_memory_free_gb": info.free / (1024**3),
                        "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    })
                except Exception as e:
                    logger.warning(f"Error getting NVML metrics: {e}")
            
        # Log formatted metrics
        if metrics:
            logger.info(f"{prefix}System Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.2f}")
                else:
                    logger.info(f"  {k}: {v}")
                    
    except Exception as e:
        logger.error(f"Error logging system metrics: {e}")
        
    return metrics

def log_error_with_context(
    error: Exception,
    context: str,
    include_traceback: bool = True
) -> None:
    """Log errors with additional context information."""
    try:
        import traceback
        error_msg = [
            f"Error in {context}:",
            f"  Type: {type(error).__name__}",
            f"  Message: {str(error)}"
        ]
        
        if include_traceback:
            error_msg.append("\nTraceback:")
            error_msg.extend(
                f"  {line}" for line in traceback.format_exc().split("\n")
            )
            
        logger.error("\n".join(error_msg))
        
    except Exception as e:
        logger.error(f"Error logging error: {e}")