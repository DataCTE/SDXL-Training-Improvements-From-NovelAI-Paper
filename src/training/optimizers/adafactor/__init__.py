
import torch
import logging
from typing import Dict, Any, List
from transformers.optimization import Adafactor
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _get_adafactor_config(
    learning_rate: float,
    weight_decay: float,
    scale_parameter: bool = True,
    relative_step: bool = False,
    warmup_init: bool = False,
) -> Dict[str, Any]:
    """Cache Adafactor optimizer configurations."""
    return {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "scale_parameter": scale_parameter,
        "relative_step": relative_step,
        "warmup_init": warmup_init,
    }

def setup_adafactor_optimizer(
    params_to_optimize: List[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 1e-2,
    scale_parameter: bool = True,
    relative_step: bool = False,
    warmup_init: bool = False,
) -> torch.optim.Optimizer:
    """
    Set up Adafactor optimizer with proper configuration and memory optimizations.
    
    Args:
        params_to_optimize: List of parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        scale_parameter: Whether to scale parameters
        relative_step: Whether to use relative step sizes
        warmup_init: Whether to use warmup initialization
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If no parameters to optimize
        RuntimeError: If optimizer initialization fails
    """
    if not params_to_optimize:
        raise ValueError("No parameters to optimize")

    # Get cached optimizer config
    opt_config = _get_adafactor_config(
        learning_rate,
        weight_decay,
        scale_parameter,
        relative_step,
        warmup_init,
    )
    
    try:
        optimizer = Adafactor(params_to_optimize, **opt_config)
        return optimizer
        
    except Exception as e:
        logger.error("Failed to initialize Adafactor optimizer: %s", str(e))
        raise RuntimeError(f"Optimizer initialization failed: {str(e)}")
