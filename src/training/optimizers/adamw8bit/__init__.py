
import torch
import logging
from typing import Dict, Any, List
from bitsandbytes.optim import AdamW8bit
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _get_adamw_config(
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
) -> Dict[str, Any]:
    """Cache AdamW optimizer configurations."""
    return {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon
    }

def setup_adamw_optimizer(
    params_to_optimize: List[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 1e-2,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """
    Set up AdamW optimizer with proper configuration and memory optimizations.
    
    Args:
        params_to_optimize: List of parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        adam_beta1: Beta1 factor for Adam
        adam_beta2: Beta2 factor for Adam
        adam_epsilon: Epsilon for Adam
        use_8bit: Whether to use 8-bit AdamW
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If no parameters to optimize
        RuntimeError: If 8-bit optimization fails
    """
    if not params_to_optimize:
        raise ValueError("No parameters to optimize")

    # Get cached optimizer config
    opt_config = _get_adamw_config(
        learning_rate,
        weight_decay,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
    )
    
    try:
        if use_8bit:
            logger.info("Using 8-bit AdamW optimizer")
            optimizer = AdamW8bit(params_to_optimize, **opt_config)
        else:
            optimizer = torch.optim.AdamW(params_to_optimize, **opt_config)
            
        return optimizer
        
    except Exception as e:
        logger.error("Failed to initialize AdamW optimizer: %s", str(e))
        raise RuntimeError(f"Optimizer initialization failed: {str(e)}")
