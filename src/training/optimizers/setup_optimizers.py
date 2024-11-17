from typing import Dict, Any, List, Optional, Union
import torch
import logging
from src.training.optimizers.adamw8bit import setup_adamw_optimizer
from src.training.optimizers.soap import SOAP

logger = logging.getLogger(__name__)

def create_optimizer(
    model_params: Union[List[torch.nn.Parameter], List[Dict[str, Any]]],
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    use_8bit_optimizer: bool = False,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for model training.
    
    Args:
        model_params: List of parameters to optimize or param groups
        optimizer_type: Type of optimizer to use ('adamw' supported)
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        adam_beta1: Beta1 factor for Adam
        adam_beta2: Beta2 factor for Adam
        adam_epsilon: Epsilon for Adam
        use_8bit_optimizer: Whether to use 8-bit optimization when available
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: For unsupported optimizer types or invalid configurations
    """
    if not model_params:
        raise ValueError("No parameters provided for optimization")
        
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adamw":
        return setup_adamw_optimizer(
            params_to_optimize=model_params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            use_8bit=use_8bit_optimizer
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def setup_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    use_8bit_optimizer: bool = False,
) -> torch.optim.Optimizer:
    """
    Setup an optimizer for a given model.
    
    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer to use ('adamw' supported)
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        adam_beta1: Beta1 factor for Adam
        adam_beta2: Beta2 factor for Adam
        adam_epsilon: Epsilon for Adam
        use_8bit_optimizer: Whether to use 8-bit optimization when available
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: For unsupported optimizer types or invalid configurations
    """
    # Get model parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    logger.info(f"Setting up {optimizer_type} optimizer with learning rate {learning_rate}")
    
    return create_optimizer(
        model_params=params_to_optimize,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        use_8bit_optimizer=use_8bit_optimizer
    )