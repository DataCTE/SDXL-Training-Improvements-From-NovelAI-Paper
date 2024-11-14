import torch
import copy
import math
import logging
from typing import Optional, Union, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

class EMAModel:
    """
    Exponential Moving Average model with NovelAI improvements including dynamic decay,
    parameter-specific momentum, and gradient-based update weighting.
    """
    def __init__(
        self, 
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        inv_gamma: float = 1.0,
        power: float = 2/3,
        min_decay: float = 0.0,
        max_decay: float = 0.9999,
        device: Optional[torch.device] = None,
        update_every: int = 1,
        use_ema_warmup: bool = True,
        grad_scale_factor: float = 0.5,
        model_cls: Optional[Any] = None
    ):
        """
        Initialize EMA model with NovelAI improvements.
        
        Args:
            model: Model to create EMA of
            decay: Base decay rate (default: 0.9999)
            update_after_step: Start EMA after this many steps
            inv_gamma: Inverse multiplicative factor of EMA warmup length
            power: Power for decay rate schedule
            min_decay: Minimum decay rate
            max_decay: Maximum decay rate
            device: Device to store EMA model on
            update_every: Update every N steps
            use_ema_warmup: Whether to use EMA warmup
            grad_scale_factor: Factor for gradient-based update weighting
            model_cls: Optional model class for specialized initialization
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.grad_scale_factor = grad_scale_factor
        
        # Initialize EMA model
        self.model_cls = model_cls or type(model)
        self.ema_model = self._initialize_ema_model(model)
        
        # Advanced tracking
        self.optimization_step = 0
        self.decay_history = deque(maxlen=1000)  # Track decay rate history
        self.param_momentum = {}  # Parameter-specific momentum
        self.grad_norms = {}  # Store gradient norms for adaptive updates
        
        # Store initial parameter values
        with torch.no_grad():
            self.initial_params = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }
    
    def _initialize_ema_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Initialize EMA model with improved memory efficiency"""
        try:
            # Try specialized initialization if model class is provided
            if self.model_cls is not None:
                ema_model = self.model_cls()
                ema_model.load_state_dict(model.state_dict())
            else:
                ema_model = copy.deepcopy(model)
            
            ema_model.to(self.device)
            ema_model.eval()
            ema_model.requires_grad_(False)
            
            # Initialize parameter-specific tracking
            for name, param in ema_model.named_parameters():
                self.param_momentum[name] = 0.0
                self.grad_norms[name] = 0.0
            
            return ema_model
            
        except Exception as e:
            logger.error(f"EMA model initialization failed: {e}")
            raise
    
    def get_decay(self, optimization_step: int) -> float:
        """
        Compute dynamic decay rate based on optimization step.
        Uses improved scheduling from NovelAI paper.
        """
        step = max(0, optimization_step - self.update_after_step)
        
        if self.use_ema_warmup and step <= self.update_after_step:
            decay = min(
                self.max_decay,
                (1 + step / (self.update_after_step * self.inv_gamma)) ** -self.power
            )
        else:
            decay = min(
                self.max_decay,
                (1 + step) ** -self.power
            )
        
        decay = max(self.min_decay, decay)
        self.decay_history.append(decay)
        
        return decay
    
    def get_param_decay(self, name: str, grad_norm: float) -> float:
        """
        Compute parameter-specific decay rate based on gradient norm.
        """
        base_decay = self.get_decay(self.optimization_step)
        
        # Update gradient norm moving average
        self.grad_norms[name] = (
            self.grad_norms[name] * 0.9 +
            grad_norm * 0.1
        )
        
        # Scale decay by gradient norm
        decay = base_decay * (1.0 - self.grad_scale_factor * 
                            math.tanh(self.grad_norms[name]))
        
        return max(self.min_decay, min(self.max_decay, decay))
    
    def step(self, model: torch.nn.Module):
        """
        Update EMA model with improved parameter-specific momentum.
        """
        self.optimization_step += 1
        
        # Skip update if before start or not on update step
        if (self.optimization_step < self.update_after_step or
            self.optimization_step % self.update_every != 0):
            return
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Get EMA parameter
                ema_param = self.ema_model.get_parameter(name)
                
                # Compute gradient norm for adaptive decay
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                else:
                    grad_norm = 0.0
                
                # Get parameter-specific decay
                decay = self.get_param_decay(name, grad_norm)
                
                # Update parameter-specific momentum
                self.param_momentum[name] = (
                    decay * self.param_momentum[name] +
                    (1 - decay) * (param.data - ema_param.data)
                )
                
                # Update EMA parameter
                ema_param.data.add_(self.param_momentum[name])
    
    def copy_to(self, target_model: torch.nn.Module):
        """
        Copy EMA parameters to target model.
        """
        with torch.no_grad():
            for name, param in self.ema_model.named_parameters():
                target_param = target_model.get_parameter(name)
                target_param.copy_(param)
    
    def store(self, parameters: Dict[str, torch.Tensor]):
        """
        Store current parameters for restoration.
        """
        with torch.no_grad():
            for name, param in self.ema_model.named_parameters():
                parameters[name] = param.clone()
    
    def restore(self, parameters: Dict[str, torch.Tensor]):
        """
        Restore parameters from storage.
        """
        with torch.no_grad():
            for name, param in self.ema_model.named_parameters():
                param.copy_(parameters[name])
    
    def get_decay_info(self) -> Dict[str, Any]:
        """
        Get information about current decay rates and momentum.
        """
        return {
            'current_decay': self.get_decay(self.optimization_step),
            'decay_history': list(self.decay_history),
            'param_momentum': {
                k: v for k, v in self.param_momentum.items()
            },
            'grad_norms': {
                k: v for k, v in self.grad_norms.items()
            }
        }