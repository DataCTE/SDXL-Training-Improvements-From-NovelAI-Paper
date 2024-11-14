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
        Calculate decay rate using improved scheduling from NovelAI.
        
        Args:
            optimization_step: Current optimization step
            
        Returns:
            float: Current decay rate
        """
        if optimization_step < self.update_after_step:
            return 0.0
        
        value = 1 - (1 + optimization_step - self.update_after_step) ** -self.power
        
        if self.use_ema_warmup:
            value = 1 - (1 - value) * math.cos(math.pi * 0.5 * min(1.0, optimization_step / self.inv_gamma))
        
        return max(self.min_decay, min(value, self.max_decay))
    
    def _update_param(
        self,
        ema_param: torch.Tensor,
        model_param: torch.Tensor,
        param_name: str,
        decay: float
    ) -> None:
        """
        Update a single parameter with momentum and gradient-based weighting.
        
        Args:
            ema_param: Parameter in EMA model
            model_param: Parameter in training model
            param_name: Name of the parameter
            decay: Current decay rate
        """
        # Calculate parameter difference
        diff = model_param.data - ema_param.data
        
        # Update momentum
        self.param_momentum[param_name] = (
            self.param_momentum[param_name] * decay +
            diff * (1 - decay)
        )
        
        # Apply update with gradient-based weighting
        if model_param.grad is not None:
            grad_norm = model_param.grad.norm().item()
            self.grad_norms[param_name] = (
                self.grad_norms[param_name] * 0.9 +
                grad_norm * 0.1
            )
            update_weight = 1.0 / (1.0 + self.grad_norms[param_name] * self.grad_scale_factor)
        else:
            update_weight = 1.0
        
        ema_param.data.add_(self.param_momentum[param_name] * update_weight)
    
    def step(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
        grad_scale: Optional[float] = None
    ) -> None:
        """
        Update EMA model parameters with improved update strategy.
        
        Args:
            model: Current model
            step: Optional step number (uses internal count if None)
            grad_scale: Optional gradient scale factor
        """
        if step is not None:
            self.optimization_step = step
        else:
            self.optimization_step += 1
        
        # Skip updates before warmup or if not update step
        if self.optimization_step < self.update_after_step or self.optimization_step % self.update_every != 0:
            return
        
        # Calculate current decay rate
        decay = self.get_decay(self.optimization_step)
        self.decay_history.append(decay)
        
        with torch.no_grad():
            # Update parameters with momentum and gradient weighting
            for (name, ema_param), (_, model_param) in zip(
                self.ema_model.named_parameters(),
                model.named_parameters()
            ):
                if model_param.requires_grad:
                    self._update_param(ema_param, model_param, name, decay)
    
    def get_model(self) -> torch.nn.Module:
        """Get current EMA model"""
        return self.ema_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get complete state including momentum and gradient tracking"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'optimization_step': self.optimization_step,
            'param_momentum': self.param_momentum,
            'grad_norms': self.grad_norms,
            'decay_history': list(self.decay_history),
            'initial_params': self.initial_params
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load complete state including momentum and gradient tracking"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.optimization_step = state_dict.get('optimization_step', 0)
        self.param_momentum = state_dict.get('param_momentum', {})
        self.grad_norms = state_dict.get('grad_norms', {})
        self.decay_history = deque(
            state_dict.get('decay_history', []),
            maxlen=self.decay_history.maxlen
        )
        self.initial_params = state_dict.get('initial_params', {})