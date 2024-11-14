import torch
import copy
import math
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class EMAModel:
    """
    Advanced Exponential Moving Average of model weights with sophisticated decay scheduling
    """
    def __init__(
        self, 
        model, 
        decay: float = 0.9999, 
        min_decay: float = 0.0, 
        warmup_steps: int = 2000, 
        device: Optional[torch.device] = None,
        adaptive_decay: bool = True
    ):
        """
        Initialize EMA model with advanced configuration
        
        Args:
            model: Model to create EMA of
            decay: Maximum decay rate for EMA (default: 0.9999)
            min_decay: Minimum decay rate during warmup (default: 0.0)
            warmup_steps: Number of steps for decay warmup (default: 2000)
            device: Device to store EMA model on
            adaptive_decay: Enable adaptive decay based on model performance
        """
        self.decay = decay
        self.min_decay = min_decay
        self.warmup_steps = warmup_steps
        self.device = device or model.device
        self.adaptive_decay = adaptive_decay
        
        # Create EMA model with memory-efficient deep copy
        self.ema_model = self._efficient_deepcopy(model)
        self.ema_model.to(self.device)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Advanced tracking variables
        self.num_updates = 0
        self.cur_decay_value = min_decay
        
        # Performance tracking
        self.performance_history = []
        self.performance_window = 10  # Track last 10 performance metrics
    
    def _efficient_deepcopy(self, model):
        """
        Create memory-efficient deep copy of model
        
        Args:
            model: Model to copy
        
        Returns:
            Copied model
        """
        try:
            ema_model = copy.deepcopy(model)
            return ema_model
        except Exception as e:
            logger.warning(f"Deep copy failed, falling back to state dict copy: {e}")
            ema_model = type(model)(*model.args, **model.kwargs)
            ema_model.load_state_dict(model.state_dict())
            return ema_model
    
    def get_current_decay(self):
        """
        Calculate current decay rate with advanced scheduling
        
        Returns:
            float: Current decay rate
        """
        if self.num_updates < self.warmup_steps:
            # Cosine warmup schedule
            progress = self.num_updates / self.warmup_steps
            return self.min_decay + (self.decay - self.min_decay) * 0.5 * (1 + math.cos(math.pi * progress))
        
        if self.adaptive_decay and self.performance_history:
            # Adaptive decay based on performance stability
            performance_variance = torch.tensor(self.performance_history).std().item()
            adaptive_factor = 1 - min(performance_variance, 1.0)
            return self.decay * adaptive_factor
        
        return self.decay
    
    def step(self, model, performance_metric: Optional[Union[float, torch.Tensor]] = None):
        """
        Update EMA model parameters with optional performance tracking
        
        Args:
            model: Current model to update EMA from
            performance_metric: Optional performance metric for adaptive decay
        """
        with torch.no_grad():
            self.num_updates += 1
            
            # Track performance if provided
            if performance_metric is not None:
                if isinstance(performance_metric, torch.Tensor):
                    performance_metric = performance_metric.item()
                
                self.performance_history.append(performance_metric)
                self.performance_history = self.performance_history[-self.performance_window:]
            
            # Get current decay value
            self.cur_decay_value = self.get_current_decay()
            
            # Update each parameter efficiently
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                if model_param.requires_grad:
                    ema_param.lerp_(model_param.data, 1 - self.cur_decay_value)
    
    def get_model(self):
        """
        Get the current EMA model for inference
        
        Returns:
            Model: Current EMA model
        """
        return self.ema_model
    
    def state_dict(self):
        """
        Get comprehensive EMA state for saving
        
        Returns:
            dict: State dictionary with model, tracking, and performance info
        """
        return {
            'ema_model': self.ema_model.state_dict(),
            'num_updates': self.num_updates,
            'cur_decay_value': self.cur_decay_value,
            'performance_history': self.performance_history
        }
    
    def load_state_dict(self, state_dict):
        """
        Load comprehensive EMA state
        
        Args:
            state_dict (dict): State dictionary to load
        """
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.num_updates = state_dict.get('num_updates', 0)
        self.cur_decay_value = state_dict.get('cur_decay_value', self.min_decay)
        self.performance_history = state_dict.get('performance_history', [])
    
    def get_decay_stats(self):
        """
        Get comprehensive decay statistics
        
        Returns:
            dict: Decay-related metrics
        """
        return {
            'num_updates': self.num_updates,
            'current_decay': self.cur_decay_value,
            'performance_variance': (
                torch.tensor(self.performance_history).std().item() 
                if self.performance_history else 0.0
            )
        }