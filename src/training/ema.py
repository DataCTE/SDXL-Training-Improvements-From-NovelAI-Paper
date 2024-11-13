import torch
import copy
import math

class EMAModel:
    """
    Exponential Moving Average of model weights with warmup and decay scheduling
    """
    def __init__(self, model, decay=0.9999, min_decay=0.0, warmup_steps=2000, device=None):
        """
        Initialize EMA model
        
        Args:
            model: Model to create EMA of
            decay: Maximum decay rate for EMA (default: 0.9999)
            min_decay: Minimum decay rate during warmup (default: 0.0)
            warmup_steps: Number of steps for decay warmup (default: 2000)
            device: Device to store EMA model on
        """
        self.decay = decay
        self.min_decay = min_decay
        self.warmup_steps = warmup_steps
        self.device = device if device else model.device
        
        # Create EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.to(device)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Initialize tracking variables
        self.num_updates = 0
        self.cur_decay_value = min_decay
        
    def get_current_decay(self):
        """Calculate current decay rate based on warmup schedule"""
        if self.num_updates < self.warmup_steps:
            # Linear warmup from min_decay to decay
            return self.min_decay + (self.decay - self.min_decay) * (self.num_updates / self.warmup_steps)
        return self.decay
    
    def step(self, model):
        """
        Update EMA model parameters
        
        Args:
            model: Current model to update EMA from
        """
        with torch.no_grad():
            self.num_updates += 1
            self.cur_decay_value = self.get_current_decay()
            
            # Update each parameter
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                if model_param.requires_grad:
                    # Use memory efficient update
                    ema_param.lerp_(model_param.data, 1 - self.cur_decay_value)
                    
    def get_decay_value(self):
        """Get current decay value for logging"""
        return self.cur_decay_value
    
    def load_state_dict(self, state_dict):
        """Load EMA state"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.num_updates = state_dict.get('num_updates', 0)
        self.cur_decay_value = state_dict.get('cur_decay_value', self.min_decay)
    
    def state_dict(self):
        """Get EMA state for saving"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'num_updates': self.num_updates,
            'cur_decay_value': self.cur_decay_value
        }
    
    def get_model(self):
        """Get the current EMA model for inference"""
        return self.ema_model