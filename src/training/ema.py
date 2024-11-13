import torch
import copy

class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Initialize EMA model
        
        Args:
            model: Model to create EMA of
            decay: Decay rate for EMA
            device: Device to store EMA model on
        """
        self.decay = decay
        self.device = device if device else model.device
        
        # Create EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.to(device)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Initialize decay value
        self.cur_decay_value = 0.0
        self.num_updates = 0
        
    def step(self, model):
        """
        Update EMA model parameters
        
        Args:
            model: Current model to update EMA from
        """
        with torch.no_grad():
            self.num_updates += 1
            
            # Get decay value (can be adjusted based on number of updates)
            if self.num_updates < 2000:
                self.cur_decay_value = 0.0
            else:
                self.cur_decay_value = self.decay
            
            # Update each parameter
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                if model_param.requires_grad:
                    ema_param.data.mul_(self.cur_decay_value)
                    ema_param.data.add_((1 - self.cur_decay_value) * model_param.data)
                    
    def get_model(self):
        """Get the current EMA model"""
        return self.ema_model 