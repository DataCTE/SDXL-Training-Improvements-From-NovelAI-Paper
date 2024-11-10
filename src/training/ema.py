import torch
import copy

class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Initialize EMA
        Args:
            model: Model to apply EMA to
            decay: EMA decay rate (higher = slower moving average)
            device: Device to store EMA model on
        """
        self.decay = decay
        self.device = device

        # Create EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        if device is not None:
            self.ema_model.to(device)

    def update_parameters(self, model):
        """
        Update EMA parameters
        Args:
            model: Current model to update EMA from
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                if model_param.requires_grad:
                    ema_param.data.mul_(self.decay)
                    ema_param.data.add_((1 - self.decay) * model_param.data)

    def state_dict(self):
        """Get state dict"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.ema_model.load_state_dict(state_dict) 