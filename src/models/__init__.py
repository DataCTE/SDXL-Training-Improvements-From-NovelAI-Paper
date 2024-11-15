from diffusers.models.attention import BasicTransformerBlock
from torch.utils.checkpoint import checkpoint

def enable_efficient_attention():
    """Enable memory efficient attention globally"""
    try:
        import xformers
        import xformers.ops
        BasicTransformerBlock.forward = memory_efficient_forward
        return True
    except ImportError:
        return False

def memory_efficient_forward(self, hidden_states, attention_mask=None, **kwargs):
    """Memory efficient transformer forward pass"""
    if hasattr(self, "enable_xformers_memory_efficient_attention"):
        self.enable_xformers_memory_efficient_attention()
    return checkpoint(self._forward, hidden_states, attention_mask, **kwargs)

__all__ = ['enable_efficient_attention']
