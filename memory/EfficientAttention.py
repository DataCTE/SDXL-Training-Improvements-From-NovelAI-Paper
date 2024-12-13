import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from memory.quantization import LinearFp8


class MemoryEfficientAttention(nn.Module):
    """Memory efficient attention using scaled_dot_product_attention"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use FP8 quantization for linear layers
        self.qkv = LinearFp8(dim, dim * 3, bias=qkv_bias)
        self.proj = LinearFp8(dim, dim)
        self.dropout = dropout  # Pass dropout rate directly to scaled_dot_product_attention
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project to q, k, v and split heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, N, head_dim)
        
        # Handle mask if provided
        if mask is not None:
            # Ensure mask has correct shape [B, 1, N] or [B, N]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask.masked_fill_(~mask, float('-inf'))
        else:
            attn_mask = None
            
        # Use scaled_dot_product_attention for efficient attention computation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out

def replace_attention_layers(model: nn.Module):
    """Replace attention layers with memory efficient quantized version"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            new_attention = MemoryEfficientAttention(
                dim=module.embed_dim,
                num_heads=module.num_heads,
                qkv_bias=module.in_proj_bias is not None,
                dropout=module.dropout
            )
            setattr(model, name, new_attention)
        elif len(list(module.children())) > 0:
            replace_attention_layers(module)

class MemoryEfficientSDXLAttention(nn.Module):
    """Memory efficient attention for SDXL using scaled_dot_product_attention"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers without FP8 quantization for SDXL compatibility
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout  # Pass dropout rate directly to scaled_dot_product_attention
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project to q, k, v and split heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, N, head_dim)
        
        # Handle mask if provided
        if mask is not None:
            # Ensure mask has correct shape [B, 1, N] or [B, N]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask.masked_fill_(~mask, float('-inf'))
        else:
            attn_mask = None
            
        # Use scaled_dot_product_attention for efficient attention computation
        # Flash Attention will be automatically used when available
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale  # Explicitly pass scale factor
        )
        
        # Reshape and project output
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out

def replace_sdxl_attention_layers(model: nn.Module):
    """Replace SDXL attention layers with memory efficient version"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            new_attention = MemoryEfficientSDXLAttention(
                dim=module.embed_dim,
                num_heads=module.num_heads,
                qkv_bias=module.in_proj_bias is not None,
                dropout=module.dropout
            )
            setattr(model, name, new_attention)
        elif len(list(module.children())) > 0:
            replace_sdxl_attention_layers(module)

def is_quantized_parameter(module: nn.Module, name: str) -> bool:
    """Check if a parameter is quantized"""
    return hasattr(module, f"{name}_quantized")

def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Get size of offloaded tensor in bytes"""
    total_bytes = 0
    for name, param in module.named_parameters():
        if is_quantized_parameter(module, name):
            total_bytes += param.numel() * param.element_size()
    return total_bytes

def quantize_layers(module: nn.Module, device: torch.device, train_dtype: torch.dtype):
    """Quantize layers in module"""
    for name, submodule in module.named_modules():
        if isinstance(submodule, LinearFp8):
            submodule.quantize(device)
            submodule.compute_dtype = train_dtype

def offload_quantized(module: nn.Module, device: torch.device, non_blocking: bool = True):
    """Offload quantized parameters to specified device"""
    for name, param in module.named_parameters():
        if is_quantized_parameter(module, name):
            param.data = param.data.to(device, non_blocking=non_blocking)