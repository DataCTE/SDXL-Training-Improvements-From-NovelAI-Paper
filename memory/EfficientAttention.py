import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import math
import logging
from memory.quantization import LinearFp8
from flash_attn import flash_attn_func

logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for attention layers"""
    use_flash_attention: bool = True  # Whether to use Flash Attention when available
    use_memory_efficient_attention: bool = True  # Whether to use memory efficient attention
    attention_dropout: float = 0.0  # Attention dropout rate
    head_dropout: float = 0.0  # Head dropout rate
    use_rotary_embeddings: bool = False  # Whether to use rotary embeddings
    use_bias: bool = True  # Whether to use bias in linear layers
    use_scaled_init: bool = True  # Whether to use scaled initialization

class MemoryEfficientAttention(nn.Module):
    """Memory efficient attention using optimized implementation"""
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        dropout: float = 0.0,
        config: Optional[AttentionConfig] = None
    ):
        super().__init__()
        self.config = config or AttentionConfig()
        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use FP8 quantization for linear layers with improved initialization
        self.qkv = self._init_qkv_layer(dim, qkv_bias)
        self.proj = self._init_proj_layer(dim)
        
        self.dropout = dropout
        self.attention_dropout = nn.Dropout(self.config.attention_dropout)
        self.head_dropout = nn.Dropout(self.config.head_dropout)
        
        # Optional rotary embeddings
        self.rotary_emb = None
        if self.config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
            
    def _init_qkv_layer(self, dim: int, qkv_bias: bool) -> nn.Module:
        """Initialize QKV projection with optimal scaling"""
        layer = LinearFp8(dim, dim * 3, bias=qkv_bias)
        if self.config.use_scaled_init:
            # Scale initialization by head dimension
            std = math.sqrt(2.0 / (5 * dim))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if qkv_bias:
                nn.init.zeros_(layer.bias)
        return layer
        
    def _init_proj_layer(self, dim: int) -> nn.Module:
        """Initialize output projection with optimal scaling"""
        layer = LinearFp8(dim, dim, bias=self.config.use_bias)
        if self.config.use_scaled_init:
            # Use smaller init scale for output projection
            std = math.sqrt(1.0 / dim)
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if self.config.use_bias:
                nn.init.zeros_(layer.bias)
        return layer
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optimized attention computation
        
        Args:
            x: Input tensor of shape [B, N, C]
            mask: Attention mask
            past_key_value: Cached key/value tensors for incremental decoding
            use_cache: Whether to return key/value for caching
            
        Returns:
            Output tensor or tuple of (output, (key, value)) if use_cache=True
        """
        try:
            B, N, C = x.shape
            
            # Project to q, k, v and split heads
            qkv = self._compute_qkv(x)
            q, k, v = self._split_qkv(qkv, B, N)
            
            # Apply rotary embeddings if enabled
            if self.rotary_emb is not None:
                q, k = self.rotary_emb(q, k)
                
            # Handle cached key/values for incremental decoding
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
                
            # Compute attention with optimal implementation
            if self.config.use_flash_attention and self._can_use_flash_attention(q, k, v):
                attn_output = self._flash_attention(q, k, v, mask)
            else:
                attn_output = self._memory_efficient_attention(q, k, v, mask)
                
            # Project output
            out = self._project_output(attn_output, B, N)
            
            if use_cache:
                return out, (k, v)
            return out
            
        except Exception as e:
            logger.error(f"Error in attention forward pass: {e}")
            raise
            
    def _compute_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute QKV projections efficiently"""
        # Fuse QKV computation
        qkv = self.qkv(x)  # [B, N, 3*C]
        return qkv
        
    def _split_qkv(
        self, 
        qkv: torch.Tensor, 
        batch_size: int, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor efficiently"""
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [B, num_heads, N, head_dim]
        
        # Scale query
        q = q * self.scale
        return q, k, v
        
    def _can_use_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> bool:
        """Check if Flash Attention can be used"""
        if not self.config.use_flash_attention:
            return False
            
        # Check if Flash Attention is available
        try:
            
            # Check tensor requirements
            return (
                q.is_cuda and k.is_cuda and v.is_cuda and
                q.dtype in [torch.float16, torch.bfloat16] and
                k.dtype == v.dtype == q.dtype and
                q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            )
        except ImportError:
            return False
            
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention using Flash Attention"""
       
        
        # Reshape inputs for Flash Attention
        q = q.transpose(1, 2)  # [B, N, H, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Convert mask if needed
        if mask is not None:
            mask = mask.to(q.dtype)
            
        # Compute attention
        output = flash_attn_func(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False
        )
        
        return output.transpose(1, 2)  # [B, H, N, D]
        
    def _memory_efficient_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention with memory efficient implementation"""
        # Validate input shapes
        batch_size, num_heads, seq_len, head_dim = q.shape
        _, kv_heads, kv_len, _ = k.shape
        
        # Shape validation
        if kv_heads != num_heads:
            raise ValueError(f"Key/value heads {kv_heads} != query heads {num_heads}")
        if k.shape != v.shape:
            raise ValueError(f"Key shape {k.shape} != value shape {v.shape}")
        if head_dim != k.size(-1):
            raise ValueError(f"Query head dim {head_dim} != key head dim {k.size(-1)}")
            
        # Handle mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # Add head dimension
            mask = mask.unsqueeze(1)  # Add query dimension
            mask = (1.0 - mask) * torch.finfo(q.dtype).min
            
            # Validate mask shape
            expected_mask_shape = (batch_size, 1, 1, kv_len)
            if mask.shape != expected_mask_shape:
                raise ValueError(f"Mask shape {mask.shape} != expected shape {expected_mask_shape}")
            
        # Compute attention with chunked operations
        # Use smaller chunks for large batch sizes to maintain memory efficiency
        base_chunk_size = 1024
        chunk_size = min(seq_len, base_chunk_size // (batch_size * num_heads) + 1)
        
        output_chunks = []
        for i in range(0, seq_len, chunk_size):
            chunk_q = q[:, :, i:i+chunk_size]
            
            # Compute attention scores for chunk
            if mask is not None:
                chunk_mask = mask[:, :, :, i:i+chunk_size]
            else:
                chunk_mask = None
                
            # Compute scaled dot product attention
            # Reuse head_dim for proper scaling
            attn_weights = torch.matmul(chunk_q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if chunk_mask is not None:
                attn_weights = attn_weights + chunk_mask
                
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            
            # Compute attention output
            chunk_output = torch.matmul(attn_weights, v)
            output_chunks.append(chunk_output)
            
        # Concatenate chunks
        output = torch.cat(output_chunks, dim=2)
        
        # Validate output shape
        expected_output_shape = (batch_size, num_heads, seq_len, head_dim)
        if output.shape != expected_output_shape:
            raise ValueError(f"Output shape {output.shape} != expected shape {expected_output_shape}")
            
        return output
        
    def _project_output(
        self,
        attn_output: torch.Tensor,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """Project attention output efficiently"""
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.head_dropout(attn_output)
        out = self.proj(attn_output)
        return out

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self._rotary_cache = {}
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = q.shape[-2]
            
        if seq_len not in self._rotary_cache:
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            self._rotary_cache[seq_len] = (cos, sin)
            
        cos, sin = self._rotary_cache[seq_len]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
        
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return torch.cat([
            x[..., ::2] * cos - x[..., 1::2] * sin,
            x[..., 1::2] * cos + x[..., ::2] * sin
        ], dim=-1)

def replace_attention_layers(
    model: nn.Module,
    config: Optional[AttentionConfig] = None
) -> None:
    """Replace attention layers with memory efficient version"""
    try:
        for name, module in model.named_children():
            if isinstance(module, nn.MultiheadAttention):
                new_attention = MemoryEfficientAttention(
                    dim=module.embed_dim,
                    num_heads=module.num_heads,
                    qkv_bias=module.in_proj_bias is not None,
                    dropout=module.dropout,
                    config=config
                )
                setattr(model, name, new_attention)
            elif len(list(module.children())) > 0:
                replace_attention_layers(module, config)
                
    except Exception as e:
        logger.error(f"Error replacing attention layers: {e}")
        raise

class MemoryEfficientSDXLAttention(MemoryEfficientAttention):
    """Memory efficient attention specifically optimized for SDXL"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        config: Optional[AttentionConfig] = None
    ):
        super().__init__(dim, num_heads, qkv_bias, dropout, config)
        
        # Override QKV layer to use standard linear for SDXL compatibility
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Initialize with SDXL-specific scaling
        self._init_sdxl_parameters()
        
    def _init_sdxl_parameters(self):
        """Initialize parameters with SDXL-specific scaling"""
        nn.init.xavier_uniform_(self.qkv.weight, gain=1/math.sqrt(2))
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
            
        nn.init.xavier_uniform_(self.proj.weight, gain=1/math.sqrt(2))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

def replace_sdxl_attention_layers(
    model: nn.Module,
    config: Optional[AttentionConfig] = None
) -> None:
    """Replace SDXL attention layers with memory efficient version"""
    try:
        for name, module in model.named_children():
            if isinstance(module, nn.MultiheadAttention):
                new_attention = MemoryEfficientSDXLAttention(
                    dim=module.embed_dim,
                    num_heads=module.num_heads,
                    qkv_bias=module.in_proj_bias is not None,
                    dropout=module.dropout,
                    config=config
                )
                setattr(model, name, new_attention)
            elif len(list(module.children())) > 0:
                replace_sdxl_attention_layers(module, config)
                
    except Exception as e:
        logger.error(f"Error replacing SDXL attention layers: {e}")
        raise

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

class EfficientAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_flash_attention = True
        self.use_memory_efficient_attention = True
        
    def forward(self, q, k, v, mask=None):
        # Use Flash Attention when available
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False,
                scale=None
            )
        
        # Fallback to memory efficient attention
        if self.use_memory_efficient_attention:
            return self.memory_efficient_attention(q, k, v, mask)