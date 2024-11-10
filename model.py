import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # Initialize using weight normalization instead of batch norm
        self.q_norm = nn.utils.weight_norm(self.to_q)
        self.k_norm = nn.utils.weight_norm(self.to_k)
        self.v_norm = nn.utils.weight_norm(self.to_v)
        self.out_norm = nn.utils.weight_norm(self.to_out)

    def forward(self, x, context=None):
        context = context if context is not None else x
        
        q = self.q_norm(x)
        k = self.k_norm(context)
        v = self.v_norm(context)
        
        q = q.reshape(q.shape[0], -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(k.shape[0], -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(v.shape[0], -1, self.heads, self.dim_head).transpose(1, 2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.heads * self.dim_head)
        return self.out_norm(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=num_heads)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Replace LayerNorm with AdaptiveWeightNorm
        self.norm1 = AdaptiveWeightNorm(dim)
        self.norm2 = AdaptiveWeightNorm(dim)
        self.norm3 = AdaptiveWeightNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class AdaptiveWeightNorm(nn.Module):
    """Adaptive weight normalization with learnable scale per filter"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        # Normalize each filter independently
        norm = torch.sqrt(torch.sum(x * x, dim=(2, 3), keepdim=True) + self.eps)
        return self.weight.view(1, -1, 1, 1) * x / norm

class UNet2DConditionModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, context_dim=2048):
        super().__init__()
        # SDXL architecture specs
        self.channels = [320, 640, 1280]  # Channel multipliers
        self.transformer_depth = [0, 2, 10]  # Transformer blocks per stage
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)
        self.init_norm = AdaptiveWeightNorm(self.channels[0])
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        curr_channels = self.channels[0]
        
        for ch, n_transformers in zip(self.channels, self.transformer_depth):
            # Add downsampling block
            self.down_blocks.append(nn.ModuleList([
                nn.Conv2d(curr_channels, ch, 3, padding=1),
                AdaptiveWeightNorm(ch),
                nn.Conv2d(ch, ch, 3, padding=1),
                AdaptiveWeightNorm(ch)
            ]))
            
            # Add transformer blocks
            for _ in range(n_transformers):
                self.down_blocks.append(
                    TransformerBlock(ch, context_dim=context_dim, num_heads=8)
                )
            curr_channels = ch
        
        # Up blocks (mirror of down blocks)
        self.up_blocks = nn.ModuleList()
        for ch, n_transformers in zip(reversed(self.channels), reversed(self.transformer_depth)):
            # Add upsampling block
            self.up_blocks.append(nn.ModuleList([
                nn.Conv2d(curr_channels * 2, ch, 3, padding=1),  # *2 for skip connection
                AdaptiveWeightNorm(ch),
                nn.Conv2d(ch, ch, 3, padding=1),
                AdaptiveWeightNorm(ch)
            ]))
            
            # Add transformer blocks
            for _ in range(n_transformers):
                self.up_blocks.append(
                    TransformerBlock(ch, context_dim=context_dim, num_heads=8)
                )
            curr_channels = ch
            
        # Output conv
        self.out_conv = nn.Conv2d(self.channels[0], out_channels, 3, padding=1)
        
        # Size and crop conditioning
        self.size_embedding = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )
        self.crop_embedding = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )

        # Add DInitial decoder path
        self.initial_decoder = nn.ModuleList([
            nn.Conv2d(curr_channels, ch, 3, padding=1),
            AdaptiveWeightNorm(ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            AdaptiveWeightNorm(ch)
        ])

    def forward(self, x, sigma, context=None, initial_output=None, size_cond=None, crop_cond=None):
        # Default size conditioning to 1024x1024 if not provided
        if size_cond is None:
            size_cond = torch.tensor([[1024, 1024]], device=x.device, dtype=x.dtype)
            
        # Default crop conditioning to no crop if not provided    
        if crop_cond is None:
            crop_cond = torch.tensor([[0, 0]], device=x.device, dtype=x.dtype)
            
        # Embed conditioning signals
        size_emb = self.size_embedding(size_cond)
        crop_emb = self.crop_embedding(crop_cond)
        context = context if context is not None else x
        
        # Initial conv
        h = self.init_conv(x)
        h = self.init_norm(h)
        h = F.silu(h)
        
        # Store skip connections
        skips = []
        
        # Down path
        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):  # Conv block
                for layer in block:
                    h = layer(h)
                    if isinstance(layer, AdaptiveWeightNorm):
                        h = F.silu(h)
                skips.append(h)
                h = F.avg_pool2d(h, 2)
            else:  # Transformer block
                h = block(h, context)
        
        # Up path
        for block in self.up_blocks:
            if isinstance(block, nn.ModuleList):  # Conv block
                h = F.interpolate(h, scale_factor=2, mode='nearest')
                h = torch.cat([h, skips.pop()], dim=1)
                for layer in block:
                    h = layer(h)
                    if isinstance(layer, AdaptiveWeightNorm):
                        h = F.silu(h)
            else:  # Transformer block
                h = block(h, context)
        
        # Add initial decoder output to refinement path
        if initial_output is not None:
            h = torch.cat([h, initial_output], dim=1)
        
        # Output
        return self.out_conv(h)

    def initial_decode(self, latents):
        """Initial decoding path that produces coarse output"""
        h = latents
        for layer in self.initial_decoder:
            h = layer(h)
            if isinstance(layer, AdaptiveWeightNorm):
                h = F.silu(h)
        return h