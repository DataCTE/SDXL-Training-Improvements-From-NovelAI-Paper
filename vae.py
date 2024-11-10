import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = AdaptiveWeightNorm(channels)
        self.norm2 = AdaptiveWeightNorm(channels)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class SWYCCAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16,),
        channel_multipliers: Tuple[int] = (1, 2, 4, 8),
        num_head_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        ])
        
        current_channels = hidden_channels
        feature_size = 256  # Starting size
        
        # Down blocks
        for multiplier in channel_multipliers:
            out_channels = hidden_channels * multiplier
            
            # Add res blocks
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(current_channels))
                if feature_size in attention_resolutions:
                    self.encoder.append(
                        nn.MultiheadAttention(
                            current_channels,
                            num_head_channels,
                            batch_first=True
                        )
                    )
                
            # Add downsampling
            if multiplier != channel_multipliers[-1]:  # Don't downsample on last block
                self.encoder.append(
                    nn.Conv2d(current_channels, out_channels, 3, stride=2, padding=1)
                )
                feature_size = feature_size // 2
                current_channels = out_channels
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(current_channels, latent_channels * 2, 1)
        
        # Decoder (DInitial)
        self.decoder_initial = nn.ModuleList([
            nn.Conv2d(latent_channels, current_channels, 1)
        ])
        
        # Up blocks
        for multiplier in reversed(channel_multipliers[:-1]):
            out_channels = hidden_channels * multiplier
            
            # Add res blocks
            for _ in range(num_res_blocks):
                self.decoder_initial.append(ResBlock(current_channels))
                if feature_size in attention_resolutions:
                    self.decoder_initial.append(
                        nn.MultiheadAttention(
                            current_channels,
                            num_head_channels,
                            batch_first=True
                        )
                    )
            
            # Add upsampling
            self.decoder_initial.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(current_channels, out_channels, 3, padding=1)
                )
            )
            feature_size = feature_size * 2
            current_channels = out_channels
        
        # Final conv
        self.decoder_initial.append(
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        
        # Apply encoder layers
        for module in self.encoder:
            if isinstance(module, nn.MultiheadAttention):
                # Reshape for attention
                b, c, h, w = h.shape
                h = h.reshape(b, c, -1).permute(0, 2, 1)
                h, _ = module(h, h, h)
                h = h.permute(0, 2, 1).reshape(b, c, h, w)
            else:
                h = module(h)
        
        # Get mean and log variance
        moments = self.bottleneck(h)
        mean, logvar = moments.chunk(2, dim=1)
        
        # Return distribution
        return mean, logvar

    def decode_initial(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        
        # Apply decoder layers
        for module in self.decoder_initial:
            if isinstance(module, nn.MultiheadAttention):
                # Reshape for attention
                b, c, h, w = h.shape
                h = h.reshape(b, c, -1).permute(0, 2, 1)
                h, _ = module(h, h, h)
                h = h.permute(0, 2, 1).reshape(b, c, h, w)
            else:
                h = module(h)
        
        return h

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode_initial(z)
        return x_recon, mean, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_channels, 32, 32, device=device)
        samples = self.decode_initial(z)
        return samples