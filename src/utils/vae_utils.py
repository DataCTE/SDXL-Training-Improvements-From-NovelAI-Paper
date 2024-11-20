"""VAE-related utility functions."""

import torch

def normalize_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Normalize VAE latents using NAI's statistics (section B.2).
    Makes each channel a standard Gaussian.
    """
    # NAI statistics from paper figure 9
    means = torch.tensor([4.8119, 0.1607, 1.3538, -1.7753], device=latents.device)
    stds = torch.tensor([9.9181, 6.2753, 7.5978, 5.9956], device=latents.device)
    
    # Reshape for broadcasting
    means = means.view(1, -1, 1, 1)
    stds = stds.view(1, -1, 1, 1)
    
    # Normalize to standard Gaussian
    return (latents - means) / stds