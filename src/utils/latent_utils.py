"""Utilities for latent space operations."""

import torch
import logging

logger = logging.getLogger(__name__)

def apply_noise_offset(latents, noise_offset):
    """
    Apply noise offset for improved sample quality.
    
    Args:
        latents (torch.Tensor): Input latents
        noise_offset (float): Noise offset value
        
    Returns:
        torch.Tensor: Processed latents
    """
    if noise_offset > 0:
        noise = torch.randn_like(latents)
        latents = latents + noise_offset * noise
        
    logger.debug("Latents shape after noise offset: %s", str(latents.shape))
    
    return latents


def get_latents_from_seed(
    seed,
    num_images,
    height,
    width,
    latent_channels=4,
    device="cuda",
    dtype=torch.float16,
    use_cache=True,
    cache_size=1000
):
    """
    Generate reproducible latents from seed.
    
    Args:
        seed (int): Random seed
        num_images (int): Number of images to generate
        height (int): Image height
        width (int): Image width
        latent_channels (int): Number of latent channels
        device (str): Device to generate latents on
        dtype (torch.dtype): Data type for latents
        
    Returns:
        torch.Tensor: Generated latents
    """
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Calculate latent dimensions
    latent_height = height // 8
    latent_width = width // 8
    
    # Generate latents
    latents = torch.randn(
        (num_images, latent_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype
    )
    
    logger.debug("Generated latents shape: %s", str(latents.shape))
    
    return latents
