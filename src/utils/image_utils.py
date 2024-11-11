import torch
import torch.nn.functional as F

def encode_images(images, vae=None):
    """
    Encode images to latent space using VAE
    
    Args:
        images: Tensor of images [B, C, H, W]
        vae: Optional VAE model (if None, assumes images are already latents)
    
    Returns:
        Tensor of latents [B, 4, H/8, W/8]
    """
    if vae is None:
        return images
        
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    return latents

def add_noise(latents, noise, timesteps, noise_scheduler):
    """
    Add noise to latents according to diffusion schedule
    
    Args:
        latents: Clean latents [B, 4, H, W]
        noise: Random noise of same shape as latents
        timesteps: Timesteps to sample noise for [B]
        noise_scheduler: Noise scheduling object
        
    Returns:
        Noisy latents of same shape as input
    """
    noisy_latents = noise_scheduler.add_noise(
        original_samples=latents,
        noise=noise,
        timesteps=timesteps
    )
    
    return noisy_latents

def prepare_latents(batch_size, num_channels, height, width, dtype, device, generator=None):
    """
    Prepare random latents for generation
    
    Args:
        batch_size: Number of images to generate
        num_channels: Number of latent channels (usually 4)
        height: Height of latent image
        width: Width of latent image
        dtype: Data type of latents
        device: Device to create latents on
        generator: Optional random number generator
        
    Returns:
        Random latents of shape [B, C, H, W]
    """
    shape = (batch_size, num_channels, height, width)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    
    return latents 