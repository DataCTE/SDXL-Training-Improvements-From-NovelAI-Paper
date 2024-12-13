from dataclasses import dataclass, field
from typing import List, Tuple
import torch
from torchvision import transforms

@dataclass
class TrainingConfig:
    # Dataset paths
    image_dirs: List[str] = field(default_factory=lambda: [
       r'path/to/your/dataset'
    ])
    
    # Training hyperparameters
    batch_size: int = 32
    grad_accum_steps: int = 4
    learning_rate: float = 4e-7
    num_epochs: int = 10
    save_interval: int = 1000
    log_interval: int = 10
    
    # Model configuration
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Optimization parameters
    min_snr_gamma: float = 0.1
    sigma_min: float = 0.002
    sigma_max: float = 20000.0
    rho: float = 7.0
    num_timesteps: int = 1000
    
    # Image processing
    max_image_size: Tuple[int, int] = (768, 1024)
    max_dim: int = 1024
    bucket_step: int = 64
    
    # Paths
    cache_dir: str = "latent_cache"
    text_cache_dir: str = "text_cache"
    checkpoint_dir: str = "checkpoints"
    
    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            lambda x: x[:3],  # ensure 3 channels
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            lambda x: x.to(torch.bfloat16)
        ]) 