from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Union
import yaml




@dataclass
class VAEModelConfig:
    latent_channels: int = 16
    kl_divergence_weight: float = 0.1
    lpips_weight: float = 1.0
    discriminator_weight: float = 0.1
    use_attention: bool = False
    zero_init_last: bool = True
    pretrained_vae_name: str = "madebyollin/sdxl-vae-fp16-fix"

@dataclass
class ModelConfig:
    hidden_size: int = 768
    cross_attention_dim: int = 2048
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 20000.0
    rho: float = 7.0
    num_timesteps: int = 1000
    min_snr_gamma: float = 0.1
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: VAEModelConfig = field(default_factory=VAEModelConfig)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 4.0e-7
    num_epochs: int = 10
    save_steps: int = 1000
    log_steps: int = 10
    mixed_precision: str = "bf16"
    weight_decay: float = 1.0e-2
    optimizer_eps: float = 1.0e-8
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    vae_learning_rate: float = 4.5e-5
    vae_warmup_steps: int = 1000
    vae_min_lr: float = 1e-6
    use_discriminator: bool = True
    discriminator_learning_rate: float = 4.5e-5
    prediction_type: str = "v_prediction"  # v_prediction or epsilon
    
    # Learning rate scheduler settings
    lr_scheduler: Optional[Literal["cosine", "linear", "none"]] = "none"
    max_train_steps: Optional[int] = None
    warmup_steps: Optional[int] = 0
    
    # Timestep bias parameters
    timestep_bias_strategy: Literal["none", "earlier", "later", "range"] = "none"
    timestep_bias_multiplier: float = 1.0
    timestep_bias_begin: int = 0
    timestep_bias_end: int = 1000
    timestep_bias_portion: float = 0.25
    
    # SNR parameters
    snr_gamma: Optional[float] = 5.0
    max_grad_norm: Optional[float] = 1.0
    
    # Early stopping parameters
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: float = 0.01
    
    # Add wandb configuration
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    image_dirs: List[str]
    image_size: Tuple[int, int] = (1024, 1024)
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle: bool = True
    cache_dir: str = "latent_cache"
    text_cache_dir: str = "text_cache"
    vae_batch_size: int = 32
    vae_image_size: Tuple[int, int] = (256, 256)
    vae_validation_split: float = 0.1
    
    # Add missing bucketing and size configuration
    min_size: int = 256  # Minimum image dimension
    max_dim: int = 8192  # Maximum image dimension
    bucket_step: int = 8  # Resolution step size for buckets
    min_bucket_size: int = 16  # Minimum images per bucket
    bucket_tolerance: float = 0.2  # Tolerance for bucket aspect ratios
    max_aspect_ratio: float = 2.0  # Maximum allowed aspect ratio
    use_caching: bool = True  # Enable latent caching
    proportion_empty_prompts: float = 0.0  # Proportion of empty prompt training

@dataclass
class TagWeightingConfig:
    enabled: bool = True
    min_weight: float = 0.1
    max_weight: float = 2.0
    default_weight: float = 1.0
    update_frequency: int = 1000
    smoothing_factor: float = 0.1

@dataclass
class ScoringConfig:
    aesthetic_score: float = 6.0
    crop_score: float = 3.0

@dataclass
class SystemConfig:
    # Essential settings
    enable_xformers: bool = True  # Only memory optimization we'll keep
    gradient_checkpointing: bool = True  # Essential for SDXL
    mixed_precision: str = "bf16"  # Essential for training
    gradient_accumulation_steps: int = 4  # Essential for training
    channels_last: bool = True  # Memory format optimization
    
    def __init__(self, **kwargs):
        """Initialize with support for legacy fields."""
        # List of fields we want to keep
        valid_fields = {
            'enable_xformers',
            'gradient_checkpointing',
            'mixed_precision',
            'gradient_accumulation_steps',
            'channels_last'
        }
        
        # Filter to only keep essential fields
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        # Set attributes
        for key, value in valid_kwargs.items():
            setattr(self, key, value)

@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    output_dir: str = "outputs"
    vae_checkpoints_dir: str = "vae_checkpoints"
    vae_samples_dir: str = "vae_samples"

@dataclass
class NovelAIDatasetConfig:
    """Configuration for NovelAI dataset."""
    image_size: Union[Tuple[int, int], int] = (8192, 8192)
    min_size: Union[Tuple[int, int], int] = (256, 256)
    max_dim: Optional[int] = None
    bucket_step: int = 64
    min_bucket_size: Optional[int] = None
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 3.0
    cache_dir: str = "cache"
    text_cache_dir: str = "text_cache"
    use_caching: bool = True
    proportion_empty_prompts: float = 0.0
    
    # Add batch-related fields
    batch_size: int = 32
    max_consecutive_batch_samples: int = 2
    
    def __post_init__(self):
        """Convert single integers to tuples for sizes and validate."""
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        if isinstance(self.min_size, int):
            self.min_size = (self.min_size, self.min_size)
            
        # Handle max_dim if specified
        if self.max_dim is not None:
            self.image_size = (min(self.image_size[0], self.max_dim), 
                             min(self.image_size[1], self.max_dim))
            
        # Set default min_bucket_size if not specified
        if self.min_bucket_size is None:
            self.min_bucket_size = self.min_size[0] * self.min_size[1]

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        if 'model' in config_dict and 'vae' in config_dict['model']:
            config_dict['model']['vae'] = VAEModelConfig(**config_dict['model']['vae'])
            
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            tag_weighting=TagWeightingConfig(**config_dict['tag_weighting']),
            scoring=ScoringConfig(**config_dict['scoring']),
            system=SystemConfig(**config_dict['system']),
            paths=PathsConfig(**config_dict['paths'])
        )

    def get_vae_config(self) -> VAEModelConfig:
        """Helper method to easily access VAE config"""
        return self.model.vae
    