from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Union
import yaml
import torch




@dataclass
class VAEModelConfig:
    latent_channels: int = 3
    kl_divergence_weight: float = 0.1
    lpips_weight: float = 1.0
    discriminator_weight: float = 0.1
    use_attention: bool = False
    zero_init_last: bool = True
    pretrained_vae_name: str = "madebyollin/sdxl-vae-fp16-fix"

@dataclass
class ModelConfig:
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 20000.0
    rho: float = 7.0
    num_timesteps: int = 1000
    min_snr_gamma: float = 0.1
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hidden_size: int = 768
    cross_attention_dim: int = 2048

@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 4.0e-7
    vae_learning_rate: float = 4.5e-5
    vae_warmup_steps: int = 1000
    vae_min_lr: float = 1e-6
    use_discriminator: bool = True
    discriminator_learning_rate: float = 4.5e-5
    num_epochs: int = 10
    save_steps: int = 1000
    log_steps: int = 10
    mixed_precision: str = "bf16"
    weight_decay: float = 1.0e-2
    optimizer_eps: float = 1.0e-8
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate scheduler settings
    lr_scheduler: Literal["cosine", "linear", "none"] = "none"
    max_train_steps: Optional[int] = None
    warmup_steps: int = 0
    
    # Prediction settings
    prediction_type: Literal["v_prediction", "epsilon"] = "v_prediction"
    timestep_bias_strategy: Literal["none", "earlier", "later", "range"] = "none"
    timestep_bias_multiplier: float = 1.0
    timestep_bias_begin: int = 0
    timestep_bias_end: int = 1000
    timestep_bias_portion: float = 0.25
    snr_gamma: float = 5.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "sdxl-training"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    image_dirs: List[str]
    image_size: Tuple[int, int] = (1024, 1024)
    max_image_size: Tuple[int, int] = (2048, 2048)
    min_image_size: Union[Tuple[int, int], int] = (256, 256)
    max_dim: int = 2048
    bucket_step: int = 8
    min_bucket_size: int = 16
    min_bucket_resolution: Optional[int] = None
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 2.0
    
    # Cache settings
    cache_dir: str = "latent_cache"
    text_cache_dir: str = "text_cache"
    use_caching: bool = True
    
    # Text settings
    max_token_length: int = 77  # Default CLIP token length
    
    # VAE settings
    vae_batch_size: int = 32
    vae_image_size: Tuple[int, int] = (256, 256)
    vae_validation_split: float = 0.1
    
    # DataLoader settings
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle: bool = True
    proportion_empty_prompts: float = 0.0

    def __post_init__(self):
        """Convert and validate configuration."""
        # Convert image sizes to tuples if needed
        if isinstance(self.image_size, (list, tuple)):
            self.image_size = tuple(self.image_size)
        if isinstance(self.max_image_size, (list, tuple)):
            self.max_image_size = tuple(self.max_image_size)
        if isinstance(self.min_image_size, (list, tuple)):
            self.min_image_size = tuple(self.min_image_size)
        elif isinstance(self.min_image_size, int):
            self.min_image_size = (self.min_image_size, self.min_image_size)
            
        # Set min_bucket_resolution if not specified
        if self.min_bucket_resolution is None:
            self.min_bucket_resolution = min(self.min_image_size)
            
        # Update max_dim to be the sum of max_image_size dimensions
        self.max_dim = sum(self.max_image_size)

@dataclass
class TagWeighterConfig:
    """Configuration for tag weighting."""
    default_weight: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 3.0
    smoothing_factor: float = 1e-4
    dtype: torch.dtype = torch.float32

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
    # Required parameter first
    model_name: str  # Required for text embedder
    
    # Optional parameters with defaults after
    image_size: Tuple[int, int] = (1024, 1024)
    max_image_size: Tuple[int, int] = (2048, 2048)
    min_image_size: Union[Tuple[int, int], int] = (256, 256)
    max_dim: int = 2048
    
    # Bucket settings
    bucket_step: int = 8
    min_bucket_size: int = 16
    min_bucket_resolution: Optional[int] = None
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 2.0
    
    # Cache settings
    cache_dir: str = "cache"
    text_cache_dir: str = "text_cache"
    use_caching: bool = True
    
    # Dataset settings
    proportion_empty_prompts: float = 0.0
    max_consecutive_batch_samples: int = 2
    max_token_length: int = 77  # Default CLIP token length
    
    # Tag weighting settings
    tag_weighting: TagWeighterConfig = field(default_factory=TagWeighterConfig)

    def __post_init__(self):
        """Convert and validate configuration."""
        # Convert image sizes to tuples if needed
        if isinstance(self.image_size, (list, tuple)):
            self.image_size = tuple(self.image_size)
        if isinstance(self.max_image_size, (list, tuple)):
            self.max_image_size = tuple(self.max_image_size)
        if isinstance(self.min_image_size, (list, tuple)):
            self.min_image_size = tuple(self.min_image_size)
        elif isinstance(self.min_image_size, int):
            self.min_image_size = (self.min_image_size, self.min_image_size)
            
        # Set min_bucket_resolution if not specified
        if self.min_bucket_resolution is None:
            self.min_bucket_resolution = min(self.min_image_size)

@dataclass
class DeviceConfig:
    """Base configuration for device and memory settings."""
    device: torch.device = torch.device('cuda')
    dtype: torch.dtype = torch.float16
    max_memory_usage: float = 0.9
    enable_memory_efficient_attention: bool = True

@dataclass
class CacheConfig:
    """Base configuration for caching."""
    use_caching: bool = True
    cache_dir: str = "cache"

@dataclass
class ImageSizeConfig:
    """Base configuration for image dimensions."""
    max_image_size: Tuple[int, int] = (2048, 2048)
    min_image_size: Tuple[int, int] = (256, 256)

@dataclass
class BatchProcessorConfig(DeviceConfig):
    """Configuration for batch processing."""
    batch_size: int = 32
    prefetch_factor: int = 2
    log_interval: float = 5.0
    num_workers: int = 16
    
    # Batch size adjustment settings
    min_batch_size: int = 1
    max_batch_size: int = 64
    memory_check_interval: float = 30.0
    memory_growth_factor: float = 0.7
    retry_count: int = 3
    backoff_factor: float = 1.5
    cleanup_interval: int = 1000
    high_memory_threshold: float = 0.95

@dataclass
class VAEEncoderConfig(DeviceConfig, ImageSizeConfig):
    """Configuration for VAE encoding and image processing."""
    enable_vae_slicing: bool = True
    vae_batch_size: int = 32
    num_workers: int = 16
    prefetch_factor: int = 2
    
    # Image normalization settings
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5)

@dataclass
class BucketConfig(ImageSizeConfig):
    """Configuration for image bucketing."""
    bucket_step: int = 64
    min_bucket_resolution: int = 2048 * 2048
    max_aspect_ratio: float = 4.0
    bucket_tolerance: float = 0.2

@dataclass
class TextProcessorConfig(DeviceConfig, CacheConfig):
    """Configuration for text processing."""
    num_workers: int = 16
    batch_size: int = 32
    max_token_length: int = 77
    
    # Tag weighting settings
    enable_tag_weighting: bool = True
    tag_frequency_threshold: int = 5
    tag_weight_smoothing: float = 0.1

@dataclass
class GlobalConfig:
    """Global configuration settings shared across all components."""
    image_sizes: ImageSizeConfig = field(default_factory=ImageSizeConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    tag_weighting: TagWeighterConfig
    scoring: ScoringConfig
    system: SystemConfig
    paths: PathsConfig
    global_config: GlobalConfig  # Add global config
    vae_encoder: VAEEncoderConfig
    batch_processor: BatchProcessorConfig
    bucket: BucketConfig
    text_processor: TextProcessorConfig

    def __post_init__(self):
        """Apply global settings to components."""
        # Apply device settings
        for component in [self.vae_encoder, self.batch_processor, self.text_processor]:
            component.device = self.global_config.device.device
            component.dtype = self.global_config.device.dtype
            component.max_memory_usage = self.global_config.device.max_memory_usage
            component.enable_memory_efficient_attention = self.global_config.device.enable_memory_efficient_attention

        # Apply image size settings where needed
        self.vae_encoder.max_image_size = self.global_config.image_sizes.max_image_size
        self.vae_encoder.min_image_size = self.global_config.image_sizes.min_image_size
        self.bucket.max_image_size = self.global_config.image_sizes.max_image_size
        self.bucket.min_image_size = self.global_config.image_sizes.min_image_size

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file with improved error handling."""
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['model', 'training', 'data', 'tag_weighting', 
                               'scoring', 'system', 'paths']
            missing_sections = [s for s in required_sections if s not in config_dict]
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
            
            # Handle VAE config if present
            if 'vae' in config_dict.get('model', {}):
                config_dict['model']['vae'] = VAEModelConfig(**config_dict['model']['vae'])
            
            # Create config instance with proper type conversion
            return cls(
                model=ModelConfig(**config_dict['model']),
                training=TrainingConfig(**config_dict['training']),
                data=DataConfig(**config_dict['data']),
                tag_weighting=TagWeighterConfig(**config_dict['tag_weighting']),
                scoring=ScoringConfig(**config_dict['scoring']),
                system=SystemConfig(**config_dict['system']),
                paths=PathsConfig(**config_dict['paths']),
                global_config=GlobalConfig(**config_dict['global_config']),
                vae_encoder=VAEEncoderConfig(**config_dict['vae_encoder']),
                batch_processor=BatchProcessorConfig(**config_dict['batch_processor']),
                bucket=BucketConfig(**config_dict['bucket']),
                text_processor=TextProcessorConfig(**config_dict['text_processor'])
            )
        except Exception as e:
            raise ValueError(f"Error loading config from {path}: {str(e)}") from e

    def get_vae_config(self) -> VAEModelConfig:
        """Helper method to easily access VAE config"""
        return self.model.vae
    