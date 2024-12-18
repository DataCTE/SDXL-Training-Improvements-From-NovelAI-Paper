from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Union
import yaml
import torch

# Common default values
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 16
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_MAX_TOKEN_LENGTH = 77
DEFAULT_CACHE_DIR = "cache"
DEFAULT_IMAGE_SIZE = (1024, 1024)
DEFAULT_MAX_IMAGE_SIZE = (2048, 2048)
DEFAULT_MIN_IMAGE_SIZE = (256, 256)

# Common learning rates
DEFAULT_LR = 4.0e-7
DEFAULT_VAE_LR = 4.5e-5
DEFAULT_DISC_LR = 4.5e-5

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
    pretrained_model_name: str = DEFAULT_MODEL_NAME
    hidden_size: int = 768
    cross_attention_dim: int = 2048

@dataclass
class TrainingConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = 4
    learning_rate: float = DEFAULT_LR
    vae_learning_rate: float = DEFAULT_VAE_LR
    vae_warmup_steps: int = 1000
    vae_min_lr: float = 1e-6
    use_discriminator: bool = True
    discriminator_learning_rate: float = DEFAULT_DISC_LR
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
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    max_image_size: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
    min_image_size: Union[Tuple[int, int], int] = DEFAULT_MIN_IMAGE_SIZE
    max_dim: int = 4194304
    bucket_step: int = 8
    min_bucket_size: int = 16
    min_bucket_resolution: Optional[int] = None
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 2.0
    
    # Cache settings
    cache_dir: str = DEFAULT_CACHE_DIR
    text_cache_dir: str = "text_cache"
    use_caching: bool = True
    
    # Text settings
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    
    # VAE settings
    vae_batch_size: int = DEFAULT_BATCH_SIZE
    vae_image_size: Tuple[int, int] = (256, 256)
    vae_validation_split: float = 0.1
    
    # DataLoader settings
    num_workers: int = DEFAULT_NUM_WORKERS
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
    enable_xformers: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 4
    channels_last: bool = True
    
    def __init__(self, **kwargs):
        valid_fields = {
            'enable_xformers',
            'gradient_checkpointing',
            'mixed_precision',
            'gradient_accumulation_steps',
            'channels_last'
        }
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
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
    model_name: str = DEFAULT_MODEL_NAME
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    max_image_size: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
    min_image_size: Union[Tuple[int, int], int] = DEFAULT_MIN_IMAGE_SIZE
    max_dim: int = 2048
    
    # Bucket settings
    bucket_step: int = 8
    min_bucket_size: int = 16
    min_bucket_resolution: Optional[int] = None
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 2.0
    
    # Cache settings
    cache_dir: str = DEFAULT_CACHE_DIR
    text_cache_dir: str = "text_cache"
    use_caching: bool = True
    skip_cached_latents: bool = True
    
    # Dataset settings
    proportion_empty_prompts: float = 0.0
    max_consecutive_batch_samples: int = 2
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    
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
    device: torch.device = torch.device('cuda')
    dtype: torch.dtype = torch.float16
    max_memory_usage: float = 0.9
    enable_memory_efficient_attention: bool = True

@dataclass
class CacheConfig:
    use_caching: bool = True
    cache_dir: str = DEFAULT_CACHE_DIR

@dataclass
class ImageSizeConfig:
    max_size: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
    min_size: Tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    max_dim: int = 4194304
    bucket_step: int = 64
    min_bucket_resolution: int = 256 * 256
    max_aspect_ratio: float = 4.0
    bucket_tolerance: float = 0.2

@dataclass
class BatchProcessorConfig(DeviceConfig):
    batch_size: int = DEFAULT_BATCH_SIZE
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR
    log_interval: float = 5.0
    num_workers: int = DEFAULT_NUM_WORKERS
    
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
class VAEEncoderConfig(DeviceConfig):
    enable_vae_slicing: bool = True
    vae_batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR
    
    # Image normalization settings
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5)

@dataclass
class BucketConfig:
    bucket_tolerance: float = 0.2

@dataclass
class TextProcessorConfig(DeviceConfig, CacheConfig):
    num_workers: int = DEFAULT_NUM_WORKERS
    batch_size: int = DEFAULT_BATCH_SIZE
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    
    # Tag weighting settings
    enable_tag_weighting: bool = True
    tag_frequency_threshold: int = 5
    tag_weight_smoothing: float = 0.1

@dataclass
class TextEmbedderConfig(DeviceConfig):
    max_length: int = DEFAULT_MAX_TOKEN_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    model_name: str = DEFAULT_MODEL_NAME
    
    # Model settings
    use_fast_tokenizer: bool = True
    low_cpu_mem_usage: bool = True
    
    # Performance settings
    growth_factor: float = 0.3
    proportion_empty_prompts: float = 0.0
    
    # Subfolder settings
    tokenizer_subfolder: str = "tokenizer"
    tokenizer_2_subfolder: str = "tokenizer_2"
    text_encoder_subfolder: str = "text_encoder"
    text_encoder_2_subfolder: str = "text_encoder_2"

@dataclass
class GlobalConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    image: ImageSizeConfig = field(default_factory=ImageSizeConfig)

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    tag_weighting: TagWeighterConfig
    scoring: ScoringConfig
    system: SystemConfig
    paths: PathsConfig
    global_config: GlobalConfig
    vae_encoder: VAEEncoderConfig
    batch_processor: BatchProcessorConfig
    bucket: BucketConfig
    text_processor: TextProcessorConfig
    text_embedder: TextEmbedderConfig

    def __post_init__(self):
        """Apply global settings to components."""
        # Apply device settings
        for component in [self.vae_encoder, self.batch_processor, 
                         self.text_processor, self.text_embedder]:
            component.device = self.global_config.device.device
            component.dtype = self.global_config.device.dtype
            component.max_memory_usage = self.global_config.device.max_memory_usage
            component.enable_memory_efficient_attention = self.global_config.device.enable_memory_efficient_attention

        # Apply image size settings to DataConfig
        self.data.image_size = self.global_config.image.target_size
        self.data.max_image_size = self.global_config.image.max_size
        self.data.min_image_size = self.global_config.image.min_size
        self.data.max_dim = self.global_config.image.max_dim
        self.data.bucket_step = self.global_config.image.bucket_step
        self.data.min_bucket_resolution = self.global_config.image.min_bucket_resolution
        self.data.max_aspect_ratio = self.global_config.image.max_aspect_ratio
        self.data.bucket_tolerance = self.global_config.image.bucket_tolerance

        # Apply image size settings to NovelAIDatasetConfig
        if hasattr(self, 'novel_ai'):
            self.novel_ai.image_size = self.global_config.image.target_size
            self.novel_ai.max_image_size = self.global_config.image.max_size
            self.novel_ai.min_image_size = self.global_config.image.min_size
            self.novel_ai.max_dim = self.global_config.image.max_dim
            self.novel_ai.bucket_step = self.global_config.image.bucket_step
            self.novel_ai.min_bucket_resolution = self.global_config.image.min_bucket_resolution
            self.novel_ai.max_aspect_ratio = self.global_config.image.max_aspect_ratio
            self.novel_ai.bucket_tolerance = self.global_config.image.bucket_tolerance

        # Apply bucket settings
        self.bucket.bucket_tolerance = self.global_config.image.bucket_tolerance

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
                text_processor=TextProcessorConfig(**config_dict['text_processor']),
                text_embedder=TextEmbedderConfig(**config_dict['text_embedder'])
            )
        except Exception as e:
            raise ValueError(f"Error loading config from {path}: {str(e)}") from e

    def get_vae_config(self) -> VAEModelConfig:
        """Helper method to easily access VAE config"""
        return self.model.vae
    
@dataclass
class ImageProcessorConfig(DeviceConfig):
    """Configuration for image processor."""
    max_image_size: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
    min_image_size: Tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE
    enable_vae_slicing: bool = True
    vae_batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR
    # We can let the parent DeviceConfig handle max_memory_usage, etc.,
    # but you can add them here if you prefer to override default values.
    