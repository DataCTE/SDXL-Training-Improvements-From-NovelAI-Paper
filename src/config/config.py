from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml

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
    enable_xformers: bool = True
    channels_last: bool = True
    gradient_checkpointing: bool = True
    cudnn_benchmark: bool = True
    disable_debug_apis: bool = True
    compile_model: bool = True
    num_gpu_workers: Optional[int] = None

@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    output_dir: str = "outputs"

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
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            tag_weighting=TagWeightingConfig(**config_dict['tag_weighting']),
            scoring=ScoringConfig(**config_dict['scoring']),
            system=SystemConfig(**config_dict['system']),
            paths=PathsConfig(**config_dict['paths'])
        ) 