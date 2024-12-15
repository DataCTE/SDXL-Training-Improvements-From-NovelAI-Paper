from dataclasses import dataclass
from typing import List, Tuple
import yaml

@dataclass
class ModelConfig:
    hidden_size: int
    cross_attention_dim: int
    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float
    num_timesteps: int
    min_snr_gamma: float
    pretrained_model_name: str

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    save_steps: int
    log_steps: int
    mixed_precision: str
    weight_decay: float
    optimizer_eps: float
    optimizer_betas: Tuple[float, float]

@dataclass
class DataConfig:
    image_size: List[int]
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    shuffle: bool
    min_tag_weight: float = 0.5
    max_tag_weight: float = 2.0
    image_dirs: List[str]
    cache_dir: str
    text_cache_dir: str

@dataclass
class TagWeightingConfig:
    min_weight: float
    max_weight: float
    default_weight: float
    enabled: bool
    update_frequency: int
    smoothing_factor: float

@dataclass
class ScoringConfig:
    aesthetic_score: float
    crop_score: float

@dataclass
class SystemConfig:
    enable_xformers: bool
    channels_last: bool
    gradient_checkpointing: bool
    cudnn_benchmark: bool
    disable_debug_apis: bool

@dataclass
class PathsConfig:
    checkpoints_dir: str
    logs_dir: str
    output_dir: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    tag_weighting: TagWeightingConfig
    scoring: ScoringConfig
    system: SystemConfig
    paths: PathsConfig
    
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