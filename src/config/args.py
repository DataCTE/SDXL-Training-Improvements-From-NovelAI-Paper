"""Command line argument configuration for SDXL training.

This module provides a structured way to define and parse command line arguments
for the SDXL training pipeline, organized by functional categories.
"""

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False

@dataclass
class SchedulerConfig:
    use_scheduler: bool = True
    num_warmup_steps: int = 1000
    num_training_steps: int = 10000
    num_cycles: int = 1

@dataclass
class EMAConfig:
    decay: float = 0.9999
    update_after_step: int = 100
    use_warmup: bool = True
    warmup_steps: int = 2000

@dataclass
class VAEConfig:
    enable_vae_finetuning: bool = False
    vae_path: Optional[str] = None
    learning_rate: float = 1e-6
    train_freq: int = 10
    kl_weight: float = 0.0
    perceptual_weight: float = 0.0
    use_channel_scaling: bool = True
    initial_scale_factor: float = 1.0

@dataclass
class TagWeightingConfig:
    token_dropout_rate: float = 0.1
    caption_dropout_rate: float = 0.1
    rarity_factor: float = 0.9
    emphasis_factor: float = 1.2
    min_tag_freq: int = 10
    min_cluster_size: int = 5
    similarity_threshold: float = 0.3

@dataclass
class TrainingConfig:
    # Required arguments (no defaults)
    model_path: str
    data_dir: str
    
    # Optional arguments (with defaults)
    output_dir: str = "./output"
    batch_size: int = 1
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    training_mode: str = "v_prediction"
    training: bool = True
    use_ztsnr: bool = False
    rescale_cfg: bool = False
    rescale_multiplier: float = 0.7
    resolution_scaling: bool = True
    min_snr_gamma: float = 5.0
    sigma_data: float = 1.0
    sigma_min: float = 0.029
    sigma_max: float = 160.0
    scale_method: str = "karras"
    scale_factor: float = 0.7
    device: str = "cuda"
    enable_compile: bool = False
    compile_mode: str = "default"
    gradient_checkpointing: bool = True
    num_workers: int = 4
    validation_dir: Optional[str] = None
    cache_dir: str = "latents_cache"
    no_caching: bool = False
    min_size: int = 512
    max_size: int = 4096
    bucket_step_size: int = 64
    max_bucket_area: int = 1024*1024
    validation_prompts: Optional[List[str]] = None
    validation_epochs: int = 1
    save_epochs: int = 1
    validation_num_inference_steps: int = 50
    validation_guidance_scale: float = 7.5
    validation_image_height: int = 1024
    validation_image_width: int = 1024
    validation_num_images_per_prompt: int = 1
    
    # Component configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    vae_args: VAEConfig = field(default_factory=VAEConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)

def parse_args() -> TrainingConfig:
    """
    Parse command line arguments and convert them into a structured config.

    Returns:
        TrainingConfig: The parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Train a Stable Diffusion XL model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Training mode
    parser.add_argument("--training_mode", type=str, default="v_prediction", choices=["v_prediction", "epsilon"])
    parser.add_argument("--use_ztsnr", action="store_true")
    parser.add_argument("--rescale_cfg", action="store_true")
    parser.add_argument("--rescale_multiplier", type=float, default=0.7)
    parser.add_argument("--resolution_scaling", action="store_true", default=True)
    parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    
    # Optimizer arguments
    parser.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw", "adamw8bit"])
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--use_8bit_adam", action="store_true")
    
    # EMA arguments
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmup_steps", type=int, default=2000)
    
    # System arguments
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Data arguments
    parser.add_argument("--validation_dir", type=str, help="Validation data directory")
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--no_caching", action="store_true")
    
    args = parser.parse_args()
    
    # Convert to config
    config = TrainingConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        training_mode=args.training_mode,
        use_ztsnr=args.use_ztsnr,
        rescale_cfg=args.rescale_cfg,
        rescale_multiplier=args.rescale_multiplier,
        resolution_scaling=args.resolution_scaling,
        min_snr_gamma=args.min_snr_gamma,
        validation_dir=args.validation_dir,
        cache_dir=args.cache_dir,
        no_caching=args.no_caching,
        enable_compile=args.enable_compile,
        compile_mode=args.compile_mode,
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
    )
    
    # Update optimizer config
    config.optimizer.optimizer_type = args.optimizer_type
    config.optimizer.learning_rate = args.learning_rate
    config.optimizer.weight_decay = args.weight_decay
    config.optimizer.use_8bit_adam = args.use_8bit_adam
    
    # Update EMA config
    config.ema.decay = args.ema_decay
    config.ema.warmup_steps = args.ema_warmup_steps
    
    return config
