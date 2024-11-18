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
    # Basic training params
    enable_vae_finetuning: bool = False
    vae_path: Optional[str] = None
    learning_rate: float = 1e-6
    batch_size: int = 1
    num_epochs: int = 1
    
    # Training optimizations
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    
    # VAE specific settings
    use_channel_scaling: bool = True
    enable_cuda_graphs: bool = False
    cache_size: int = 10000
    num_warmup_steps: int = 100
    
    # Additional settings from original config
    train_freq: int = 10
    kl_weight: float = 0.0
    perceptual_weight: float = 0.0
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
class WandBConfig:
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    logging_steps: int = 50

@dataclass
class TrainingConfig:
    # Required parameters
    model_path: str
    data_dir: str
    
    # Output configuration
    output_dir: str = "output"
    cache_dir: Optional[str] = None 
    
    # Training hyperparameters
    batch_size: int = 1
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    
    # Model and training mode configuration
    training_mode: str = "v_prediction"
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    use_8bit_adam: bool = False
    use_ema: bool = True
    
    # Component configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    vae_args: VAEConfig = field(default_factory=VAEConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    
    # SDXL specific parameters
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
    all_ar: bool = False

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
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_epochs", type=int, default=1)
    
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
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--no_caching", action="store_true")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every N steps")
    
    # Training control arguments
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save checkpoints during training")
    parser.add_argument("--all_ar", action="store_true", help="Use all aspect ratios for training")
    
    # VAE arguments
    vae_group = parser.add_argument_group('VAE training arguments')
    vae_group.add_argument("--enable_vae_finetuning", action="store_true",
                        help="Enable VAE finetuning")
    vae_group.add_argument("--vae_path", type=str,
                        help="Path to pretrained VAE model")
    vae_group.add_argument("--vae_learning_rate", type=float, default=1e-6,
                        help="Learning rate for VAE training")
    vae_group.add_argument("--vae_batch_size", type=int, default=1,
                        help="Batch size for VAE training")
    vae_group.add_argument("--vae_num_epochs", type=int, default=1,
                        help="Number of epochs for VAE training")
    vae_group.add_argument("--vae_mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode for VAE training")
    vae_group.add_argument("--vae_use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer for VAE")
    vae_group.add_argument("--vae_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for VAE")
    vae_group.add_argument("--vae_max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for VAE training")
    vae_group.add_argument("--vae_use_channel_scaling", action="store_true",
                        help="Enable channel scaling for VAE")
    vae_group.add_argument("--vae_enable_cuda_graphs", action="store_true",
                        help="Enable CUDA graphs for VAE")
    vae_group.add_argument("--vae_cache_size", type=int, default=10000,
                        help="Cache size for VAE training")
    vae_group.add_argument("--vae_num_warmup_steps", type=int, default=100,
                        help="Number of warmup steps for VAE training")
    vae_group.add_argument("--vae_train_freq", type=int, default=10,
                        help="Frequency of VAE training steps")
    vae_group.add_argument("--vae_kl_weight", type=float, default=0.0,
                        help="Weight for KL divergence loss")
    vae_group.add_argument("--vae_perceptual_weight", type=float, default=0.0,
                        help="Weight for perceptual loss")
    vae_group.add_argument("--vae_initial_scale_factor", type=float, default=1.0,
                        help="Initial scale factor for VAE training")
    
    # Add sampling parameters
    parser.add_argument("--sigma_min", type=float, default=0.029,
                      help="Minimum sigma value for sampling")
    parser.add_argument("--scale_method", type=str, default="karras",
                      choices=["karras", "simple"],
                      help="Method for scaling noise schedule")
    
    # Add tag processing arguments
    parser.add_argument("--use_tag_weighting", action="store_true",
                      help="Enable tag weighting based on frequency and emphasis")
    
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
        cache_dir=args.cache_dir,
        no_caching=args.no_caching,
        enable_compile=args.enable_compile,
        compile_mode=args.compile_mode,
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
        save_epochs=args.save_epochs,
        warmup_steps=args.warmup_steps,
        save_checkpoints=args.save_checkpoints,
        all_ar=args.all_ar,
        sigma_min=args.sigma_min,
        scale_method=args.scale_method,
        use_tag_weighting=args.use_tag_weighting,
    )
    
    # Update optimizer config
    config.optimizer.optimizer_type = args.optimizer_type
    config.optimizer.learning_rate = args.learning_rate
    config.optimizer.weight_decay = args.weight_decay
    config.optimizer.use_8bit_adam = args.use_8bit_adam
    
    # Update EMA config
    config.ema.decay = args.ema_decay
    config.ema.warmup_steps = args.ema_warmup_steps
    
    # Update WandB config
    config.wandb.use_wandb = args.use_wandb
    config.wandb.wandb_project = args.wandb_project
    config.wandb.wandb_run_name = args.wandb_run_name
    config.wandb.logging_steps = args.logging_steps

    # Update VAE config
    config.vae_args.enable_vae_finetuning = args.enable_vae_finetuning
    config.vae_args.vae_path = args.vae_path
    config.vae_args.learning_rate = args.vae_learning_rate
    config.vae_args.batch_size = args.vae_batch_size
    config.vae_args.num_epochs = args.vae_num_epochs
    config.vae_args.mixed_precision = args.vae_mixed_precision
    config.vae_args.use_8bit_adam = args.vae_use_8bit_adam
    config.vae_args.gradient_checkpointing = args.vae_gradient_checkpointing
    config.vae_args.max_grad_norm = args.vae_max_grad_norm
    config.vae_args.use_channel_scaling = args.vae_use_channel_scaling
    config.vae_args.enable_cuda_graphs = args.vae_enable_cuda_graphs
    config.vae_args.cache_size = args.vae_cache_size
    config.vae_args.num_warmup_steps = args.vae_num_warmup_steps
    
    return config
