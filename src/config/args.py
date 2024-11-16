"""Command line argument configuration for SDXL training.

This module provides a structured way to define and parse command line arguments
for the SDXL training pipeline, organized by functional categories.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    """Model-related arguments."""
    model_path: str
    output_dir: str = "./output"

@dataclass
class TrainingArgs:
    """Training hyperparameters and configuration."""
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    training_mode: str = "v_prediction"
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

@dataclass
class OptimizerArgs:
    """Optimizer configuration."""
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-2
    use_adafactor: bool = False
    use_8bit_adam: bool = False

@dataclass
class EMAArgs:
    """EMA model configuration."""
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100
    ema_power: float = 0.6667
    ema_min_decay: float = 0.0
    ema_max_decay: float = 0.9999
    ema_update_every: int = 1
    use_ema_warmup: bool = True

@dataclass
class VAEArgs:
    """VAE configuration and finetuning parameters."""
    use_vae: bool = True
    vae_path: Optional[str] = None
    vae_decay: float = 0.9999
    vae_update_after_step: int = 100
    finetune_vae: bool = False
    vae_learning_rate: float = 1e-6
    vae_train_freq: int = 10
    adaptive_loss_scale: bool = True
    kl_weight: float = 0.0
    perceptual_weight: float = 0.0
    vae_use_channel_scaling: bool = True
    vae_initial_scale_factor: float = 1.0

@dataclass
class DataArgs:
    """Data loading and processing configuration."""
    data_dir: str
    cache_dir: str = "latents_cache"
    no_caching: bool = False
    num_inference_steps: int = 28

@dataclass
class TagWeightingArgs:
    """Tag-based loss weighting configuration."""
    use_tag_weighting: bool = True
    min_tag_weight: float = 0.1
    max_tag_weight: float = 3.0

@dataclass
class SystemArgs:
    """System and optimization settings."""
    enable_compile: bool = False
    compile_mode: str = "default"
    gradient_checkpointing: bool = False
    verbose: bool = False

@dataclass
class LoggingArgs:
    """Logging and monitoring configuration."""
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    save_checkpoints: bool = False
    resume_from_checkpoint: Optional[str] = None
    push_to_hub: bool = False

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    model: ModelArgs
    training: TrainingArgs
    optimizer: OptimizerArgs
    ema: EMAArgs
    vae: VAEArgs
    data: DataArgs
    tag_weighting: TagWeightingArgs
    system: SystemArgs
    logging: LoggingArgs

def parse_args() -> TrainingConfig:
    """Parse command line arguments and return structured configuration."""
    parser = argparse.ArgumentParser(description="Train a Stable Diffusion XL model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--training_mode", type=str, default="v_prediction",
                       choices=["v_prediction", "epsilon"])
    parser.add_argument("--use_ztsnr", action="store_true")
    parser.add_argument("--rescale_cfg", action="store_true")
    parser.add_argument("--rescale_multiplier", type=float, default=0.7)
    parser.add_argument("--resolution_scaling", action="store_true", default=True)
    parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    parser.add_argument("--sigma_data", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.029)
    parser.add_argument("--sigma_max", type=float, default=160.0)
    parser.add_argument("--scale_method", type=str, default="karras",
                       choices=["karras", "simple"])
    parser.add_argument("--scale_factor", type=float, default=0.7)
    
    # Optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    
    # EMA arguments
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--ema_power", type=float, default=0.6667)
    parser.add_argument("--ema_min_decay", type=float, default=0.0)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--use_ema_warmup", action="store_true", default=True)
    
    # VAE arguments
    parser.add_argument("--use_vae", action="store_true", default=True)
    parser.add_argument("--vae_path", type=str)
    parser.add_argument("--vae_decay", type=float, default=0.9999)
    parser.add_argument("--vae_update_after_step", type=int, default=100)
    parser.add_argument("--finetune_vae", action="store_true")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6)
    parser.add_argument("--vae_train_freq", type=int, default=10)
    parser.add_argument("--adaptive_loss_scale", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.0)
    parser.add_argument("--vae_use_channel_scaling", action="store_true", default=True)
    parser.add_argument("--vae_initial_scale_factor", type=float, default=1.0)
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--no_caching", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    
    # Tag weighting arguments
    parser.add_argument("--use_tag_weighting", action="store_true", default=True)
    parser.add_argument("--min_tag_weight", type=float, default=0.1)
    parser.add_argument("--max_tag_weight", type=float, default=3.0)
    
    # System arguments
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str,
                       choices=["default", "reduce-overhead", "max-autotune"],
                       default="default")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    
    args = parser.parse_args()
    
    # Convert namespace to structured config
    config = TrainingConfig(
        model=ModelArgs(
            model_path=args.model_path,
            output_dir=args.output_dir
        ),
        training=TrainingArgs(
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=args.warmup_steps,
            training_mode=args.training_mode,
            use_ztsnr=args.use_ztsnr,
            rescale_cfg=args.rescale_cfg,
            rescale_multiplier=args.rescale_multiplier,
            resolution_scaling=args.resolution_scaling,
            min_snr_gamma=args.min_snr_gamma,
            sigma_data=args.sigma_data,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            scale_method=args.scale_method,
            scale_factor=args.scale_factor
        ),
        optimizer=OptimizerArgs(
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
            use_adafactor=args.use_adafactor,
            use_8bit_adam=args.use_8bit_adam
        ),
        ema=EMAArgs(
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            ema_update_after_step=args.ema_update_after_step,
            ema_power=args.ema_power,
            ema_min_decay=args.ema_min_decay,
            ema_max_decay=args.ema_max_decay,
            ema_update_every=args.ema_update_every,
            use_ema_warmup=args.use_ema_warmup
        ),
        vae=VAEArgs(
            use_vae=args.use_vae,
            vae_path=args.vae_path,
            vae_decay=args.vae_decay,
            vae_update_after_step=args.vae_update_after_step,
            finetune_vae=args.finetune_vae,
            vae_learning_rate=args.vae_learning_rate,
            vae_train_freq=args.vae_train_freq,
            adaptive_loss_scale=args.adaptive_loss_scale,
            kl_weight=args.kl_weight,
            perceptual_weight=args.perceptual_weight,
            vae_use_channel_scaling=args.vae_use_channel_scaling,
            vae_initial_scale_factor=args.vae_initial_scale_factor
        ),
        data=DataArgs(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            no_caching=args.no_caching,
            num_inference_steps=args.num_inference_steps
        ),
        tag_weighting=TagWeightingArgs(
            use_tag_weighting=args.use_tag_weighting,
            min_tag_weight=args.min_tag_weight,
            max_tag_weight=args.max_tag_weight
        ),
        system=SystemArgs(
            enable_compile=args.enable_compile,
            compile_mode=args.compile_mode,
            gradient_checkpointing=args.gradient_checkpointing,
            verbose=args.verbose
        ),
        logging=LoggingArgs(
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            save_checkpoints=args.save_checkpoints,
            resume_from_checkpoint=args.resume_from_checkpoint,
            push_to_hub=args.push_to_hub
        )
    )
    
    return config
