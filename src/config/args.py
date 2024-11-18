"""Command line argument configuration for SDXL training.

This module provides a structured way to define and parse command line arguments
for the SDXL training pipeline, organized by functional categories.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

def load_defaults():
    """Load default configuration values from defaults.json"""
    defaults_path = Path(__file__).parent / "defaults.json"
    with open(defaults_path) as f:
        return json.load(f)

DEFAULTS = load_defaults()

@dataclass
class OptimizerConfig:
    learning_rate: float = DEFAULTS["optimizer"]["learning_rate"]
    weight_decay: float = DEFAULTS["optimizer"]["weight_decay"]
    optimizer_type: str = DEFAULTS["optimizer"]["optimizer_type"]
    adam_beta1: float = DEFAULTS["optimizer"]["adam_beta1"]
    adam_beta2: float = DEFAULTS["optimizer"]["adam_beta2"]
    adam_epsilon: float = DEFAULTS["optimizer"]["adam_epsilon"]
    use_8bit_adam: bool = DEFAULTS["optimizer"]["use_8bit_adam"]

@dataclass
class SchedulerConfig:
    use_scheduler: bool = DEFAULTS["scheduler"]["use_scheduler"]
    num_warmup_steps: int = DEFAULTS["scheduler"]["num_warmup_steps"]
    num_training_steps: int = DEFAULTS["scheduler"]["num_training_steps"]
    num_cycles: int = DEFAULTS["scheduler"]["num_cycles"]

@dataclass
class EMAConfig:
    decay: float = DEFAULTS["ema"]["decay"]
    update_after_step: int = DEFAULTS["ema"]["update_after_step"]
    use_warmup: bool = DEFAULTS["ema"]["use_warmup"]
    warmup_steps: int = DEFAULTS["ema"]["warmup_steps"]

@dataclass
class VAEConfig:
    enable_vae_finetuning: bool = DEFAULTS["vae"]["enable_vae_finetuning"]
    vae_path: Optional[str] = None
    learning_rate: float = DEFAULTS["vae"]["learning_rate"]
    batch_size: int = DEFAULTS["vae"]["batch_size"]
    num_epochs: int = DEFAULTS["vae"]["num_epochs"]
    mixed_precision: str = DEFAULTS["vae"]["mixed_precision"]
    use_8bit_adam: bool = DEFAULTS["vae"]["use_8bit_adam"]
    gradient_checkpointing: bool = DEFAULTS["vae"]["gradient_checkpointing"]
    max_grad_norm: float = DEFAULTS["vae"]["max_grad_norm"]
    use_channel_scaling: bool = DEFAULTS["vae"]["use_channel_scaling"]
    enable_cuda_graphs: bool = DEFAULTS["vae"]["enable_cuda_graphs"]
    cache_size: int = DEFAULTS["vae"]["cache_size"]
    num_warmup_steps: int = DEFAULTS["vae"]["num_warmup_steps"]
    train_freq: int = DEFAULTS["vae"]["train_freq"]
    kl_weight: float = DEFAULTS["vae"]["kl_weight"]
    perceptual_weight: float = DEFAULTS["vae"]["perceptual_weight"]
    initial_scale_factor: float = DEFAULTS["vae"]["initial_scale_factor"]

@dataclass
class TagWeightingConfig:
    token_dropout_rate: float = DEFAULTS["tag_weighting"]["token_dropout_rate"]
    caption_dropout_rate: float = DEFAULTS["tag_weighting"]["caption_dropout_rate"]
    rarity_factor: float = DEFAULTS["tag_weighting"]["rarity_factor"]
    emphasis_factor: float = DEFAULTS["tag_weighting"]["emphasis_factor"]
    min_tag_freq: int = DEFAULTS["tag_weighting"]["min_tag_freq"]
    min_cluster_size: int = DEFAULTS["tag_weighting"]["min_cluster_size"]
    similarity_threshold: float = DEFAULTS["tag_weighting"]["similarity_threshold"]

@dataclass
class WandBConfig:
    use_wandb: bool = DEFAULTS["wandb"]["use_wandb"]
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    logging_steps: int = DEFAULTS["wandb"]["logging_steps"]

@dataclass
class CachingConfig:
    vae_cache_size: int = DEFAULTS["caching"]["vae_cache"]["max_cache_size"]
    vae_cache_num_workers: int = DEFAULTS["caching"]["vae_cache"]["num_workers"]
    vae_cache_batch_size: int = DEFAULTS["caching"]["vae_cache"]["batch_size"]
    text_cache_size: int = DEFAULTS["caching"]["text_cache"]["max_cache_size"]
    text_cache_num_workers: int = DEFAULTS["caching"]["text_cache"]["num_workers"]
    text_cache_batch_size: int = DEFAULTS["caching"]["text_cache"]["batch_size"]

@dataclass
class TrainingConfig:
    # Required parameters
    pretrained_model_path: str  # Changed from model_path
    train_data_dir: str  
    
    # Output configuration
    output_dir: str = DEFAULTS["training"]["output_dir"]
    cache_dir: Optional[str] = None
    no_caching: bool = False
    
    # Training hyperparameters
    batch_size: int = DEFAULTS["training"]["batch_size"]
    num_epochs: int = DEFAULTS["training"]["num_epochs"]
    gradient_accumulation_steps: int = DEFAULTS["training"]["gradient_accumulation_steps"]
    learning_rate: float = DEFAULTS["training"]["learning_rate"]
    max_grad_norm: float = DEFAULTS["training"]["max_grad_norm"]
    warmup_steps: int = DEFAULTS["training"]["warmup_steps"]
    save_checkpoints: bool = False
    use_tag_weighting: bool = False
    rescale_cfg: bool = False
    save_epochs: int = DEFAULTS["training"]["save_epochs"]
    
    # Model and training mode configuration
    training_mode: str = DEFAULTS["training"]["training_mode"]
    mixed_precision: str = DEFAULTS["training"]["mixed_precision"]
    gradient_checkpointing: bool = DEFAULTS["training"]["gradient_checkpointing"]
    use_8bit_adam: bool = DEFAULTS["training"]["use_8bit_adam"]
    use_ema: bool = DEFAULTS["training"]["use_ema"]
    train_text_encoder: bool = DEFAULTS["training"]["train_text_encoder"]
    
    # System configuration
    enable_compile: bool = DEFAULTS["training"]["enable_compile"]
    compile_mode: str = DEFAULTS["training"]["compile_mode"]
    num_workers: int = DEFAULTS["training"]["num_workers"]
    device: str = DEFAULTS["training"]["device"]
    
    # Component configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    vae_args: VAEConfig = field(default_factory=VAEConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    
    # Validation parameters
    validation_epochs: int = DEFAULTS["validation"]["validation_epochs"]
    validation_prompts: Optional[List[str]] = field(default_factory=lambda: DEFAULTS["validation"]["prompts"])
    validation_num_inference_steps: int = DEFAULTS["validation"]["num_inference_steps"]
    validation_guidance_scale: float = DEFAULTS["validation"]["guidance_scale"]
    validation_image_height: int = DEFAULTS["validation"]["height"]
    validation_image_width: int = DEFAULTS["validation"]["width"]
    validation_num_images_per_prompt: int = DEFAULTS["validation"]["num_images_per_prompt"]

def parse_args() -> TrainingConfig:
    """Parse command line arguments and convert them into a structured config."""
    parser = argparse.ArgumentParser(description="Train a Stable Diffusion XL model")
    
    # Required arguments
    parser.add_argument("--pretrained_model_path", type=str, required=True, 
                       help="Path to pretrained model")
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Training data directory")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULTS["training"]["warmup_steps"],
                       help="Number of warmup steps")
    parser.add_argument("--save_checkpoints", action="store_true",
                       help="Enable checkpoint saving")
    parser.add_argument("--use_tag_weighting", action="store_true",
                       default=False,
                       help="Enable tag weighting for training")
    parser.add_argument("--rescale_cfg", action="store_true",
                       default=False,
                       help="Enable CFG rescaling")
    
    # Optional arguments with defaults from json
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["training"]["output_dir"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["training"]["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULTS["training"]["num_epochs"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULTS["training"]["learning_rate"])
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                       default=DEFAULTS["training"]["gradient_accumulation_steps"])
    parser.add_argument("--mixed_precision", type=str, default=DEFAULTS["training"]["mixed_precision"],
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["training"]["max_grad_norm"])
    parser.add_argument("--save_epochs", type=int, default=DEFAULTS["training"]["save_epochs"])
    
    # Training mode and related settings
    parser.add_argument("--training_mode", type=str, default=DEFAULTS["training"]["training_mode"],
                       choices=["v_prediction", "epsilon"])
    parser.add_argument("--use_ztsnr", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       default=DEFAULTS["training"]["gradient_checkpointing"])
    
    # Optimizer arguments
    parser.add_argument("--optimizer_type", type=str, default=DEFAULTS["optimizer"]["optimizer_type"],
                       choices=["adamw", "adamw8bit"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["optimizer"]["weight_decay"])
    parser.add_argument("--use_8bit_adam", action="store_true",
                       default=DEFAULTS["optimizer"]["use_8bit_adam"])
    
    # EMA arguments
    parser.add_argument("--use_ema", action="store_true", default=DEFAULTS["training"]["use_ema"])
    parser.add_argument("--ema_decay", type=float, default=DEFAULTS["ema"]["decay"])
    parser.add_argument("--ema_warmup_steps", type=int, default=DEFAULTS["ema"]["warmup_steps"])
    
    # System arguments
    parser.add_argument("--enable_compile", action="store_true",
                       default=DEFAULTS["training"]["enable_compile"])
    parser.add_argument("--compile_mode", type=str, default=DEFAULTS["training"]["compile_mode"],
                       choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["training"]["num_workers"])
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=DEFAULTS["wandb"]["use_wandb"])
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--logging_steps", type=int, default=DEFAULTS["wandb"]["logging_steps"])
    
    # VAE arguments
    vae_group = parser.add_argument_group('VAE training arguments')
    vae_group.add_argument("--enable_vae_finetuning", action="store_true",
                          default=DEFAULTS["vae"]["enable_vae_finetuning"])
    vae_group.add_argument("--vae_path", type=str)
    vae_group.add_argument("--vae_learning_rate", type=float,
                          default=DEFAULTS["vae"]["learning_rate"])
    vae_group.add_argument("--vae_batch_size", type=int,
                          default=DEFAULTS["vae"]["batch_size"])
    vae_group.add_argument("--vae_num_epochs", type=int,
                          default=DEFAULTS["vae"]["num_epochs"])
    vae_group.add_argument("--vae_mixed_precision", type=str,
                          default=DEFAULTS["vae"]["mixed_precision"],
                          choices=["no", "fp16", "bf16"])
    
    args = parser.parse_args()
    
    # Convert to config
    config = TrainingConfig(
        pretrained_model_path=args.pretrained_model_path,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        training_mode=args.training_mode,
        enable_compile=args.enable_compile,
        compile_mode=args.compile_mode,
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
        save_epochs=args.save_epochs,
        use_ema=args.use_ema,
        use_8bit_adam=args.use_8bit_adam,
        warmup_steps=args.warmup_steps,  # Add new field
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
    
    return config