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
    with open(defaults_path, 'r', encoding='utf-8') as f:
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
    lion_betas: tuple = (0.95, 0.98)  # NAI recommended values

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
class TagWeightingConfig:
    token_dropout_rate: float = DEFAULTS["tag_weighting"]["token_dropout_rate"]
    caption_dropout_rate: float = DEFAULTS["tag_weighting"]["caption_dropout_rate"]
    rarity_factor: float = DEFAULTS["tag_weighting"]["rarity_factor"]
    emphasis_factor: float = DEFAULTS["tag_weighting"]["emphasis_factor"]
    min_tag_freq: int = DEFAULTS["tag_weighting"]["min_tag_freq"]
    min_cluster_size: int = DEFAULTS["tag_weighting"]["min_cluster_size"]
    similarity_threshold: float = DEFAULTS["tag_weighting"]["similarity_threshold"]

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
    num_workers: int = DEFAULTS["vae"]["num_workers"]
    num_warmup_steps: int = DEFAULTS["vae"]["num_warmup_steps"]
    train_freq: int = DEFAULTS["vae"]["train_freq"]
    kl_weight: float = DEFAULTS["vae"]["kl_weight"]
    perceptual_weight: float = DEFAULTS["vae"]["perceptual_weight"]
    initial_scale_factor: float = DEFAULTS["vae"]["initial_scale_factor"]
    max_resolution: int = DEFAULTS["vae"]["max_resolution"]
    min_resolution: int = DEFAULTS["vae"]["min_resolution"]
    resolution_type: str = DEFAULTS["vae"]["resolution_type"]
    bucket_resolution_steps: int = DEFAULTS["vae"]["bucket_resolution_steps"]
    bucket_no_upscale: bool = DEFAULTS["vae"]["bucket_no_upscale"]
    random_crop: bool = DEFAULTS["vae"]["random_crop"]
    random_flip: bool = DEFAULTS["vae"]["random_flip"]
    shuffle_tags: bool = DEFAULTS["vae"]["shuffle_tags"]
    keep_tokens: int = DEFAULTS["vae"]["keep_tokens"]
    caption_dropout_probability: float = 0.0
    caption_tag_dropout_probability: float = 0.0
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)



@dataclass
class WandBConfig:
    """WandB configuration settings."""
    use_wandb: bool = DEFAULTS["wandb"]["use_wandb"]
    project: str = DEFAULTS["wandb"].get("project", "")
    run_name: str = DEFAULTS["wandb"].get("run_name", "")
    logging_steps: int = DEFAULTS["wandb"]["logging_steps"]
    log_model: bool = DEFAULTS["wandb"].get("log_model", False)
    window_size: int = DEFAULTS["wandb"].get("window_size", 100)

@dataclass
class CachingConfig:
    """Configuration for VAE and text embedding caches."""
    # VAE cache settings
    vae_cache_size: int = DEFAULTS["caching"]["vae_cache_size"]
    vae_cache_num_workers: int = DEFAULTS["caching"]["vae_cache_num_workers"]
    vae_cache_batch_size: int = DEFAULTS["caching"]["vae_cache_batch_size"]
    vae_cache_memory_gb: float = DEFAULTS["caching"]["vae_cache_memory_gb"]
    
    # Text embedding cache settings
    text_cache_size: int = DEFAULTS["caching"]["text_cache_size"]
    text_cache_num_workers: int = DEFAULTS["caching"]["text_cache_num_workers"]
    text_cache_batch_size: int = DEFAULTS["caching"]["text_cache_batch_size"]
    text_cache_memory_gb: float = DEFAULTS["caching"]["text_cache_memory_gb"]

@dataclass
class TrainingConfig:
    # Required parameters
    pretrained_model_path: str  
    train_data_dir: str  
    
    # Training parameters
    output_dir: str = DEFAULTS["training"]["output_dir"]
    cache_dir: Optional[str] = None
    no_caching: bool = False
    cache_size: int = DEFAULTS["training"]["cache_size"]
    
    # Image configuration
    max_resolution: int = DEFAULTS["training"]["max_resolution"]
    resolution_type: str = DEFAULTS["training"]["resolution_type"]
    
    # Training hyperparameters
    batch_size: int = DEFAULTS["training"]["batch_size"]
    num_epochs: int = DEFAULTS["training"]["num_epochs"]
    gradient_accumulation_steps: int = DEFAULTS["training"]["gradient_accumulation_steps"]
    learning_rate: float = DEFAULTS["optimizer"]["learning_rate"]
    min_learning_rate: float = DEFAULTS["optimizer"].get("min_learning_rate", 1e-6)
    max_grad_norm: float = DEFAULTS["training"]["max_grad_norm"]
    warmup_steps: int = DEFAULTS["training"]["warmup_steps"]
    save_checkpoints: bool = False
    use_tag_weighting: bool = False
    rescale_cfg: bool = False
    save_epochs: int = DEFAULTS["training"]["save_epochs"]
    
    # Model and training mode configuration
    training_mode: str = DEFAULTS["training"]["training_mode"]
    mixed_precision: str = DEFAULTS["training"]["mixed_precision"]
    
    # Model settings
    gradient_checkpointing: bool = DEFAULTS["training"]["gradient_checkpointing"]
    use_8bit_adam: bool = DEFAULTS["training"]["use_8bit_adam"]
    use_ema: bool = DEFAULTS["training"]["use_ema"]
    train_text_encoder: bool = DEFAULTS["training"]["train_text_encoder"]
    
    # System configuration
    enable_compile: bool = DEFAULTS["training"]["enable_compile"]
    compile_mode: str = DEFAULTS["training"]["compile_mode"]
    num_workers: int = DEFAULTS["training"]["num_workers"]
    device: str = DEFAULTS["training"]["device"]
    cache_size: int = DEFAULTS["training"]["cache_size"]
    
    # Resolution settings
    max_resolution: int = DEFAULTS["training"]["max_resolution"]
    resolution_type: str = DEFAULTS["training"]["resolution_type"]
    
    # Training improvements
    use_min_snr: bool = DEFAULTS["training"]["use_min_snr"]
    min_snr_gamma: float = DEFAULTS["training"]["min_snr_gamma"]
    use_ztsnr: bool = DEFAULTS["training"]["use_ztsnr"]
    ztsnr_sigma: float = DEFAULTS["training"]["ztsnr_sigma"]
    
    # EDM settings
    sigma_min: float = DEFAULTS["training"]["sigma_min"]
    sigma_max: Optional[float] = DEFAULTS["training"]["sigma_max"]
    rho: float = DEFAULTS["training"]["rho"]
    
    # Inference settings
    num_inference_steps: int = DEFAULTS["validation"]["num_inference_steps"]
    guidance_scale: float = DEFAULTS["validation"]["guidance_scale"]
    
    # Component configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    vae_args: VAEConfig = field(default_factory=VAEConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    
    # Validation settings
    validation_epochs: int = DEFAULTS["validation"]["validation_epochs"]
    validation_prompts: List[str] = field(
        default_factory=lambda: DEFAULTS["validation"]["prompts"]
    )
    validation_num_inference_steps: int = DEFAULTS["validation"]["num_inference_steps"]
    validation_guidance_scale: float = DEFAULTS["validation"]["guidance_scale"]
    validation_image_height: int = DEFAULTS["validation"]["height"]
    validation_image_width: int = DEFAULTS["validation"]["width"]
    validation_num_images_per_prompt: int = DEFAULTS["validation"]["num_images_per_prompt"]
    
    # Add optimizer parameters
    adam_beta1: float = DEFAULTS["optimizer"]["adam_beta1"]
    adam_beta2: float = DEFAULTS["optimizer"]["adam_beta2"]
    adam_epsilon: float = DEFAULTS["optimizer"]["adam_epsilon"]
    adam_weight_decay: float = DEFAULTS["optimizer"]["weight_decay"]
    
    # Add scheduler parameters
    scheduler_type: str = DEFAULTS["scheduler"].get("scheduler_type", "cosine")
    num_warmup_steps: int = DEFAULTS["scheduler"]["num_warmup_steps"]
    num_training_steps: int = DEFAULTS["scheduler"]["num_training_steps"]
    num_cycles: float = DEFAULTS["scheduler"]["num_cycles"]

    @property
    def use_wandb(self) -> bool:
        return self.wandb.use_wandb

    @property
    def wandb_enabled(self) -> bool:
        return self.wandb.use_wandb and self.wandb.project is not None

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
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                       default=DEFAULTS["training"]["gradient_accumulation_steps"])
    parser.add_argument("--mixed_precision", type=str, default=DEFAULTS["training"]["mixed_precision"],
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["training"]["max_grad_norm"])
    parser.add_argument("--save_epochs", type=int, default=DEFAULTS["training"]["save_epochs"])
    parser.add_argument("--cache_size", type=int, 
                       default=DEFAULTS["training"]["cache_size"],
                       help="Size of the cache for VAE and text embeddings")
    
    # Add resolution arguments
    parser.add_argument("--max_resolution", type=int, 
                       default=DEFAULTS["training"]["max_resolution"],
                       help="Maximum resolution for training images")
    parser.add_argument("--resolution_type", type=str,
                       default=DEFAULTS["training"]["resolution_type"],
                       choices=["pixel", "area"],
                       help="How to interpret max_resolution (pixel or area based)")
    
    # Training mode and related settings
    parser.add_argument("--training_mode", type=str, default=DEFAULTS["training"]["training_mode"],
                       choices=["v_prediction", "epsilon"])
    parser.add_argument("--use_ztsnr", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       default=DEFAULTS["training"]["gradient_checkpointing"])
    
    # Optimizer arguments
    parser.add_argument("--optimizer_type", type=str, 
                       default=DEFAULTS["optimizer"]["optimizer_type"],
                       choices=["adamw", "adamw8bit", "lion"])
    parser.add_argument("--learning_rate", type=float,
                       default=DEFAULTS["optimizer"]["learning_rate"])
    parser.add_argument("--min_learning_rate", type=float,
                       default=DEFAULTS["optimizer"].get("min_learning_rate", 1e-6))
    parser.add_argument("--weight_decay", type=float,
                       default=DEFAULTS["optimizer"]["weight_decay"])
    parser.add_argument("--adam_beta1", type=float,
                       default=DEFAULTS["optimizer"]["adam_beta1"])
    parser.add_argument("--adam_beta2", type=float,
                       default=DEFAULTS["optimizer"]["adam_beta2"])
    parser.add_argument("--adam_epsilon", type=float,
                       default=DEFAULTS["optimizer"]["adam_epsilon"])
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
    wandb_group = parser.add_argument_group("WandB Configuration")
    wandb_group.add_argument("--use_wandb", action="store_true",
                            default=DEFAULTS["wandb"]["use_wandb"],
                            help="Enable WandB logging")
    wandb_group.add_argument("--wandb_project", type=str,
                            help="WandB project name")
    wandb_group.add_argument("--wandb_run_name", type=str,
                            help="WandB run name")
    wandb_group.add_argument("--logging_steps", type=int,
                            default=DEFAULTS["wandb"]["logging_steps"],
                            help="Log every N steps")
    wandb_group.add_argument("--wandb_log_model", action="store_true",
                            default=DEFAULTS["wandb"].get("log_model", False),
                            help="Log model checkpoints to WandB")
    wandb_group.add_argument("--wandb_window_size", type=int,
                            default=DEFAULTS["wandb"].get("window_size", 100),
                            help="Window size for moving averages")
    
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
    
    # Tag weighting arguments
    tag_group = parser.add_argument_group('Tag weighting arguments')
    tag_group.add_argument("--token_dropout_rate", type=float,
                          default=DEFAULTS["tag_weighting"]["token_dropout_rate"])
    tag_group.add_argument("--caption_dropout_rate", type=float,
                          default=DEFAULTS["tag_weighting"]["caption_dropout_rate"])
    tag_group.add_argument("--rarity_factor", type=float,
                          default=DEFAULTS["tag_weighting"]["rarity_factor"])
    tag_group.add_argument("--emphasis_factor", type=float,
                          default=DEFAULTS["tag_weighting"]["emphasis_factor"])
    tag_group.add_argument("--min_tag_freq", type=int,
                          default=DEFAULTS["tag_weighting"]["min_tag_freq"])
    tag_group.add_argument("--min_cluster_size", type=int,
                          default=DEFAULTS["tag_weighting"]["min_cluster_size"])
    tag_group.add_argument("--similarity_threshold", type=float,
                          default=DEFAULTS["tag_weighting"]["similarity_threshold"])
    
    # Caching arguments
    cache_group = parser.add_argument_group('Caching arguments')
    cache_group.add_argument("--vae_cache_size", type=int,
                           default=DEFAULTS["caching"]["vae_cache"]["max_cache_size"])
    cache_group.add_argument("--vae_cache_workers", type=int,
                           default=DEFAULTS["caching"]["vae_cache"]["num_workers"])
    cache_group.add_argument("--vae_cache_batch_size", type=int,
                           default=DEFAULTS["caching"]["vae_cache"]["batch_size"])
    cache_group.add_argument("--text_cache_size", type=int,
                           default=DEFAULTS["caching"]["text_cache"]["max_cache_size"])
    cache_group.add_argument("--text_cache_workers", type=int,
                           default=DEFAULTS["caching"]["text_cache"]["num_workers"])
    cache_group.add_argument("--text_cache_batch_size", type=int,
                           default=DEFAULTS["caching"]["text_cache"]["batch_size"])

    
    # Add scheduler arguments
    parser.add_argument("--scheduler_type", type=str,
                       default=DEFAULTS["scheduler"].get("scheduler_type", "cosine"))
    parser.add_argument("--num_warmup_steps", type=int,
                       default=DEFAULTS["scheduler"]["num_warmup_steps"])
    parser.add_argument("--num_training_steps", type=int,
                       default=DEFAULTS["scheduler"]["num_training_steps"])
    parser.add_argument("--num_cycles", type=float,
                       default=DEFAULTS["scheduler"]["num_cycles"])
    
    args = parser.parse_args()
    
    # Convert to config
    config = TrainingConfig(
        pretrained_model_path=args.pretrained_model_path,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        cache_size=args.cache_size,
        max_resolution=args.max_resolution,
        resolution_type=args.resolution_type,
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
        warmup_steps=args.warmup_steps,
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
    config.wandb.project = args.wandb_project
    config.wandb.run_name = args.wandb_run_name
    config.wandb.logging_steps = args.logging_steps
    config.wandb.log_model = args.wandb_log_model
    config.wandb.window_size = args.wandb_window_size
    
    # Update VAE config
    config.vae_args.enable_vae_finetuning = args.enable_vae_finetuning
    config.vae_args.vae_path = args.vae_path
    config.vae_args.learning_rate = args.vae_learning_rate
    config.vae_args.batch_size = args.vae_batch_size
    config.vae_args.num_epochs = args.vae_num_epochs
    config.vae_args.mixed_precision = args.vae_mixed_precision
    config.vae_args.num_workers = args.num_workers
    
    # Add tag weighting configuration to VAE config
    config.vae_args.tag_weighting.token_dropout_rate = args.token_dropout_rate
    config.vae_args.tag_weighting.caption_dropout_rate = args.caption_dropout_rate
    config.vae_args.tag_weighting.rarity_factor = args.rarity_factor
    config.vae_args.tag_weighting.emphasis_factor = args.emphasis_factor
    config.vae_args.tag_weighting.min_tag_freq = args.min_tag_freq
    config.vae_args.tag_weighting.min_cluster_size = args.min_cluster_size
    config.vae_args.tag_weighting.similarity_threshold = args.similarity_threshold
    
    # Update caching config
    config.caching.vae_cache_size = args.vae_cache_size
    config.caching.vae_cache_num_workers = args.vae_cache_workers
    config.caching.vae_cache_batch_size = args.vae_cache_batch_size
    config.caching.text_cache_size = args.text_cache_size
    config.caching.text_cache_num_workers = args.text_cache_workers
    config.caching.text_cache_batch_size = args.text_cache_batch_size
    
    return config