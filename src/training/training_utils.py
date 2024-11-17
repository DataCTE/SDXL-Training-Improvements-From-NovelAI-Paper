import torch
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from src.data.prompt.caption_processor import CaptionProcessor
from src.training.optimizers.setup_optimizers import setup_optimizer
from src.training.ema import setup_ema_model
from src.training.vae_finetuner import setup_vae_finetuner
from src.training.loss_functions import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _get_tag_weighter_config(
    token_dropout_rate: float = 0.1,
    caption_dropout_rate: float = 0.1,
    rarity_factor: float = 0.9,
    emphasis_factor: float = 1.2,
    min_tag_freq: int = 10,
    min_cluster_size: int = 5,
    similarity_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Cache tag weighter configuration."""
    return {
        "token_dropout_rate": token_dropout_rate,
        "caption_dropout_rate": caption_dropout_rate,
        "rarity_factor": rarity_factor,
        "emphasis_factor": emphasis_factor,
        "min_tag_freq": min_tag_freq,
        "min_cluster_size": min_cluster_size,
        "similarity_threshold": similarity_threshold,
    }

def setup_tag_weighter(args) -> Optional[CaptionProcessor]:
    """
    Initialize tag weighting system with CaptionProcessor.
    
    Args:
        args: Configuration arguments containing tag weighting settings
        
    Returns:
        Configured CaptionProcessor instance or None if tag weighting is disabled
        
    Raises:
        ValueError: If required configuration is missing
        RuntimeError: If initialization fails
    """
    try:
        if not getattr(args, "use_tag_weighting", False):
            return None

        # Get cached config
        config = _get_tag_weighter_config(
            token_dropout_rate=args.tag_weighting.token_dropout_rate,
            caption_dropout_rate=args.tag_weighting.caption_dropout_rate,
            rarity_factor=args.tag_weighting.rarity_factor,
            emphasis_factor=args.tag_weighting.emphasis_factor,
            min_tag_freq=args.tag_weighting.min_tag_freq,
            min_cluster_size=args.tag_weighting.min_cluster_size,
            similarity_threshold=args.tag_weighting.similarity_threshold,
        )

        # Initialize processor
        processor = CaptionProcessor(**config)
        logger.info("Tag weighter initialized successfully")
        return processor

    except Exception as e:
        logger.error("Failed to initialize tag weighter: %s", str(e))
        raise

def _validate_components(components: Dict[str, Any]) -> None:
    """Validate initialized components."""
    required_components = ["optimizer", "train_dataloader"]
    for component in required_components:
        if component not in components:
            raise ValueError(f"Missing required component: {component}")

def _cleanup_failed_initialization(components: Dict[str, Any]) -> None:
    """Clean up resources in case of failed initialization."""
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Close dataloaders
        for key in ["train_dataloader", "val_dataloader"]:
            if key in components:
                try:
                    components[key].dataset.close()
                except:
                    pass

    except Exception as e:
        logger.error("Cleanup failed: %s", str(e))

def initialize_training_components(
    args,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Initialize all training components with proper error handling.
    
    Args:
        args: Training configuration
        models: Dictionary containing models
        
    Returns:
        Dictionary of initialized components
        
    Raises:
        ValueError: If initialization fails
    """
    components = {}
    try:
        # Set up optimizer
        optimizer = setup_optimizer(
            model=models.get("unet", None),  # Primary model for training
            optimizer_type=args.optimizer.optimizer_type if hasattr(args.optimizer, "optimizer_type") else "adamw",
            learning_rate=args.optimizer.learning_rate if hasattr(args.optimizer, "learning_rate") else 1e-5,
            weight_decay=args.optimizer.weight_decay if hasattr(args.optimizer, "weight_decay") else 1e-2,
            adam_beta1=args.optimizer.adam_beta1 if hasattr(args.optimizer, "adam_beta1") else 0.9,
            adam_beta2=args.optimizer.adam_beta2 if hasattr(args.optimizer, "adam_beta2") else 0.999,
            adam_epsilon=args.optimizer.adam_epsilon if hasattr(args.optimizer, "adam_epsilon") else 1e-8,
            use_8bit_optimizer=args.optimizer.use_8bit_adam if hasattr(args.optimizer, "use_8bit_adam") else False
        )
        components["optimizer"] = optimizer

        # Set up learning rate scheduler
        if args.scheduler.use_scheduler:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.scheduler.num_warmup_steps,
                num_training_steps=args.scheduler.num_training_steps,
                num_cycles=args.scheduler.num_cycles if hasattr(args.scheduler, "num_cycles") else 1
            )
            components["scheduler"] = scheduler

        # Set up AMP scaler
        if args.mixed_precision != "no":
            components["scaler"] = torch.cuda.amp.GradScaler()

        # Set up EMA
        if args.use_ema:
            components["ema_model"] = setup_ema_model(
                model=models.get("unet", None),
                model_path=args.model_path if hasattr(args, "model_path") else None,
                device=args.device if hasattr(args, "device") else "cuda",
                decay=args.ema.decay if hasattr(args.ema, "decay") else 0.9999,
                update_after_step=args.ema.update_after_step if hasattr(args.ema, "update_after_step") else 100,
                mixed_precision=args.mixed_precision if hasattr(args, "mixed_precision") else "no"
            )

        # Set up tag weighter
        components["tag_weighter"] = setup_tag_weighter(args)

        # Set up VAE finetuner
        if hasattr(args, "vae_args") and hasattr(args.vae_args, "enable_vae_finetuning") and args.vae_args.enable_vae_finetuning:
            components["vae_finetuner"] = setup_vae_finetuner(args.vae_args, models)

        # Validate components
        _validate_components(components)
        logger.info("Training components initialized successfully")
        return components

    except Exception as e:
        logger.error("Failed to initialize training components: %s", str(e))
        _cleanup_failed_initialization(components)
        raise
