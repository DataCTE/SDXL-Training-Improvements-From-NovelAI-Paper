import torch
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from src.data.prompt.caption_processor import CaptionProcessor
from src.training.optimizers.setup_optimizers import setup_optimizer
from src.training.ema import setup_ema_model
from src.training.loss_functions import get_cosine_schedule_with_warmup
import warnings
from torch.cuda.amp import GradScaler

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message=".*torch.cuda.amp.GradScaler.*")

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
    config: Any,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize all training components."""
    components = {}
    
    # Setup optimizer
    logger.info(f"Setting up {config.optimizer.optimizer_type} optimizer with learning rate {config.optimizer.learning_rate}")
    components["optimizer"] = setup_optimizer(
        model=models["unet"],
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        optimizer_type=config.optimizer.optimizer_type,
        use_8bit_optimizer=config.optimizer.use_8bit_adam,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_epsilon=config.optimizer.adam_epsilon
    )
    
    # Setup scheduler using get_cosine_schedule_with_warmup instead of setup_scheduler
    if config.scheduler.use_scheduler:
        components["scheduler"] = get_cosine_schedule_with_warmup(
            optimizer=components["optimizer"],
            num_warmup_steps=config.scheduler.num_warmup_steps,
            num_training_steps=config.scheduler.num_training_steps,
            num_cycles=config.scheduler.num_cycles,
            device=config.device if hasattr(config, "device") else "cuda"
        )
    
    # Setup gradient scaler for mixed precision training
    if config.mixed_precision != "no":
        components["scaler"] = GradScaler("cuda")
    
    # Setup EMA using custom implementation
    if config.use_ema:
        components["ema_model"] = setup_ema_model(
            model=models["unet"],
            power=0.75,
            max_value=config.ema.decay,
            update_after_step=config.ema.update_after_step,
            inv_gamma=1.0,
            device=config.device if hasattr(config, "device") else "cuda",
            jit_compile=True,
            use_cuda_graph=True
        )
    
    return components
