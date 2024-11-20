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
from torch.optim import Lion

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
) -> Dict[str, Any]:
    """Cache tag weighter configuration."""
    return {
        "token_dropout_rate": token_dropout_rate,
        "caption_dropout_rate": caption_dropout_rate,
        "rarity_factor": rarity_factor,
        "emphasis_factor": emphasis_factor,
    }

def setup_tag_weighter(args) -> Optional[CaptionProcessor]:
    """Initialize tag weighting system with CaptionProcessor."""
    try:
        if not getattr(args, "use_tag_weighting", False):
            return None

        # Get cached config
        config = _get_tag_weighter_config(
            token_dropout_rate=args.tag_weighting.token_dropout_rate,
            caption_dropout_rate=args.tag_weighting.caption_dropout_rate,
            rarity_factor=args.tag_weighting.rarity_factor,
            emphasis_factor=args.tag_weighting.emphasis_factor,
        )

        # Initialize processor
        processor = CaptionProcessor(**config)
        logger.info("Tag weighter initialized successfully")
        return processor

    except Exception as e:
        logger.error("Failed to initialize tag weighter: %s", str(e))
        raise

def initialize_training_components(
    config: Any,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize training components with NAI recommendations"""
    components = {}
    
    # Use Lion optimizer with NAI's settings
    components["optimizer"] = Lion(
        models["unet"].parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(0.95, 0.98)  # NAI recommended values
    )
    
    # NAI-style warmup scheduler
    components["scheduler"] = get_cosine_schedule_with_warmup(
        optimizer=components["optimizer"],
        num_warmup_steps=int(config.scheduler.num_training_steps * 0.02),  # 2% warmup
        num_training_steps=config.scheduler.num_training_steps
    )
    
    # Setup gradient scaler for mixed precision training
    if config.mixed_precision != "no":
        components["scaler"] = GradScaler("cuda")
    
    # Setup EMA using custom implementation
    if config.use_ema:
        components["ema_model"] = setup_ema_model(
            model=models["unet"],
            device=models["unet"].device,
            power=0.75,
            max_value=config.ema.decay,
            update_after_step=config.ema.update_after_step,
            inv_gamma=1.0
        )
    
    return components
