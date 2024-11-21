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
from src.training.optimizers.lion.__init__ import Lion
import math

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
    
    # Calculate resolution-aware sigma max based on image size
    height, width = config.image_size
    area = height * width
    base_res = 1024.0 * 1024.0  # SDXL base resolution
    sigma_scale = math.sqrt(area / base_res)
    sigma_max = 20000.0 * sigma_scale  # Scale sigma_max based on resolution
    
    # Setup optimizer based on type
    if config.optimizer.optimizer_type == "lion":
        components["optimizer"] = Lion(
            models["unet"].parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.lion_betas
        )
    else:
        components["optimizer"] = setup_optimizer(
            model=models["unet"],
            optimizer_type=config.optimizer.optimizer_type,
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            adam_beta1=config.optimizer.adam_beta1,
            adam_beta2=config.optimizer.adam_beta2,
            adam_epsilon=config.optimizer.adam_epsilon,
            use_8bit_optimizer=config.optimizer.use_8bit_adam
        )
    
    # Get ZTSNR schedule
    get_cosine_schedule_with_warmup(
        num_training_steps=config.scheduler.num_training_steps,
        num_warmup_steps=int(config.scheduler.num_training_steps * 0.02),  # 2% warmup
        height=height,
        width=width,
        sigma_min=0.0292,
        sigma_max=sigma_max,  # Use resolution-scaled sigma_max
        rho=7.0,
        device=models["unet"].device
    )
    
    # Setup scheduler with ZTSNR sigmas
    components["scheduler"] = torch.optim.lr_scheduler.LambdaLR(
        optimizer=components["optimizer"],
        lr_lambda=lambda step: max(0.0, 1.0 - step / config.scheduler.num_training_steps)
    )
    
    # Setup gradient scaler for mixed precision training
    if config.mixed_precision != "no":
        components["scaler"] = GradScaler()
    
    # Setup EMA
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
