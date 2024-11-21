"""Ultra-optimized training wrappers implementing NAI's SDXL improvements."""

import torch
import logging
from typing import Optional, Union
from pathlib import Path
from src.config.args import TrainingConfig, VAEConfig
from src.training.trainer import SDXLTrainer
from src.training.vae_finetuner import VAEFinetuner
from src.data.multiaspect.dataset import create_train_dataloader, create_validation_dataloader
from src.models.model_loader import create_sdxl_models, create_vae_model
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.image_processing.validation import validate_image
from src.data.prompt.caption_processor import load_captions
from src.data.multiaspect.dataset import MultiAspectDataset
from PIL import Image
from src.data.multiaspect.bucket_manager import BucketManager


logger = logging.getLogger(__name__)

def train_sdxl(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: TrainingConfig,
    val_data_dir: Optional[Union[str, Path]] = None,
    device: torch.device = torch.device("cuda"),
) -> SDXLTrainer:
    """
    High-level wrapper for training SDXL with NAI improvements.
    Args:
        train_data_dir: Training data directory
        output_dir: Output directory for checkpoints and logs
        config: Training configuration
        val_data_dir: Optional validation data directory
        device: Compute device
    Returns:
        Configured SDXL trainer
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models with NAI settings
    logger.info("Creating SDXL models...")
    models_dict, _ = create_sdxl_models(
        pretrained_model_path=config.pretrained_model_path,
        device=device,
        dtype=torch.float16,  # NAI: trained in float32 with tf32
        vae_path=config.vae_path if hasattr(config, 'vae_path') else None
    )
    
    # Setup caches
    vae_cache = VAECache(
        vae=models_dict['vae'],
        cache_dir=str(output_dir / "vae_cache"),
        max_cache_size=config.caching.vae_cache_size,
        num_workers=config.caching.vae_cache_num_workers,
        batch_size=config.caching.vae_cache_batch_size,
        max_memory_gb=config.caching.vae_cache_memory_gb
    )
    
    text_embedding_cache = TextEmbeddingCache(
        text_encoder_1=models_dict['text_encoder'],
        text_encoder_2=models_dict['text_encoder_2'],
        tokenizer_1=models_dict['tokenizer'],
        tokenizer_2=models_dict['tokenizer_2'],
        cache_dir=str(output_dir / "text_embeds_cache"),
        max_cache_size=config.caching.text_cache_size,
        max_memory_gb=config.caching.text_cache_memory_gb
    )
    
    # Load and validate training data
    logger.info("Loading training data...")
    train_image_paths = [
        str(path) for path in Path(train_data_dir).glob("*.[jJ][pP][gG]")
        if validate_image(str(path))
    ]
    
    if not train_image_paths:
        raise ValueError(f"No valid training images found in {train_data_dir}")
    
    train_captions = load_captions(train_image_paths)
    
    # Create train dataset
    logger.info("Creating training dataset...")
    train_dataset = MultiAspectDataset(
        image_paths=train_image_paths,
        captions=train_captions,
        bucket_manager=None,  # Will be created internally
        vae_cache=vae_cache,
        text_cache=text_embedding_cache,
        num_workers=config.num_workers,
        token_dropout=config.tag_weighting.token_dropout_rate,
        caption_dropout=config.tag_weighting.caption_dropout_rate,
        rarity_factor=config.tag_weighting.rarity_factor,
        emphasis_factor=config.tag_weighting.emphasis_factor
    )
    
    # Create train dataloader
    logger.info("Creating training dataloader...")
    train_dataloader = create_train_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation dataloader if provided
    val_dataloader = None
    if val_data_dir:
        logger.info("Setting up validation...")
        val_image_paths = [
            str(path) for path in Path(val_data_dir).glob("*.[jJ][pP][gG]")
            if validate_image(str(path))
        ]
        
        if val_image_paths:
            val_captions = load_captions(val_image_paths)
            
            # Create bucket manager for validation
            val_bucket_manager = BucketManager(
                max_resolution=config.max_resolution,
                min_batch_size=1,
                max_batch_size=config.batch_size,
                num_workers=config.num_workers
            )
            
            # Add validation images to bucket manager
            for path in val_image_paths:
                with Image.open(path) as img:
                    width, height = img.size
                val_bucket_manager.add_image(path, width, height)
            
            # Create validation dataloader with bucket manager
            val_dataloader = create_validation_dataloader(
                image_paths=val_image_paths,
                captions=val_captions,
                config=config,
                bucket_manager=val_bucket_manager,
                vae_cache=vae_cache,
                text_cache=text_embedding_cache
            )
    
    # Create trainer with NAI improvements
    logger.info("Initializing SDXL trainer...")
    trainer = SDXLTrainer(
        config=config,
        models=models_dict,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device
    )
    
    return trainer

def train_vae(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: VAEConfig,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    tokenizer_1: any,
    tokenizer_2: any,
    device: torch.device = torch.device("cuda"),
) -> VAEFinetuner:
    """
    High-level wrapper for VAE finetuning with NAI improvements.
    Args:
        train_data_dir: Training data directory
        output_dir: Output directory
        config: VAE training configuration
        text_encoder_1: First CLIP text encoder
        text_encoder_2: Second CLIP text encoder
        tokenizer_1: First CLIP tokenizer
        tokenizer_2: Second CLIP tokenizer
        device: Compute device
    Returns:
        Configured VAE trainer
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create VAE with proper settings
    logger.info("Initializing VAE...")
    vae = create_vae_model(
        vae_path=config.vae_path,
        device=device,
        dtype=torch.float32,  # NAI: trained in float32 with tf32
        force_upcast=True
    )
    
    # Setup VAE cache
    vae_cache = VAECache(
        vae=vae,
        cache_dir=str(output_dir / "vae_cache"),
        max_cache_size=config.cache_size,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        max_memory_gb=config.max_memory_gb
    )
    
    # Load and validate training data
    logger.info("Loading training data...")
    train_image_paths = [
        str(path) for path in Path(train_data_dir).glob("*.[jJ][pP][gG]")
        if validate_image(str(path))
    ]
    
    if not train_image_paths:
        raise ValueError(f"No valid training images found in {train_data_dir}")
    
    # Create empty captions (VAE training doesn't need them)
    train_captions = {path: "" for path in train_image_paths}
    
    # Setup text cache
    text_cache = TextEmbeddingCache(
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        cache_dir=str(output_dir / "text_embeds_cache"),
        max_cache_size=config.cache_size,
        max_memory_gb=config.max_memory_gb
    )
    
    # Create dataset
    train_dataset = MultiAspectDataset(
        image_paths=train_image_paths,
        captions=train_captions,
        bucket_manager=None,
        vae_cache=vae_cache,
        text_cache=text_cache,
        num_workers=config.num_workers,
        token_dropout=0.0,  # No dropout for VAE training
        caption_dropout=0.0,
        rarity_factor=0.0,
        emphasis_factor=0.0
    )
    
    # Create dataloader
    train_dataloader = create_train_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    # Create VAE trainer
    logger.info("Initializing VAE trainer...")
    trainer = VAEFinetuner(
        vae=vae,
        config=config,
        train_dataloader=train_dataloader,
        device=device
    )
    
    return trainer