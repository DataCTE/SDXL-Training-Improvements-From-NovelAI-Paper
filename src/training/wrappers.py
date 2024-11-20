import torch
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import asdict, dataclass, fields
from PIL import Image

from src.config.args import TrainingConfig, VAEConfig
from src.training.trainer import SDXLTrainer
from src.training.vae_finetuner import VAEFinetuner
from src.data.multiaspect.dataset import create_train_dataloader, create_validation_dataloader
from src.models.model_loader import create_sdxl_models, create_vae_model
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.multiaspect.bucket_manager import BucketManager
from src.data.image_processing.validation import validate_image
from src.data.prompt.caption_processor import load_captions
from src.models.model_loader import save_checkpoint, load_checkpoint, save_diffusers_format
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
import traceback

logger = logging.getLogger(__name__)




def train_sdxl(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_data_dir: Optional[Union[str, Path]] = None,
    pretrained_model_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> SDXLTrainer:
    """High-level wrapper for training SDXL models."""
    try:
        # Setup configuration
        if config is None:
            config = TrainingConfig(
                pretrained_model_path=str(pretrained_model_path) if pretrained_model_path else "",
                train_data_dir=str(train_data_dir),
                output_dir=str(output_dir),
                **kwargs
            )
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create models
        if models is None:
            logger.info("Creating new SDXL models...")
            models_dict, _ = create_sdxl_models(pretrained_model_path)
        else:
            logger.info("Using provided models...")
            models_dict = models
            
        # Setup caches with proper config and absolute paths
        vae_cache_dir = output_dir / "vae_cache"
        text_cache_dir = output_dir / "text_embeds_cache"
        
        vae_cache = VAECache(
            vae=models_dict['vae'],
            cache_dir=str(vae_cache_dir.absolute()),
            max_cache_size=config.caching.vae_cache_size,
            num_workers=config.caching.vae_cache_num_workers,
            batch_size=config.caching.vae_cache_batch_size,
            max_memory_gb=config.caching.vae_cache_memory_gb
        )
        
        text_embedding_cache = TextEmbeddingCache(
            text_encoder1=models_dict['text_encoder'],
            text_encoder2=models_dict['text_encoder_2'],
            tokenizer1=models_dict['tokenizer'],
            tokenizer2=models_dict['tokenizer_2'],
            cache_dir=str(text_cache_dir.absolute()),
            max_cache_size=config.caching.text_cache_size,
            num_workers=config.caching.text_cache_num_workers,
            batch_size=config.caching.text_cache_batch_size,
            max_memory_gb=config.caching.text_cache_memory_gb
        )
        
        # Get image paths and captions
        logger.info("Loading and validating training data...")
        train_image_paths = []
        for path in Path(train_data_dir).glob("*.[jJ][pP][gG]"):
            if validate_image(str(path)):
                train_image_paths.append(str(path))
            else:
                logger.warning(f"Skipping invalid image: {path}")
        
        if not train_image_paths:
            raise ValueError(f"No valid training images found in {train_data_dir}")
            
        train_captions = load_captions(train_image_paths)
        
        # Create train dataloader
        logger.info("Creating training dataloader...")
        train_dataloader = create_train_dataloader(
            image_paths=train_image_paths,
            captions=train_captions,
            config=config,
            vae_cache=vae_cache,
            text_cache=text_embedding_cache
        )
        
        # Create validation dataloader if validation data provided
        val_dataloader = None
        if val_data_dir:
            logger.info("Setting up validation...")
            val_image_paths = []
            for path in Path(val_data_dir).glob("*.[jJ][pP][gG]"):
                if validate_image(str(path)):
                    val_image_paths.append(str(path))
                else:
                    logger.warning(f"Skipping invalid validation image: {path}")
                    
            if val_image_paths:
                val_captions = load_captions(val_image_paths)
                val_dataloader = create_validation_dataloader(
                    image_paths=val_image_paths,
                    captions=val_captions,
                    config=config,
                    vae_cache=vae_cache,
                    text_cache=text_embedding_cache
                )
            else:
                logger.warning(f"No valid validation images found in {val_data_dir}")
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            pipeline = StableDiffusionXLPipeline(
                vae=models_dict["vae"],
                text_encoder=models_dict["text_encoder"],
                text_encoder_2=models_dict["text_encoder_2"],
                tokenizer=models_dict["tokenizer"],
                tokenizer_2=models_dict["tokenizer_2"],
                unet=models_dict["unet"],
                scheduler=models_dict["scheduler"]
            )
            
            load_checkpoint(resume_from_checkpoint, pipeline)
            
            models_dict.update({
                "vae": pipeline.vae,
                "text_encoder": pipeline.text_encoder,
                "text_encoder_2": pipeline.text_encoder_2,
                "unet": pipeline.unet
            })
        
        # Create trainer
        logger.info("Initializing SDXL trainer...")
        trainer = SDXLTrainer(
            config=config,
            models=models_dict,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        return trainer
        
    except Exception as e:
        logger.error("SDXL training setup failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise

def train_vae(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: VAEConfig,
    **kwargs
) -> VAEFinetuner:
    """High-level wrapper for VAE finetuning."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing VAE finetuning...")
        
        # Create VAE model
        vae = create_vae_model(
            vae_path=config.vae_path,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Setup VAE cache with proper config
        vae_cache = VAECache(
            vae=vae,
            cache_dir=str(output_dir / "vae_cache"),
            max_cache_size=config.cache_size,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
            batch_size=config.batch_size
        )
        
        # Get and validate image paths
        logger.info("Loading and validating training data...")
        train_image_paths = []
        for path in Path(train_data_dir).glob("*.[jJ][pP][gG]"):
            if validate_image(str(path)):
                train_image_paths.append(str(path))
            else:
                logger.warning(f"Skipping invalid image: {path}")
        
        if not train_image_paths:
            raise ValueError(f"No valid training images found in {train_data_dir}")
        
        # Create empty captions dict (VAE training doesn't need captions)
        train_captions = {path: "" for path in train_image_paths}
        
        # Create dataloader
        logger.info("Creating training dataloader...")
        train_dataloader = create_train_dataloader(
            image_paths=train_image_paths,
            captions=train_captions,
            config=config,
            vae_cache=vae_cache,
            text_cache=None
        )
        
        # Create trainer
        logger.info("Initializing VAE trainer...")
        trainer = VAEFinetuner(
            vae=vae,
            config=config,
            train_dataloader=train_dataloader,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        return trainer
        
    except Exception as e:
        logger.error("VAE training setup failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise

