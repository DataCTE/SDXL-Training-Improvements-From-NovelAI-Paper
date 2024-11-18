import torch
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import asdict
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
from src.config.args import VAEConfig

logger = logging.getLogger(__name__)


def train_sdxl(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_data_dir: Optional[Union[str, Path]] = None,
    pretrained_model_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
    validation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SDXLTrainer:
    """
    High-level wrapper for training SDXL models with improvements from NovelAI paper.
    
    Args:
        train_data_dir: Directory containing training data
        output_dir: Directory to save checkpoints and outputs
        val_data_dir: Optional directory containing validation data
        pretrained_model_path: Optional path to pretrained model weights
        resume_from_checkpoint: Optional path to resume training from checkpoint
        models: Optional pre-loaded model dictionary
        validation_config: Optional validation configuration parameters
        **kwargs: Additional training configuration parameters
        
    Returns:
        Trained SDXLTrainer instance
    """
    try:
        # Remove validation_config from kwargs if present
        kwargs.pop('validation_config', None)
        
        # Setup configuration - Update this part
        config = TrainingConfig(
            pretrained_model_path=str(pretrained_model_path) if pretrained_model_path else "",
            train_data_dir=str(train_data_dir),
            output_dir=str(output_dir),
            **{k: v for k, v in kwargs.items() if k in TrainingConfig.__dataclass_fields__}
        )
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided models or create new ones
        if models is None:
            models_dict, _ = create_sdxl_models(pretrained_model_path)
        else:
            models_dict = models
            
        vae_cache = VAECache(
            vae=models_dict['vae'],
            cache_dir=str(output_dir / "vae_cache"),
            max_cache_size=10000,
            num_workers=4,
            batch_size=8
        )
        text_embedding_cache = TextEmbeddingCache(
            text_encoder1=models_dict['text_encoder'],
            text_encoder2=models_dict['text_encoder_2'],
            tokenizer1=models_dict['tokenizer'],
            tokenizer2=models_dict['tokenizer_2'],
            cache_dir=str(output_dir / "text_embeds_cache"),
            max_cache_size=10000,
            num_workers=4,
            batch_size=32
        )
        
        # Get image paths and captions
        train_image_paths = [str(p) for p in Path(train_data_dir).glob("*.[jJ][pP][gG]")]
        train_captions = load_captions(train_image_paths)
        
        # Create bucket manager with proper parameters
        bucket_manager = BucketManager(
            max_resolution=1024 * 1024,  # Default max resolution
            min_batch_size=1,
            max_batch_size=config.batch_size,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4
        )
        
        # Add images to bucket manager
        for image_path in train_image_paths:
            try:
                # Get image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                bucket_manager.add_image(image_path, width, height)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
        
        # Create train dataloader
        train_dataloader = create_train_dataloader(
            image_paths=train_image_paths,
            captions=train_captions,
            bucket_manager=bucket_manager,
            batch_size=config.batch_size,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
            vae_cache=vae_cache,
            text_cache=text_embedding_cache
        )
        
        val_dataloader = None
        if val_data_dir:
            val_image_paths = [str(p) for p in Path(val_data_dir).glob("*.[jJ][pP][gG]")]
            val_captions = load_captions(val_image_paths)
            
            val_dataloader = create_validation_dataloader(
                image_paths=val_image_paths,
                captions=val_captions,
                bucket_manager=bucket_manager,
                batch_size=config.batch_size,
                num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
                vae_cache=vae_cache,
                text_cache=text_embedding_cache
            )
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            trainer = SDXLTrainer.load_checkpoint(
                resume_from_checkpoint,
                train_dataloader,
                val_dataloader
            )
            logger.info(f"Resumed training from {resume_from_checkpoint}")
            return trainer
        
        # Create new training instance
        trainer = SDXLTrainer(
            config=config,
            models=models_dict,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        return trainer
        
    except Exception as e:
        import traceback
        logger.error("SDXL training setup failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise

def train_vae(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[VAEConfig] = None,
    **kwargs
) -> VAEFinetuner:
    """
    High-level wrapper for VAE finetuning with improvements.
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided config or create default
        if config is None:
            config = VAEConfig(**kwargs)
        
        # Create VAE model
        vae = create_vae_model(config.vae_path)
        
        # Setup VAE cache
        vae_cache = VAECache(
            vae=vae,
            cache_dir=str(output_dir / "vae_cache"),
            max_cache_size=config.cache_size,
            num_workers=4,
            batch_size=config.batch_size
        )
        
        # Get and validate image paths
        train_image_paths = []
        for path in Path(train_data_dir).glob("*.[jJ][pP][gG]"):
            if validate_image(str(path)):
                train_image_paths.append(str(path))
            else:
                logger.warning(f"Skipping invalid image: {path}")
        
        # Create bucket manager
        bucket_manager = BucketManager(train_image_paths)
        
        # Create empty captions dict (VAE training doesn't need captions)
        train_captions = {path: "" for path in train_image_paths}
        
        # Create dataloader
        train_dataloader = create_train_dataloader(
            image_paths=train_image_paths,
            captions=train_captions,
            bucket_manager=bucket_manager,
            batch_size=config.batch_size,
            num_workers=4,
            vae_cache=vae_cache,
            text_cache=None,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
        # Initialize trainer
        trainer = VAEFinetuner(
            vae=vae,
            config=config,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        return trainer
        
    except Exception as e:
        import traceback
        logger.error("VAE training setup failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise

def export_model(
    trainer: Union[SDXLTrainer, VAEFinetuner],
    output_dir: Union[str, Path],
    model_format: str = "safetensors",
    half_precision: bool = True
) -> None:
    """
    Export trained model in specified format.
    
    Args:
        trainer: Trained model trainer instance
        output_dir: Directory to save exported model
        model_format: Format to export model in ('safetensors' or 'pytorch')
        half_precision: Whether to export in half precision
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(trainer, SDXLTrainer):
        # Export SDXL model
        for name, model in trainer.models.items():
            if half_precision:
                model = model.half()
            
            if model_format == "safetensors":
                save_path = output_dir / f"{name}.safetensors"
                torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            else:
                save_path = output_dir / f"{name}.pt"
                torch.save(model.state_dict(), save_path)
                
            logger.info(f"Exported {name} to {save_path}")
    
    elif isinstance(trainer, VAEFinetuner):
        # Export VAE model
        if half_precision:
            trainer.vae = trainer.vae.half()
            
        if model_format == "safetensors":
            save_path = output_dir / "vae.safetensors"
            torch.save(trainer.vae.state_dict(), save_path, _use_new_zipfile_serialization=False)
        else:
            save_path = output_dir / "vae.pt"
            torch.save(trainer.vae.state_dict(), save_path)
            
        logger.info(f"Exported VAE to {save_path}")
    
    else:
        raise ValueError(f"Unsupported trainer type: {type(trainer)}")