import torch
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import asdict

from src.training.trainer import SDXLTrainer, TrainingConfig
from src.training.vae_finetuner import VAEFineTuner
from src.data.multiaspect.dataset import create_train_dataloader, create_validation_dataloader
from src.models.model_loader import create_sdxl_models, create_vae_model
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache

logger = logging.getLogger(__name__)

def train_sdxl(
    train_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_data_dir: Optional[Union[str, Path]] = None,
    pretrained_model_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
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
        **kwargs: Additional training configuration parameters
        
    Returns:
        Trained SDXLTrainer instance
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup configuration
        config = TrainingConfig(**kwargs)
        
        # Use provided models or create new ones
        if models is None:
            models_dict, _ = create_sdxl_models(pretrained_model_path)
        else:
            models_dict = models
            
        vae_cache = VAECache(
            cache_dir=str(output_dir / "vae_cache"),
            vae=models_dict['vae']
        )
        text_embedding_cache = TextEmbeddingCache(
            cache_dir=str(output_dir / "text_embeds_cache"),
            text_encoder=models_dict['text_encoder']
        )
        
        # Create dataloaders
        train_dataloader = create_train_dataloader(
            train_data_dir,
            vae_cache=vae_cache,
            text_embedding_cache=text_embedding_cache,
            batch_size=config.batch_size,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4
        )
        
        val_dataloader = None
        if val_data_dir:
            val_dataloader = create_validation_dataloader(
                val_data_dir,
                vae_cache=vae_cache,
                text_embedding_cache=text_embedding_cache,
                batch_size=config.batch_size,
                num_workers=config.num_workers if hasattr(config, 'num_workers') else 4
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
    pretrained_vae_path: Optional[str] = None,
    learning_rate: float = 1e-6,
    batch_size: int = 1,
    num_epochs: int = 1,
    mixed_precision: str = "fp16",
    use_8bit_adam: bool = False,
    gradient_checkpointing: bool = False,
    use_channel_scaling: bool = True,
    **kwargs
) -> VAEFineTuner:
    """
    High-level wrapper for VAE finetuning with improvements.
    
    Args:
        train_data_dir: Directory containing training data
        output_dir: Directory to save checkpoints and outputs
        pretrained_vae_path: Optional path to pretrained VAE weights
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        mixed_precision: Mixed precision training mode
        use_8bit_adam: Whether to use 8-bit Adam optimizer
        gradient_checkpointing: Whether to use gradient checkpointing
        use_channel_scaling: Whether to use channel-wise scaling
        **kwargs: Additional VAE training parameters
        
    Returns:
        Trained VAEFineTuner instance
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create VAE model
    vae = create_vae_model(pretrained_vae_path)
    
    # Create dataloader
    load_dataloader = create_train_dataloader(
        train_data_dir,
        vae_cache=None,
        text_embedding_cache=None,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = VAEFineTuner(
        vae=vae,
        learning_rate=learning_rate,
        mixed_precision=mixed_precision,
        use_8bit_adam=use_8bit_adam,
        gradient_checkpointing=gradient_checkpointing,
        use_channel_scaling=use_channel_scaling,
        **kwargs
    )
    
    return trainer

def export_model(
    trainer: Union[SDXLTrainer, VAEFineTuner],
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
    
    elif isinstance(trainer, VAEFineTuner):
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