"""Model loading utilities for SDXL training."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
import logging
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
from src.config.args import TrainingConfig

logger = logging.getLogger(__name__)

def create_sdxl_models(
    pretrained_model_path: str,
    vae_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], StableDiffusionXLPipeline]:
    """Create SDXL models from pretrained weights.
    
    Args:
        pretrained_model_path: Path to pretrained SDXL model
        vae_path: Optional path to custom VAE model
        dtype: Model dtype to use
        device: Device to load models on
        
    Returns:
        Tuple of (models dict, pipeline)
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load base pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=dtype,
        )
        
        # Extract components
        models = {
            "unet": pipeline.unet,
            "vae": pipeline.vae,
            "text_encoder": pipeline.text_encoder,
            "text_encoder_2": pipeline.text_encoder_2,
            "tokenizer": pipeline.tokenizer,
            "tokenizer_2": pipeline.tokenizer_2,
            "scheduler": pipeline.scheduler,
        }
        
        # Load custom VAE if specified
        if vae_path:
            models["vae"] = create_vae_model(vae_path, dtype)
        
        # Move models to device
        for name, model in models.items():
            if hasattr(model, "to"):
                models[name] = model.to(device)
                
        return models, pipeline
        
    except Exception as e:
        import traceback
        logger.error("SDXL model creation failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


def create_vae_model(
    vae_path: str,
    dtype: torch.dtype = torch.float32,
) -> AutoencoderKL:
    """Create VAE model from pretrained weights.
    
    Args:
        vae_path: Path to pretrained VAE model
        dtype: Model dtype to use
        
    Returns:
        Loaded VAE model
    """
    try:
        return AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype
        )
        
    except Exception as e:
        import traceback
        logger.error("VAE model creation failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


def load_models(config: TrainingConfig) -> Dict[str, Any]:
    """Load all required models for SDXL training."""
    try:
        # Update this line to use pretrained_model_path
        models_dict = create_sdxl_models(
            config.pretrained_model_path,  # Changed from model_path
            device=config.device if hasattr(config, "device") else "cuda",
            use_safetensors=True
        )
        
        # Rest of the function remains the same
        return models_dict
        
    except Exception as e:
        import traceback
        logger.error("Model loading failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


def save_models(models: Dict[str, Any], output_dir: str, save_vae: bool = True) -> None:
    """Save trained models to disk.
    
    Args:
        models: Dictionary containing models to save
        output_dir: Directory to save models to
        save_vae: Whether to save VAE model
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save UNET
        models["unet"].save_pretrained(os.path.join(output_dir, "unet"))
        
        # Save text encoders
        models["text_encoder"].save_pretrained(os.path.join(output_dir, "text_encoder"))
        models["text_encoder_2"].save_pretrained(os.path.join(output_dir, "text_encoder_2"))
        
        # Save tokenizers
        models["tokenizer"].save_pretrained(os.path.join(output_dir, "tokenizer"))
        models["tokenizer_2"].save_pretrained(os.path.join(output_dir, "tokenizer_2"))
        
        # Save VAE if requested
        if save_vae:
            models["vae"].save_pretrained(os.path.join(output_dir, "vae"))
        
        # Save scheduler
        models["scheduler"].save_pretrained(os.path.join(output_dir, "scheduler"))
        
    except Exception as e:
        import traceback
        logger.error("Model saving failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


def load_checkpoint(checkpoint_path: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Load model weights from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        models: Dictionary of models to load weights into
        
    Returns:
        Updated models dictionary with loaded weights
    """
    try:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # Load model weights
        models["unet"].load_state_dict(checkpoint["unet_state_dict"])
        models["text_encoder"].load_state_dict(checkpoint["text_encoder_state_dict"])
        models["text_encoder_2"].load_state_dict(checkpoint["text_encoder_2_state_dict"])
        
        if "vae_state_dict" in checkpoint and models.get("vae") is not None:
            models["vae"].load_state_dict(checkpoint["vae_state_dict"])
        
        return models
        
    except Exception as e:
        import traceback
        logger.error("Checkpoint loading failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise
