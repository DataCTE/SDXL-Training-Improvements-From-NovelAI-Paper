"""Model loading utilities for SDXL training."""

import os
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from src.models.SDXL.pipeline import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from src.config.args import TrainingConfig
from diffusers import EulerDiscreteScheduler
import logging
logger = logging.getLogger(__name__)

def create_sdxl_models(
    pretrained_model_path: str,
    vae_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], StableDiffusionXLPipeline]:
    """Create SDXL models from pretrained weights."""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load individual components first
        models = {}
        
        # Load VAE
        vae_path = vae_path or os.path.join(pretrained_model_path, "vae")
        models["vae"] = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype
        )
        
        # Load text encoders and tokenizers
        models["text_encoder"] = CLIPTextModel.from_pretrained(
            os.path.join(pretrained_model_path, "text_encoder"),
            torch_dtype=dtype
        )
        models["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(
            os.path.join(pretrained_model_path, "text_encoder_2"),
            torch_dtype=dtype
        )
        models["tokenizer"] = CLIPTokenizer.from_pretrained(
            os.path.join(pretrained_model_path, "tokenizer")
        )
        models["tokenizer_2"] = CLIPTokenizer.from_pretrained(
            os.path.join(pretrained_model_path, "tokenizer_2")
        )
        
        # Load UNet
        models["unet"] = UNet2DConditionModel.from_pretrained(
            os.path.join(pretrained_model_path, "unet"),
            torch_dtype=dtype
        )
        
        # Load scheduler
        models["scheduler"] = EulerDiscreteScheduler.from_pretrained(
            os.path.join(pretrained_model_path, "scheduler")
        )
        
        # Create pipeline with loaded components
        pipeline = StableDiffusionXLPipeline(
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            unet=models["unet"],
            scheduler=models["scheduler"],
            force_zeros_for_empty_prompt=True
        )
        
        # Move models to device
        for name, model in models.items():
            if hasattr(model, "to"):
                models[name] = model.to(device=device, dtype=dtype)
                
        return models, pipeline
        
    except Exception as e:
        logger.error(f"SDXL model creation failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def load_models(config: TrainingConfig) -> Dict[str, Any]:
    """Load all required models for SDXL training."""
    try:
        device = getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(config, "dtype", torch.float32)
        
        models_dict, _ = create_sdxl_models(
            pretrained_model_path=config.pretrained_model_path,
            vae_path=getattr(config, "vae_path", None),
            dtype=dtype,
            device=device
        )
        
        return models_dict
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def save_models(
    models: Dict[str, Any],
    output_dir: str,
    save_vae: bool = True,
    save_format: str = "diffusers"
) -> None:
    """Save trained models to disk.
    
    Args:
        models: Dictionary containing models to save
        output_dir: Directory to save models to
        save_vae: Whether to save VAE model
        save_format: Format to save models in ('diffusers' or 'safetensors')
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        component_dirs = {
            "unet": "unet",
            "text_encoder": "text_encoder", 
            "text_encoder_2": "text_encoder_2",
            "tokenizer": "tokenizer",
            "tokenizer_2": "tokenizer_2",
            "scheduler": "scheduler"
        }
        
        if save_vae:
            component_dirs["vae"] = "vae"
            
        for name, subdir in component_dirs.items():
            if name in models:
                save_path = os.path.join(output_dir, subdir)
                os.makedirs(save_path, exist_ok=True)
                
                if hasattr(models[name], "save_pretrained"):
                    models[name].save_pretrained(
                        save_path,
                        safe_serialization=(save_format == "safetensors")
                    )
                    
        logger.info(f"Saved models to {output_dir}")
        
    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
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
            
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Map of checkpoint keys to model names
        component_map = {
            "unet_state_dict": "unet",
            "text_encoder_state_dict": "text_encoder",
            "text_encoder_2_state_dict": "text_encoder_2",
            "vae_state_dict": "vae"
        }
        
        # Load state dicts for each component
        for ckpt_key, model_key in component_map.items():
            if ckpt_key in checkpoint and model_key in models:
                models[model_key].load_state_dict(checkpoint[ckpt_key])
                logger.info(f"Loaded {model_key} weights from checkpoint")
                
        return models
        
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise