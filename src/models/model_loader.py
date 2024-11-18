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
from safetensors.torch import save_file
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
            pretrained_model_path,
            subfolder="vae",
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


def create_vae_model(
    vae_path: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    use_safetensors: bool = True,
    force_upcast: bool = True,
) -> AutoencoderKL:
    """Create VAE model from pretrained weights.
    
    Args:
        vae_path: Path to pretrained VAE weights
        dtype: Data type for model parameters
        device: Device to load model on
        use_safetensors: Whether to use safetensors format
        force_upcast: Whether to force upcast VAE to float32
        
    Returns:
        Initialized AutoencoderKL model
        
    Raises:
        ValueError: If vae_path is invalid
        RuntimeError: If model loading fails
    """
    try:
        if not vae_path:
            raise ValueError("VAE path cannot be empty")
            
        # Set default device if none provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load VAE with configuration
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
            force_upcast=force_upcast
        )
        
        # Move to device and set dtype
        if device:
            vae = vae.to(device=device, dtype=dtype)
            
        # Enable memory efficient attention if available
        if hasattr(vae, "enable_xformers_memory_efficient_attention"):
            vae.enable_xformers_memory_efficient_attention()
            
        logger.info(f"Successfully loaded VAE from {vae_path}")
        return vae
        
    except Exception as e:
        logger.error(f"Failed to load VAE from {vae_path}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"VAE loading failed: {str(e)}")

def save_diffusers_format(
    pipeline: StableDiffusionXLPipeline,
    output_dir: str,
    save_vae: bool = True,
    use_safetensors: bool = True,
) -> None:
    """Save pipeline in diffusers format.
    
    Args:
        pipeline: StableDiffusionXLPipeline instance
        output_dir: Directory to save models to
        save_vae: Whether to save VAE model
        use_safetensors: Whether to use safetensors format
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        components = {
            "unet": pipeline.unet,
            "text_encoder": pipeline.text_encoder,
            "text_encoder_2": pipeline.text_encoder_2,
            "tokenizer": pipeline.tokenizer,
            "tokenizer_2": pipeline.tokenizer_2,
            "scheduler": pipeline.scheduler,
        }
        
        if save_vae:
            components["vae"] = pipeline.vae
            
        # Save each component
        for name, component in components.items():
            save_path = os.path.join(output_dir, name)
            os.makedirs(save_path, exist_ok=True)
            
            if hasattr(component, "save_pretrained"):
                component.save_pretrained(
                    save_path,
                    safe_serialization=use_safetensors
                )
                
        # Save pipeline config
        pipeline.save_config(output_dir)
        
        logger.info(f"Saved pipeline to {output_dir} in diffusers format")
        
    except Exception as e:
        logger.error(f"Failed to save pipeline: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Pipeline saving failed: {str(e)}")

def save_checkpoint(
    pipeline: StableDiffusionXLPipeline,
    checkpoint_path: str,
    save_vae: bool = True,
    use_safetensors: bool = True,
) -> None:
    """Save pipeline as a single checkpoint file.
    
    Args:
        pipeline: StableDiffusionXLPipeline instance
        checkpoint_path: Path to save checkpoint to
        save_vae: Whether to save VAE model
        use_safetensors: Whether to use safetensors format
    """
    try:
        # Create state dict
        state_dict = {
            "unet": pipeline.unet.state_dict(),
            "text_encoder": pipeline.text_encoder.state_dict(),
            "text_encoder_2": pipeline.text_encoder_2.state_dict(),
        }
        
        if save_vae:
            state_dict["vae"] = pipeline.vae.state_dict()
            
        # Save checkpoint
        if use_safetensors:
            save_file(state_dict, checkpoint_path)
        else:
            torch.save(state_dict, checkpoint_path)
            
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Checkpoint saving failed: {str(e)}")