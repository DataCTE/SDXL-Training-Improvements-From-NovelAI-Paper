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

from src.models.SDXL.pipeline import StableDiffusionXLPipeline


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
    return AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=dtype
    )


def load_models(config) -> Dict[str, Any]:
    """Load all required models for SDXL training.
    
    Args:
        config: Training configuration containing model paths and settings
        
    Returns:
        Dict containing all loaded models and components
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
    
    # Load base models
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.model_path,
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
    
    # Load VAE if specified
    if config.vae_args.vae_path:
        models["vae"] = AutoencoderKL.from_pretrained(
            config.vae_args.vae_path,
            torch_dtype=dtype
        )
    
    # Move models to device
    for name, model in models.items():
        if hasattr(model, "to"):
            models[name] = model.to(device)
    
    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        if hasattr(models["unet"], "enable_gradient_checkpointing"):
            models["unet"].enable_gradient_checkpointing()
        if hasattr(models["text_encoder"], "gradient_checkpointing_enable"):
            models["text_encoder"].gradient_checkpointing_enable()
        if hasattr(models["text_encoder_2"], "gradient_checkpointing_enable"):
            models["text_encoder_2"].gradient_checkpointing_enable()
    
    # Enable model compilation if configured
    if config.enable_compile:
        compile_mode = config.compile_mode
        if hasattr(models["unet"], "compile"):
            models["unet"] = torch.compile(models["unet"], mode=compile_mode)
        if hasattr(models["vae"], "compile"):
            models["vae"] = torch.compile(models["vae"], mode=compile_mode)
    
    return models


def save_models(models: Dict[str, Any], output_dir: str, save_vae: bool = True) -> None:
    """Save trained models to disk.
    
    Args:
        models: Dictionary containing models to save
        output_dir: Directory to save models to
        save_vae: Whether to save VAE model
    """
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


def load_checkpoint(checkpoint_path: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Load model weights from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        models: Dictionary of models to load weights into
        
    Returns:
        Updated models dictionary with loaded weights
    """
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