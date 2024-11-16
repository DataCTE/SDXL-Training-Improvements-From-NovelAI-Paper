"""Checkpoint management utilities for SDXL model training.

This module provides functionality for loading, saving, and managing model checkpoints
during the SDXL training process. It handles both full model checkpoints and intermediate
training state.

Key Features:
- Model loading with custom VAE support
- Safe checkpoint saving with error handling
- Training state management
- Final model output generation
"""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_checkpoint(
    model_path: str,
    vae_path: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load SDXL model checkpoint and initialize training components.
    
    Args:
        model_path (str): Path to the pretrained SDXL model checkpoint
        vae_path (Optional[str], optional): Path to a custom VAE model. Defaults to None.
        dtype (torch.dtype, optional): Model precision type. Defaults to torch.float16.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - models: Dictionary of model components (unet, text_encoders, vae, etc.)
            - train_components: Dictionary of training-related components
            - training_history: Dictionary tracking training metrics and state
            
    Raises:
        Exception: If model loading fails, with detailed error logging
    """
    try:
        logger.info("Loading model from %s", model_path)
        
        # Load the pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
        
        # Extract models
        models = {
            "unet": pipeline.unet,
            "text_encoder": pipeline.text_encoder,
            "text_encoder_2": pipeline.text_encoder_2,
            "tokenizer": pipeline.tokenizer,
            "tokenizer_2": pipeline.tokenizer_2,
            "scheduler": pipeline.scheduler
        }
        
        # Handle VAE loading
        if vae_path:
            logger.info("Using custom VAE from %s", vae_path)
            vae_pipeline = StableDiffusionXLPipeline.from_pretrained(
                vae_path,
                torch_dtype=dtype
            )
            models["vae"] = vae_pipeline.vae
        else:
            models["vae"] = pipeline.vae
        
        # Initialize training components and history
        train_components: Dict[str, Any] = {}
        training_history = {
            'loss_history': [],
            'validation_scores': [],
            'best_score': float('inf'),
            'total_steps': 0
        }
        
        return models, train_components, training_history
        
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise

def save_checkpoint(
    save_path: str,
    models: Dict[str, Any],
    train_components: Dict[str, Any],
    training_history: Dict[str, Any],
    is_final: bool = False
) -> None:
    """Save training checkpoint with all model components and training state.
    
    Args:
        save_path (str): Directory path to save the checkpoint
        models (Dict[str, Any]): Dictionary containing all model components (unet, text_encoders, vae, etc.)
        train_components (Dict[str, Any]): Dictionary of training-related components (optimizer, scheduler, etc.)
        training_history (Dict[str, Any]): Dictionary containing training metrics and state
        is_final (bool, optional): Whether this is the final checkpoint. Defaults to False.
        
    Raises:
        Exception: If checkpoint saving fails, with detailed error logging
        
    Note:
        Saves both the model pipeline in diffusers format and the training state.
        The training state includes optimizer state, scheduler state, and training metrics.
    """
    try:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving %s checkpoint to %s", 
                   'final' if is_final else 'intermediate', 
                   save_path)
        
        # Save models
        pipeline = StableDiffusionXLPipeline(
            vae=models["vae"],
            unet=models["unet"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            scheduler=models["scheduler"]
        )
        pipeline.save_pretrained(save_dir)
        
        # Save training state
        state_path = save_dir / 'training_state.pt'
        torch.save({
            'train_components': train_components,
            'training_history': training_history
        }, state_path)
        
        logger.info("Checkpoint saved successfully")
        
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise

def save_final_outputs(
    output_dir: str,
    models: Dict[str, Any],
    training_history: Dict[str, Any],
    train_components: Dict[str, Any]
) -> None:
    """Save final model outputs including checkpoint and training metrics.
    
    Args:
        output_dir (str): Directory to save final outputs
        models (Dict[str, Any]): Dictionary containing all model components (unet, text_encoders, vae, etc.)
        training_history (Dict[str, Any]): Dictionary containing training metrics and state including:
            - loss_history: List of training losses
            - validation_scores: List of validation metrics
            - best_score: Best validation score achieved
            - total_steps: Total number of training steps
        train_components (Dict[str, Any]): Dictionary of training-related components
        
    Raises:
        Exception: If saving final outputs fails, with detailed error logging
        
    Note:
        Saves:
        - Final model checkpoint in diffusers format
        - Training metrics in JSON format including:
            * Final loss
            * Best validation loss
            * Total training steps
    """
    try:
        # Save final checkpoint
        save_checkpoint(
            output_dir,
            models,
            train_components,
            training_history,
            is_final=True
        )
        
        # Save training metrics
        metrics_path = Path(output_dir) / 'training_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'final_loss': training_history.get('loss_history', [])[-1] if training_history.get('loss_history') else None,
                'best_validation_loss': min(training_history.get('validation_scores', [float('inf')])),
                'total_steps': training_history.get('total_steps', 0)
            }, f, indent=2)
            
        logger.info("Final outputs saved to %s", output_dir)
        
    except Exception as e:
        logger.error("Failed to save final outputs: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise
