import torch
import logging
import os
import shutil
import traceback
from safetensors.torch import load_file, save_file
from pathlib import Path
from torch.utils.data import DataLoader
import json
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.schedulers import EulerDiscreteScheduler

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure basic logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def setup_torch_backends():
    """Configure PyTorch backend settings for optimal performance"""
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def custom_collate(batch):
    """Custom collate function that validates dimensions"""
    def validate_latents(latents):
        h, w = latents.shape[-2:]
        min_size = 256 // 8  # Minimum 256 pixels in image space
        max_size = 2048 // 8  # Maximum 2048 pixels in image space
        return (h >= min_size and w >= min_size and 
                h <= max_size and w <= max_size)

    # Filter out invalid samples
    valid_batch = [item for item in batch 
                  if validate_latents(item['latents'])]
    
    if not valid_batch:
        raise ValueError("No valid samples in batch (all below minimum size)")
    
    # Create batch tensors
    batch_dict = {
        "latents": torch.stack([x["latents"] for x in valid_batch]),
        "text_embeddings": torch.stack([x["text_embeddings"] for x in valid_batch]),
        "text_embeddings_2": torch.stack([x["text_embeddings_2"] for x in valid_batch]),
        "pooled_text_embeddings_2": torch.stack([x["pooled_text_embeddings_2"] for x in valid_batch]),
        "tags": [x["tags"] for x in valid_batch]
    }
    
    return batch_dict

def verify_models(models):
    """
    Verify that all models are present and properly configured
    
    Args:
        models (dict): Dictionary of model components
        
    Returns:
        bool: True if verification passes
    """
    required_models = [
        "unet",
        "vae",
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2"
    ]
    
    try:
        # Check for required models
        for model_name in required_models:
            if model_name not in models:
                raise ValueError(f"Missing required model: {model_name}")
        
        # Put models in eval mode before verification
        models["text_encoder"].eval()
        models["text_encoder_2"].eval()
        models["vae"].eval()
        
        # Verify model states
        assert not models["text_encoder"].training, "Text encoder should be in eval mode"
        assert not models["text_encoder_2"].training, "Text encoder 2 should be in eval mode"
        assert not models["vae"].training, "VAE should be in eval mode"
        
        # Verify gradient states
        assert not models["text_encoder"].requires_grad, "Text encoder should not require gradients"
        assert not models["text_encoder_2"].requires_grad, "Text encoder 2 should not require gradients"
        assert not models["vae"].requires_grad, "VAE should not require gradients"
        
        # Put UNet back in train mode if it was in training
        if models["unet"].training:
            models["unet"].train()
        
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def verify_checkpoint_directory(checkpoint_dir):
    """
    Verify checkpoint directory structure and available files
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        tuple: (is_valid, optional_status)
    """
    required_folders = ["unet", "vae", "text_encoder", "text_encoder_2"]
    optional_files = [
        "training_state.pt",
        "optimizer.pt",
        "scheduler.pt",
        "ema.safetensors",
        "ema.pt"
    ]
    
    # Check required folders
    for folder in required_folders:
        if not os.path.isdir(os.path.join(checkpoint_dir, folder)):
            return False, {}
    
    # Check optional files
    optional_status = {
        file: os.path.exists(os.path.join(checkpoint_dir, file))
        for file in optional_files
    }
    
    return True, optional_status

def save_checkpoint(models, train_components, args, epoch, global_step, training_history, output_dir):
    """Save training checkpoint using safetensors format"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save UNet
        unet_path = os.path.join(output_dir, "unet")
        os.makedirs(unet_path, exist_ok=True)
        models["unet"].save_pretrained(unet_path, safe_serialization=True)
        
        # Save EMA model if it exists
        if models.get("ema_model") is not None:
            logger.info("Saving EMA model state...")
            ema_state = models["ema_model"].state_dict()
            
            # Convert model state dict to safetensors format
            ema_tensors = {
                f"ema_model.{key}": tensor
                for key, tensor in ema_state["ema_model"].items()
            }
            
            # Add metadata
            ema_metadata = {
                "num_updates": str(ema_state["num_updates"]),
                "cur_decay_value": str(ema_state["cur_decay_value"])
            }
            
            # Save EMA state using safetensors
            save_file(
                ema_tensors,
                os.path.join(output_dir, "ema.safetensors"),
                metadata=ema_metadata
            )
        
        # Save optimizer and training state
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "training_history": training_history,
            "args": vars(args)
        }
        
        # Save training state as JSON
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Save optimizer state using PyTorch (as it contains non-tensor data)
        torch.save({
            "optimizer": train_components["optimizer"].state_dict(),
            "lr_scheduler": train_components["lr_scheduler"].state_dict(),
        }, os.path.join(output_dir, "optimizer.pt"))
        
        logger.info(f"Checkpoint saved successfully to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

def load_checkpoint(checkpoint_dir, models, train_components):
    """Load checkpoint from safetensors format"""
    try:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load UNet
        unet_path = os.path.join(checkpoint_dir, "unet")
        if os.path.exists(unet_path):
            models["unet"].load_state_dict(
                load_file(os.path.join(unet_path, "diffusion_pytorch_model.safetensors"))
            )
        
        # Load EMA if it exists
        ema_path = os.path.join(checkpoint_dir, "ema.safetensors")
        if os.path.exists(ema_path) and models.get("ema_model") is not None:
            logger.info("Loading EMA model state...")
            
            # Load tensors and metadata
            ema_tensors = load_file(ema_path)
            ema_metadata = load_file(ema_path, metadata_only=True)
            
            # Reconstruct EMA state dict
            ema_state = {
                "ema_model": {
                    key.replace("ema_model.", ""): tensor
                    for key, tensor in ema_tensors.items()
                },
                "num_updates": int(ema_metadata["num_updates"]),
                "cur_decay_value": float(ema_metadata["cur_decay_value"])
            }
            
            models["ema_model"].load_state_dict(ema_state)
        
        # Load training state from JSON
        with open(os.path.join(checkpoint_dir, "training_state.json"), "r") as f:
            training_state = json.load(f)
        
        # Load optimizer state (using PyTorch)
        optimizer_state = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        train_components["optimizer"].load_state_dict(optimizer_state["optimizer"])
        train_components["lr_scheduler"].load_state_dict(optimizer_state["lr_scheduler"])
        
        logger.info("Checkpoint loaded successfully")
        return training_state
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

def cleanup(models, train_components, args):
    """Cleanup after training"""
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Final CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Disable gradient checkpointing for cleanup
        for model_name, model in models.items():
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            elif hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()
        
    except Exception as cleanup_error:
        logger.error(f"Error during cleanup: {cleanup_error}")

def save_final_outputs(args, models, training_history, train_components):
    """Save final model outputs using safetensors"""
    try:
        logger.info("Saving final model outputs...")
        
        # Save final UNet
        final_model_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        models["unet"].save_pretrained(final_model_path, safe_serialization=True)
        
        # Save final EMA model if it exists
        if models.get("ema_model") is not None:
            logger.info("Saving final EMA model...")
            ema_path = os.path.join(args.output_dir, "final_ema")
            os.makedirs(ema_path, exist_ok=True)
            
            ema_model = models["ema_model"].get_model()
            ema_model.save_pretrained(ema_path, safe_serialization=True)
        
        # Save final training metrics
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
            
        logger.info("Final outputs saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving final outputs: {str(e)}")
        raise
