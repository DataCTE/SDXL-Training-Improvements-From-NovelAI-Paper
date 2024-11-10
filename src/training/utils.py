import torch
import logging
import os
import shutil
import traceback
from safetensors.torch import load_file
from pathlib import Path
from torch.utils.data import DataLoader
import json

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
    """
    Custom collate function for DataLoader that handles both single and batched samples.
    
    Args:
        batch: List of dictionaries containing dataset items
    Returns:
        Dictionary with properly stacked tensors and lists
    """
    batch_size = len(batch)
    
    if batch_size == 1:
        return batch[0]
    else:
        elem = batch[0]
        collated = {}
        
        for key in elem:
            if key == "tags":
                collated[key] = [d[key] for d in batch]
            elif key == "target_size":
                collated[key] = [d[key] for d in batch]
            else:
                try:
                    collated[key] = torch.stack([d[key] for d in batch])
                except:
                    collated[key] = [d[key] for d in batch]
        
        return collated

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

def save_checkpoint(checkpoint_dir, models, train_components, training_state, use_safetensors=True):
    """
    Save training checkpoint in diffusers format with safetensors support
    
    Args:
        checkpoint_dir: Directory to save the checkpoint
        models: Dictionary containing models to save
        train_components: Dictionary containing training components
        training_state: Dictionary containing training state
        use_safetensors: Whether to use safetensors format for model weights
    """
    try:
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Temporarily put models in eval mode for saving
        training_mode = models["unet"].training
        if training_mode:
            models["unet"].eval()
        
        # Save training state
        if training_state is not None:
            logger.info("Saving training state")
            torch.save(
                training_state,
                os.path.join(checkpoint_dir, "training_state.pt")
            )
        
        # Save optimizer state
        if "optimizer" in train_components:
            logger.info("Saving optimizer state")
            torch.save(
                train_components["optimizer"].state_dict(),
                os.path.join(checkpoint_dir, "optimizer.pt")
            )
        
        # Save scheduler state
        if "lr_scheduler" in train_components:
            logger.info("Saving scheduler state")
            torch.save(
                train_components["lr_scheduler"].state_dict(),
                os.path.join(checkpoint_dir, "scheduler.pt")
            )
        
        # Save EMA state
        if "ema_model" in train_components and train_components["ema_model"] is not None:
            logger.info("Saving EMA state")
            if use_safetensors:
                from safetensors.torch import save_file
                save_file(
                    train_components["ema_model"].state_dict(),
                    os.path.join(checkpoint_dir, "ema.safetensors")
                )
            else:
                torch.save(
                    train_components["ema_model"].state_dict(),
                    os.path.join(checkpoint_dir, "ema.pt")
                )
        
        # Restore UNet training mode if it was training
        if training_mode:
            models["unet"].train()
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error(f"Checkpoint directory: {checkpoint_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_checkpoint(checkpoint_dir, models, train_components):
    """
    Load a saved checkpoint in diffusers format with safetensors support
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        models: Dictionary containing models to load
        train_components: Dictionary containing training components
            
    Returns:
        dict: Training state (if available) or None
    """
    try:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        is_valid, optional_status = verify_checkpoint_directory(checkpoint_dir)
        if not is_valid:
            raise ValueError("Invalid checkpoint directory structure")
        
        # Load training state if available
        training_state = None
        if optional_status["training_state.pt"]:
            logger.info("Loading training state")
            training_state = torch.load(
                os.path.join(checkpoint_dir, "training_state.pt"),
                map_location='cpu'
            )
            
            # Load optimizer state
            if optional_status["optimizer.pt"] and "optimizer" in train_components:
                logger.info("Loading optimizer state")
                train_components["optimizer"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "optimizer.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load scheduler state
            if optional_status["scheduler.pt"] and "lr_scheduler" in train_components:
                logger.info("Loading scheduler state")
                train_components["lr_scheduler"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "scheduler.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load EMA state
            if "ema_model" in train_components and train_components["ema_model"] is not None:
                if optional_status["ema.safetensors"]:
                    logger.info("Loading EMA state from safetensors")
                    ema_state = load_file(
                        os.path.join(checkpoint_dir, "ema.safetensors")
                    )
                    train_components["ema_model"].load_state_dict(ema_state)
                elif optional_status["ema.pt"]:
                    logger.info("Loading EMA state from pytorch")
                    train_components["ema_model"].load_state_dict(
                        torch.load(
                            os.path.join(checkpoint_dir, "ema.pt"),
                            map_location='cpu'
                        )
                    )
        
        return training_state
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        logger.error(f"Checkpoint directory: {checkpoint_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def cleanup(args, wandb_run=None):
    """Cleanup resources after training"""
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Final CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Close wandb run
        if wandb_run is not None:
            wandb_run.finish()
        
        # Remove temporary files
        if hasattr(args, 'cache_dir') and os.path.exists(args.cache_dir):
            logger.info(f"Cleaning up cache directory: {args.cache_dir}")
            shutil.rmtree(args.cache_dir, ignore_errors=True)
            
    except Exception as cleanup_error:
        logger.error(f"Error during cleanup: {cleanup_error}")

def save_final_outputs(args, models, training_history, train_components):
    """Save all final training outputs"""
    logger.info(f"Saving final outputs to {args.output_dir}")
    final_output_dir = os.path.join(args.output_dir, "final")
    
    try:
        logger.info("Saving models in diffusers format")
        
        # Save using diffusers built-in methods
        models["unet"].save_pretrained(os.path.join(final_output_dir, "unet"))
        models["vae"].save_pretrained(os.path.join(final_output_dir, "vae"))
        models["text_encoder"].save_pretrained(os.path.join(final_output_dir, "text_encoder"))
        models["text_encoder_2"].save_pretrained(os.path.join(final_output_dir, "text_encoder_2"))
        models["tokenizer"].save_pretrained(os.path.join(final_output_dir, "tokenizer"))
        models["tokenizer_2"].save_pretrained(os.path.join(final_output_dir, "tokenizer_2"))
        
        # Save EMA if available
        if train_components.get("ema_model") is not None:
            logger.info("Saving separate EMA weights")
            train_components["ema_model"].save_pretrained(os.path.join(final_output_dir, "ema"))
        
        # Save training history and metrics
        logger.info("Saving training history")
        history_path = os.path.join(final_output_dir, "training_history.pt")
        torch.save(training_history, history_path)
        
        # Save training configuration
        logger.info("Saving training configuration")
        config_path = os.path.join(final_output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info("Successfully saved all final outputs")
        
    except Exception as e:
        logger.error(f"Failed to save final outputs: {str(e)}")
        logger.error(f"Output directory: {final_output_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
