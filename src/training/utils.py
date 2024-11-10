import torch
import logging
import os
import shutil
import traceback
from safetensors.torch import load_file
from pathlib import Path
from torch.utils.data import DataLoader

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
            if optional_status["optimizer.pt"]:
                logger.info("Loading optimizer state")
                train_components["optimizer"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "optimizer.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load scheduler state
            if optional_status["scheduler.pt"]:
                logger.info("Loading scheduler state")
                train_components["lr_scheduler"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "scheduler.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load EMA state
            if "ema_model" in train_components:
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