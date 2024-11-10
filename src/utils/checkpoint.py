import logging
import torch
import os
import traceback
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel




logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_dir, models, train_components):
    """
    Load a saved checkpoint in diffusers format with safetensors support
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoint
        models (dict): Dictionary containing models to load
        train_components (dict): Dictionary containing training components
            
    Returns:
        dict: Training state (if available) or None
    """
    try:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Verify directory structure
        is_valid, optional_status = verify_checkpoint_directory(checkpoint_dir)
        if not is_valid:
            raise ValueError("Invalid checkpoint directory structure")
        
        # Load models (required)
        model_components = {
            "unet": ("unet", UNet2DConditionModel),
            "vae": ("vae", AutoencoderKL),
            "text_encoder": ("text_encoder", CLIPTextModel),
            "text_encoder_2": ("text_encoder_2", CLIPTextModel)
        }
        
        for model_key, (subfolder, model_class) in model_components.items():
            model_dir = os.path.join(checkpoint_dir, subfolder)
            logger.info(f"Loading {model_key} from {model_dir}")
            models[model_key] = model_class.from_pretrained(
                model_dir,
                use_safetensors=True,
                torch_dtype=models[model_key].dtype
            )
        
        # Load training state if available
        training_state = None
        if optional_status["training_state.pt"]:
            logger.info("Loading training state")
            training_state = torch.load(
                os.path.join(checkpoint_dir, "training_state.pt"),
                map_location='cpu'
            )
            
            # Load optimizer state if available
            if optional_status["optimizer.pt"]:
                logger.info("Loading optimizer state")
                train_components["optimizer"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "optimizer.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load scheduler state if available
            if optional_status["scheduler.pt"]:
                logger.info("Loading scheduler state")
                train_components["lr_scheduler"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "scheduler.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load EMA state if available
            if "ema_model" in train_components:
                if optional_status["ema.safetensors"]:
                    logger.info("Loading EMA state from safetensors")
                    from safetensors.torch import load_file
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

        
def verify_models(models):
    """
    Verify that all required models are present and properly configured
    
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
            
        # Verify model states
        assert not models["text_encoder"].training, "Text encoder should be in eval mode"
        assert not models["text_encoder_2"].training, "Text encoder 2 should be in eval mode"
        assert not models["vae"].training, "VAE should be in eval mode"
        
        # Verify gradient states
        assert not models["text_encoder"].requires_grad, "Text encoder should not require gradients"
        assert not models["text_encoder_2"].requires_grad, "Text encoder 2 should not require gradients"
        assert not models["vae"].requires_grad, "VAE should not require gradients"
        
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def verify_checkpoint_directory(checkpoint_dir):
    """
    Verify checkpoint directory structure and return status of optional files
    
    Args:
        checkpoint_dir: Directory to verify
        
    Returns:
        tuple: (is_valid, optional_files_status)
    """
    if not os.path.exists(checkpoint_dir):
        return False, {}
        
    optional_files = {
        "training_state.pt": False,
        "optimizer.pt": False,
        "scheduler.pt": False,
        "ema.safetensors": False,
        "ema.pt": False
    }
    
    for filename in optional_files:
        optional_files[filename] = os.path.exists(os.path.join(checkpoint_dir, filename))
        
    return True, optional_files
