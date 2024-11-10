import torch
import logging
import traceback
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
from torch.optim.swa_utils import AveragedModel
from models.model_validator import ModelValidator

logger = logging.getLogger(__name__)

def setup_models(args, device, dtype):
    """
    Initialize and configure all models
    
    Args:
        args: Training arguments
        device: Target device
        dtype: Model precision
        
    Returns:
        dict: Dictionary containing all model components
    """
    logger.info("Setting up models...")
    
    try:
        # Load UNet
        logger.info("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=dtype
        ).to(device)
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        # Compile model if requested
        if args.enable_compile:
            logger.info(f"Compiling UNet with mode: {args.compile_mode}")
            unet = torch.compile(
                unet,
                mode=args.compile_mode,
                fullgraph=True
            )
        
        # Load VAE
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=dtype
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()
        
        # Load text encoders and tokenizers
        logger.info("Loading text encoders and tokenizers...")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
            torch_dtype=dtype
        ).to(device)
        text_encoder_2 = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype
        ).to(device)
        
        # Freeze text encoders
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder.eval()
        text_encoder_2.eval()
        
        # Initialize EMA model if enabled
        if args.use_ema:
            logger.info("Initializing EMA model...")
            ema_model = AveragedModel(
                unet,
                avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (
                    args.ema_decay * averaged_model_parameter + 
                    (1 - args.ema_decay) * model_parameter
                )
            )
        else:
            ema_model = None
        
        # Initialize model validator
        logger.info("Initializing model validator...")
        validator = ModelValidator(
            unet,
            vae,
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            device=device
        )
        
        # Return all components
        models = {
            "unet": unet,
            "vae": vae,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "ema_model": ema_model,
            "validator": validator
        }
        
        logger.info("Model setup completed successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error during model setup: {str(e)}")
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
