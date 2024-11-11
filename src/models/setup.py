import torch
import logging
import traceback
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
from torch.optim.swa_utils import AveragedModel
from models.model_validator import ModelValidator
from models.reward_model import (
    AttributeBindingRewardModel,
    SpatialRewardModel,
    NonSpatialRewardModel,
    BaseRewardModel
)

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
    try:
        # Validate device and dtype compatibility
        if dtype == torch.float16 and not device.type == 'cuda':
            raise ValueError("float16 precision requires CUDA device")
            
        # Clear CUDA cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info(f"GPU memory before model setup: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        models = {}
        
        # Setup core models with validation
        models.update(_setup_core_models(args, device, dtype))
        
        # Setup EMA if enabled
        if args.use_ema:
            models["ema_model"] = _setup_ema_model(models["unet"], args)
            
        # Setup reward models if using IterComp
        if args.use_itercomp:
            models["reward_models"] = _setup_reward_models(device, dtype)
            
        # Verify model configuration
        if not verify_models(models):
            raise ValueError("Model verification failed")
            
        logger.info("Model setup completed successfully")
        return models
        
    except Exception as e:
        logger.error("Critical error during model setup")
        logger.error(traceback.format_exc())
        raise

def _setup_core_models(args, device, dtype):
    """Setup and validate core model components"""
    models = {}
    
    try:
        # Setup UNet with memory optimization
        logger.info("Setting up UNet...")
        models["unet"] = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=dtype
        ).to(device)
        
        if args.gradient_checkpointing:
            models["unet"].enable_gradient_checkpointing()
            
        if args.enable_compile:
            logger.info(f"Compiling UNet with mode: {args.compile_mode}")
            models["unet"] = torch.compile(
                models["unet"],
                mode=args.compile_mode,
                fullgraph=True
            )
            
        # Setup VAE with automatic memory management
        logger.info("Setting up VAE...")
        with torch.cuda.amp.autocast(dtype=dtype):
            models["vae"] = AutoencoderKL.from_pretrained(
                args.model_path,
                subfolder="vae",
                torch_dtype=dtype
            ).to(device)
            models["vae"].requires_grad_(False)
            models["vae"].eval()
            
        # Setup text encoders with shared memory allocation
        logger.info("Setting up text encoders...")
        models.update(_setup_text_encoders(args, device, dtype))
        
        return models
        
    except Exception as e:
        logger.error(f"Error in core model setup: {str(e)}")
        raise

def _setup_text_encoders(args, device, dtype):
    """Setup text encoders and tokenizers with shared memory"""
    try:
        models = {}
        
        # Load tokenizers
        models["tokenizer"] = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer"
        )
        models["tokenizer_2"] = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer_2"
        )
        
        # Load encoders with memory optimization
        with torch.cuda.amp.autocast(dtype=dtype):
            models["text_encoder"] = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder",
                torch_dtype=dtype
            ).to(device)
            models["text_encoder_2"] = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder_2",
                torch_dtype=dtype
            ).to(device)
            
        # Freeze encoders
        for encoder in ["text_encoder", "text_encoder_2"]:
            models[encoder].requires_grad_(False)
            models[encoder].eval()
            
        return models
        
    except Exception as e:
        logger.error(f"Error in text encoder setup: {str(e)}")
        raise

def _setup_reward_models(device, dtype):
    """Setup composition-aware reward models with validation"""
    try:
        logger.info("Setting up reward models...")
        
        # Initialize reward models for different aspects
        reward_models = {
            "attribute": AttributeBindingRewardModel(
                clip_feature_dim=768
            ),
            "spatial": SpatialRewardModel(
                detr_feature_dim=256
            ),
            "non_spatial": NonSpatialRewardModel(
                clip_dim=768,
                detr_dim=256
            )
        }
        
        # Move models to device and dtype
        for name, model in reward_models.items():
            # First move to device, then cast to dtype
            reward_models[name] = model.to(device)
            if dtype in [torch.float16, torch.bfloat16]:
                reward_models[name] = reward_models[name].to(dtype)
            model.eval()
            model.requires_grad_(False)
            logger.debug(f"Initialized {name} reward model on {device} with dtype {dtype}")
            
        # Validate model configurations
        for name, model in reward_models.items():
            if not isinstance(model, BaseRewardModel):
                raise TypeError(f"Invalid reward model type for {name}")
                
            # Check device
            model_device = next(model.parameters()).device
            if model_device != device:
                raise ValueError(f"Reward model {name} on wrong device: {model_device} vs {device}")
                
            # Check dtype if specified
            if dtype in [torch.float16, torch.bfloat16]:
                model_dtype = next(model.parameters()).dtype
                if model_dtype != dtype:
                    raise ValueError(f"Reward model {name} has wrong dtype: {model_dtype} vs {dtype}")
                
        logger.info("Successfully set up all reward models")
        return reward_models
        
    except Exception as e:
        logger.error("Failed to setup reward models")
        logger.error(traceback.format_exc())
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

def _setup_ema_model(unet, args):
    """
    Setup Exponential Moving Average (EMA) model
    
    Args:
        unet: Base UNet model to create EMA from
        args: Training arguments containing EMA configuration
        
    Returns:
        EMAModel: Configured EMA model wrapper
    """
    try:
        logger.info("Setting up EMA model...")
        
        # Validate EMA decay rate
        if not 0.0 <= args.ema_decay <= 1.0:
            raise ValueError(f"Invalid EMA decay rate: {args.ema_decay}. Must be between 0 and 1")
            
        # Create EMA model
        ema_model = AveragedModel(
            unet,
            avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (
                args.ema_decay * averaged_model_parameter + 
                (1 - args.ema_decay) * model_parameter
            )
        )
        
        # Copy current model weights
        for param in ema_model.parameters():
            param.requires_grad_(False)
            
        logger.info(f"EMA model initialized with decay rate: {args.ema_decay}")
        return ema_model
        
    except Exception as e:
        logger.error("Failed to setup EMA model")
        logger.error(traceback.format_exc())
        raise
