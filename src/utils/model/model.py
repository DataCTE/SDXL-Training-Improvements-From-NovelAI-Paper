import torch
import os
import logging
from diffusers import UNet2DConditionModel, AutoencoderKL
from safetensors.torch import load_file
from src.config.config import Config
import math
import traceback
from typing import Optional, Union
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

def is_xformers_installed():
    """Check if xformers is available."""
    try:
        import xformers
        import xformers.ops
        return True
    except ImportError:
        return False

def create_unet(config: Config, model_dtype: torch.dtype) -> UNet2DConditionModel:
    """Create and configure UNet model.
    
    Args:
        config: Configuration object
        model_dtype: Model data type
        
    Returns:
        UNet2DConditionModel: Configured UNet model
    """
    try:
        # Create UNet with SDXL architecture
        unet = UNet2DConditionModel.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="unet",
            torch_dtype=model_dtype
        )
        
        # Don't apply memory optimizations here - let trainer handle it
        return unet
        
    except Exception as e:
        logger.error(f"Failed to create UNet: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def create_vae(model_dtype: torch.dtype) -> AutoencoderKL:
    """Create and configure VAE model.
    
    Args:
        model_dtype: Model data type
        
    Returns:
        AutoencoderKL: Configured VAE model
    """
    try:
        # Load pretrained VAE directly
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=model_dtype
        )
        
        # Don't set eval/grad state here - let trainer handle it
        return vae
        
    except Exception as e:
        logger.error(f"Failed to create VAE: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def setup_model(
    args,
    device: Optional[torch.device],
    config: Config
) -> tuple[UNet2DConditionModel, AutoencoderKL]:
    """Setup UNet and VAE models with error handling."""
    try:
        logger.info("Starting model setup...")
        logger.info(f"Target device: {device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Determine model dtype from config
        model_dtype = torch.bfloat16 if config.system.mixed_precision == "bf16" else torch.float32
        logger.info(f"Using model dtype: {model_dtype}")
        pretrained_model_name = config.model.pretrained_model_name
        
        logger.info("Loading VAE...")
        if args.train_vae and args.vae_path:
            logger.info(f"Loading VAE from: {args.vae_path}")
            vae = AutoencoderKL.from_pretrained(
                args.vae_path,
                torch_dtype=model_dtype,
                use_safetensors=True
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
                subfolder="vae",
                torch_dtype=model_dtype
            )
        logger.info(f"VAE loaded, device: {next(vae.parameters()).device}")
            
        # Don't set eval/grad state here - let trainer handle it
        
        logger.info("Loading UNet...")
        if args.resume_from_checkpoint:
            logger.info(f"Loading UNet from checkpoint: {args.resume_from_checkpoint}")
            unet = UNet2DConditionModel.from_pretrained(
                args.resume_from_checkpoint,
                subfolder="unet",
                torch_dtype=model_dtype,
                use_safetensors=True
            )
        elif args.unet_path:
            logger.info(f"Loading UNet weights from: {args.unet_path}")
            unet_dir = os.path.dirname(args.unet_path)
            config_path = os.path.join(unet_dir, "config.json")
            
            if os.path.exists(config_path):
                unet = UNet2DConditionModel.from_pretrained(
                    unet_dir,
                    torch_dtype=model_dtype,
                    use_safetensors=True
                )
            else:
                unet = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name,
                    subfolder="unet",
                    torch_dtype=model_dtype
                )
                state_dict = load_file(args.unet_path)
                unet.load_state_dict(state_dict)
        else:
            logger.info("Loading fresh UNet from pretrained model")
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=model_dtype,
                use_safetensors=True
            )
        logger.info(f"UNet loaded, device: {next(unet.parameters()).device}")
        
        # Only move to device if device is specified
        if device is not None:
            logger.info(f"Moving models to device: {device}")
            try:
                unet = unet.to(device)
                logger.info(f"UNet moved to device: {next(unet.parameters()).device}")
                vae = vae.to(device)
                logger.info(f"VAE moved to device: {next(vae.parameters()).device}")
            except Exception as e:
                logger.error(f"Error moving models to device: {str(e)}")
                raise
        
        return unet, vae

    except Exception as e:
        logger.error(f"Failed to setup models: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def initialize_model_weights(model: torch.nn.Module):
    """Initialize model weights using improved techniques."""
    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Initialize attention layer weights with scaled normal distribution
            if "attn" in module._get_name().lower():
                scale = 1 / math.sqrt(module.in_features if hasattr(module, "in_features") else module.in_channels)
                torch.nn.init.normal_(module.weight, mean=0.0, std=scale)
            else:
                # Standard initialization for other layers
                torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, torch.nn.Embedding):
            # Initialize time embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # Apply initialization
    model.apply(_init_weights)
    
    # Zero initialize the output projection
    if hasattr(model, "conv_out"):
        torch.nn.init.zeros_(model.conv_out.weight)
        if model.conv_out.bias is not None:
            torch.nn.init.zeros_(model.conv_out.bias)

def configure_model_memory_format(
    model: Union[torch.nn.Module, DistributedDataParallel, FullyShardedDataParallel],
    config: Config
) -> Union[torch.nn.Module, DistributedDataParallel, FullyShardedDataParallel]:
    """Configure model memory format and optimizations."""
    try:
        # Set memory format if configured
        if config.system.channels_last:
            if not isinstance(model, (torch.nn.parallel.DistributedDataParallel, 
                                    torch.distributed.fsdp.FullyShardedDataParallel)):
                # Convert model to channels last format
                for param in model.parameters():
                    if param.dim() == 4:  # Only convert 4D tensors (NCHW format)
                        param.data = param.data.to(memory_format=torch.channels_last)
                model = model.to(memory_format=torch.channels_last)
            
        # Enable gradient checkpointing if configured
        if config.system.gradient_checkpointing:
            model.enable_gradient_checkpointing()
            
        # Enable xformers if available and configured
        if config.system.enable_xformers and is_xformers_installed():
            model.enable_xformers_memory_efficient_attention()
            
        return model
        
    except Exception as e:
        logger.error(f"Failed to configure model memory format: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def setup_text_encoders(model_name: str, device: torch.device, subfolder: str = "text_encoder"):
    """Initialize SDXL text encoders and tokenizers."""
    # Load text encoder 1
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_name, 
        subfolder=subfolder
    ).to(device)
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer"
    )
    
    # Load text encoder 2
    text_encoder_2 = CLIPTextModel.from_pretrained(
        model_name,
        subfolder=f"{subfolder}_2"
    ).to(device)
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer_2"
    )
    
    return (text_encoder_1, text_encoder_2), (tokenizer_1, tokenizer_2)

def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device where the model is located."""
    return next(model.parameters()).device