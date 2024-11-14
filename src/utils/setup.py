import torch
import logging
import traceback
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers.optimization import Adafactor
from diffusers.optimization import get_scheduler
from data.dataset import CustomDataset
from data.tag_weighter import TagBasedLossWeighter
from training.vae_finetuner import VAEFineTuner
from inference.text_to_image import SDXLInference
from training.ema import EMAModel
import wandb
import numpy as np
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from data.dataset import custom_collate
from data.dataset import CustomDataset
from inference.text_to_image import SDXLInference
from training.ema import EMAModel
from data.tag_weighter import TagBasedLossWeighter
from training.vae_finetuner import VAEFineTuner
from utils.device import cleanup
from diffusers import StableDiffusionXLPipeline
from utils.validation import validate_dataset, validate_model_components, validate_training_args, validate_vae_finetuner_config, validate_ema_config
import os
from diffusers import EulerDiscreteScheduler


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


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for a model"""
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        logger.warning(f"Model {type(model).__name__} doesn't support gradient checkpointing")

def verify_unet_config(config):
    """Verify that UNet configuration matches SDXL architecture requirements
    
    According to Table 1 and Section 2.1 of the SDXL paper:
    - Uses heterogeneous distribution of transformer blocks [0,2,10]
    - Removes lowest level (8x downsampling)
    - Channel multipliers [1,2,4]
    - Context dimension 2048 (for combined CLIP ViT-L & OpenCLIP ViT-bigG)
    """
    expected_config = {
        'cross_attention_dim': 2048,  # Combined dimension for dual text encoders
        'transformer_layers_per_block': [0, 2, 10],  # Heterogeneous transformer distribution
        'channel_mult': [1, 2, 4],  # Channel multipliers without 8x level
        'in_channels': 4,  # Standard for SD models
        'out_channels': 4,  # Standard for SD models
        'num_res_blocks': 2,  # Standard architecture 
        'attention_resolutions': [4, 8],  # Attention at 1/4 and 1/8 resolution
        'use_linear_projection': True,  # For improved stability
    }
    
    missing_keys = []
    mismatched_values = []
    
    for key, value in expected_config.items():
        if key not in config:
            missing_keys.append(key)
        elif config[key] != value:
            mismatched_values.append(
                f"{key}: expected {value}, got {config[key]}"
            )
    
    if missing_keys:
        logger.warning(f"Missing config keys: {', '.join(missing_keys)}")
    
    if mismatched_values:
        logger.warning(
            "Config mismatches found:\n" + 
            "\n".join(mismatched_values)
        )
        
    if missing_keys or mismatched_values:
        logger.warning(
            "Model architecture differs from SDXL specification. "
            "This may cause compatibility issues."
        )
    else:
        logger.info("UNet configuration verification passed")
    
    return True

def setup_models(args, device, dtype):
    """Initialize and configure all models"""
    logger.info("Setting up models...")
    
    try:
        # Load UNet
        logger.info("Loading UNet...")
        unet_config = UNet2DConditionModel.load_config(
            args.model_path,
            subfolder="unet"
        )
        logger.info(f"Loaded UNet config: {unet_config}")
        
        # Fix SDXL architecture configuration
        sdxl_updates = {
            'transformer_layers_per_block': [0, 2, 10],  # Correct transformer distribution
            'channel_mult': [1, 2, 4],  # Standard SDXL channel multipliers
            'attention_resolutions': [4, 8],  # Attention at 1/4 and 1/8 resolution
            'use_linear_projection': True,  # For improved stability
            'cross_attention_dim': 2048,  # Combined CLIP dimension
            'num_res_blocks': 2  # Standard architecture
        }
        
        # Update config with SDXL architecture requirements
        for key, value in sdxl_updates.items():
            if key not in unet_config or unet_config[key] != value:
                logger.info(f"Updating UNet config: {key} = {value}")
                unet_config[key] = value
        
        # Verify configuration matches SDXL architecture
        verify_unet_config(unet_config)
        
        # Initialize UNet with corrected config
        unet = UNet2DConditionModel.from_config(unet_config)
        
        # Load pretrained weights
        pretrained_unet = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=dtype
        )
        
        # Transfer compatible weights
        missing_keys, unexpected_keys = unet.load_state_dict(
            pretrained_unet.state_dict(), 
            strict=False
        )
        
        if missing_keys:
            logger.warning(f"Missing keys when loading UNet weights: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading UNet weights: {unexpected_keys}")
        
        unet = unet.to(device)
        
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
            args.model_path, subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.model_path, subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path, subfolder="text_encoder"
        ).to(device)
        text_encoder_2 = CLIPTextModel.from_pretrained(
            args.model_path, subfolder="text_encoder_2"
        ).to(device)
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for models")
            for model in [unet, text_encoder, text_encoder_2]:
                enable_gradient_checkpointing(model)
            logger.info("Gradient checkpointing enabled for all supported models")
        
        # Initialize EMA if requested
        ema_model = None
        if args.use_ema:
            logger.info("Initializing EMA model...")
            ema_model = EMAModel(
                unet,
                decay=args.ema_decay,
                device=device
            )
        
        # Create models dictionary
        models = {
            "unet": unet,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "ema_model": ema_model
        }
        
        logger.info("Model setup completed successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error during model setup: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def verify_models(models):
    """
    Verify that all models are present and properly configured
    
    Args:
        models (dict): Dictionary of model components
        
    Returns:
        bool: True if verification passes
    """
    try:
        is_valid, error_msg = validate_model_components(models)
        if not is_valid:
            raise ValueError(error_msg)
            
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def setup_training(args, models, device, dtype):
    """Setup training components"""
    logger.info("Setting up training components...")
    
    try:
        # Validate training arguments first
        is_valid, error_msg = validate_training_args(args)
        if not is_valid:
            raise ValueError(f"Training arguments validation failed: {error_msg}")
            
        # Add default num_workers if not specified
        if not hasattr(args, 'num_workers'):
            args.num_workers = min(8, os.cpu_count() or 1)  # Default to min(8, CPU cores)
            logger.info(f"Using default num_workers: {args.num_workers}")

        # Initialize dataset without validation if all_ar is True
        if args.all_ar:
            dataset = CustomDataset(
                data_dir=args.data_dir,
                vae=models["vae"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                cache_dir=args.cache_dir,
                no_caching_latents=args.no_caching_latents,
                all_ar=True,
                num_workers=args.num_workers
            )
        else:
            # Validate dataset first
            valid, stats = validate_dataset(args.data_dir)
            if not valid:
                raise ValueError(f"Dataset validation failed: {stats}")
                
            dataset = CustomDataset(
                data_dir=args.data_dir,
                vae=models["vae"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                cache_dir=args.cache_dir,
                no_caching_latents=args.no_caching_latents,
                all_ar=False,
                num_workers=args.num_workers
            )
            
        # Setup VAE finetuner if enabled
        if hasattr(args, 'vae_finetuning') and args.vae_finetuning:
            vae_config = {
                'learning_rate': args.vae_learning_rate,
                'min_snr_gamma': args.min_snr_gamma,
                'adaptive_loss_scale': args.adaptive_loss_scale,
                'kl_weight': args.kl_weight,
                'perceptual_weight': args.perceptual_weight,
                'use_8bit_adam': args.use_8bit_adam,
                'gradient_checkpointing': args.gradient_checkpointing,
                'mixed_precision': args.mixed_precision,
                'use_channel_scaling': args.use_channel_scaling
            }
            
            is_valid, error_msg = validate_vae_finetuner_config(vae_config)
            if not is_valid:
                raise ValueError(f"VAE finetuner validation failed: {error_msg}")
                
            vae_finetuner = VAEFineTuner(
                models["vae"],
                **vae_config
            )
            
        # Setup EMA if enabled
        if hasattr(args, 'use_ema') and args.use_ema:
            ema_config = {
                'decay': args.ema_decay,
                'update_after_step': args.ema_update_after_step,
                'inv_gamma': args.ema_inv_gamma,
                'power': args.ema_power,
                'min_decay': args.ema_min_decay,
                'max_decay': args.ema_max_decay,
                'update_every': args.ema_update_every,
                'use_ema_warmup': args.use_ema_warmup,
                'grad_scale_factor': args.ema_grad_scale_factor
            }
            
            is_valid, error_msg = validate_ema_config(ema_config)
            if not is_valid:
                raise ValueError(f"EMA validation failed: {error_msg}")
                
            ema = EMAModel(
                models["unet"],
                **ema_config,
                device=device
            )

        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Initialize optimizer
        logger.info("Initializing optimizer...")
        if args.use_adafactor:
            optimizer = Adafactor(
                models["unet"].parameters(),
                lr=args.learning_rate * args.batch_size,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False
            )
        else:
            optimizer = AdamW8bit(
                models["unet"].parameters(),
                lr=args.learning_rate * args.batch_size,
                betas=(0.9, 0.999)
            )
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_training_steps = args.num_epochs * num_update_steps_per_epoch
        
        # Initialize learning rate scheduler
        logger.info("Setting up cosine learning rate scheduler...")
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize tag-based loss weighter
        logger.info("Initializing tag-based loss weighter...")
        tag_weighter = TagBasedLossWeighter(
            min_weight=args.min_tag_weight,
            max_weight=args.max_tag_weight
        )
        
        # Create scheduler first
        noise_scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            use_karras_sigmas=True,
            sigma_min=args.sigma_min,
            sigma_max=160.0,
            steps_offset=1,
        )

        # Create validator with pipeline
        validator = SDXLInference(
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            use_resolution_binning=True
        )

        # Update validator's pipeline with our models and scheduler
        validator.pipeline = StableDiffusionXLPipeline(
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            unet=models["unet"],
            scheduler=noise_scheduler,
        ).to(device)

        # Return all components
        train_components = {
            "dataset": dataset,
            "train_dataloader": train_dataloader,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "tag_weighter": tag_weighter,
            "vae_finetuner": vae_finetuner,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
            "num_training_steps": num_training_steps,
            "ema_model": models.get("ema_model", None),
            "validator": validator
        }
        
        logger.info("Training setup completed successfully")
        return train_components
        
    except Exception as e:
        logger.error(f"Error during training setup: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
