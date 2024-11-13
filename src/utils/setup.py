import torch
import logging
import traceback
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers.optimization import Adafactor
from diffusers.optimization import get_scheduler
from data.dataset import CustomDataset
from models.tag_weighter import TagBasedLossWeighter
from models.vae_finetuner import VAEFineTuner
from inference.text_to_image import ModelValidator
from training.ema import EMAModel
import wandb
import numpy as np
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from data.dataset import custom_collate


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
        
        # Enable gradient checkpointing if requested (moved after model loading)
        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for models")
            # Enable for UNet
            enable_gradient_checkpointing(unet)
            
            # Enable for text encoders
            enable_gradient_checkpointing(text_encoder)
            enable_gradient_checkpointing(text_encoder_2)
            
            logger.info("Gradient checkpointing enabled for all supported models")
        
        # Freeze text encoders
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder.eval()
        text_encoder_2.eval()
        
        # Initialize EMA if enabled
        ema_model = None
        if args.use_ema:
            logger.info("Initializing EMA model...")
            ema_model = EMAModel(
                model=unet,
                decay=args.ema_decay,
                device=device
            )
        
        # Initialize validator
        validator = ModelValidator(
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            zsnr=True
        )
        
        # Return all components
        models = {
            "unet": unet,
            "vae": vae,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "validator": validator,
            "ema_model": ema_model
        }
        
        logger.info("Model setup completed successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error during model setup: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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

def setup_training(args, models, device, dtype):
    """
    Initialize all training components
    
    Args:
        args: Training arguments
        models: Dictionary of model components
        device: Target device
        dtype: Model precision
        
    Returns:
        dict: Dictionary containing all training components
    """
    try:
        logger.info("Setting up training components...")
        
        # Create dataset with automatic tag processing
        dataset = CustomDataset(
            data_dir=args.data_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            cache_dir=args.cache_dir
        )
        
        # Log dataset and tag processing statistics
        if args.use_wandb:
            wandb.run.summary.update({
                "dataset/total_images": dataset.tag_stats['total_images'],
                "dataset/formatted_captions": dataset.tag_stats['formatted_count'],
                "dataset/niji_ratio": dataset.tag_stats['niji_count'] / dataset.tag_stats['total_images'],
                "dataset/quality_6_ratio": dataset.tag_stats['quality_6_count'] / dataset.tag_stats['total_images'],
                "dataset/stylize_mean": np.mean(dataset.tag_stats['stylize_values']),
                "dataset/chaos_mean": np.mean(dataset.tag_stats['chaos_values']) if dataset.tag_stats['chaos_values'] else 0
            })
        
        logger.info(f"Processed {dataset.tag_stats['formatted_count']} captions")
        logger.info(f"Found {dataset.tag_stats['niji_count']} anime-style images")
        
        # Create dataloader with the imported custom_collate function
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Configure memory optimizations
        if args.gradient_checkpointing:
            logger.info("Gradient checkpointing enabled - configuring memory optimizations")
            
            # Disable model parameters' gradients before training
            for param in models["unet"].parameters():
                param.requires_grad_(False)
            models["unet"].requires_grad_(True)
            
            # Configure text encoder gradients if they're being trained
            for encoder_name in ["text_encoder", "text_encoder_2"]:
                if encoder_name in models and models[encoder_name] is not None:
                    for param in models[encoder_name].parameters():
                        param.requires_grad_(False)
                    models[encoder_name].requires_grad_(True)
            
            # Empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Memory optimizations configured")
        
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
        
        # Initialize VAE finetuner if enabled
        vae_finetuner = None
        if args.finetune_vae:
            logger.info("Initializing VAE finetuner...")
            vae_finetuner = VAEFineTuner(
                models["vae"],
                learning_rate=args.vae_learning_rate,
                device=device
            )
        
        # Initialize validator with correct parameters
        validator = ModelValidator(
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            zsnr=args.zsnr,
            sigma_min=args.sigma_min,
            sigma_data=args.sigma_data,
            min_snr_gamma=args.min_snr_gamma,
            resolution_scaling=args.resolution_scaling,
            rescale_cfg=args.rescale_cfg,
            scale_method=args.scale_method,
            rescale_multiplier=args.rescale_multiplier
        )
        
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

