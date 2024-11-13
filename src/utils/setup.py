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
    """Initialize and configure all models"""
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
    """Setup training components"""
    logger.info("Setting up training components...")
    
    try:
        # Create dataset and dataloader
        dataset = CustomDataset(
            data_dir=args.data_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            cache_dir=args.cache_dir,
            no_caching_latents=args.no_caching_latents
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
        
        # Initialize VAE finetuner if enabled
        vae_finetuner = None
        if args.finetune_vae:
            logger.info("Initializing VAE finetuner...")
            vae_finetuner = VAEFineTuner(
                models["vae"],
                learning_rate=args.vae_learning_rate,
                device=device
            )
        
        # Initialize validator with existing components
        logger.info("Initializing validator...")
        validator = SDXLInference(
            model_path=None,  # Don't load any model
            device=device,
            dtype=dtype
        )
        
        # Create pipeline directly from components
        validator.pipeline = StableDiffusionXLPipeline(
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            unet=models["unet"],
            scheduler=None  # Will be set by the validator
        ).to(device)
        
        # Configure scheduler settings
        validator.pipeline.scheduler.register_to_config(
            use_resolution_binning=True,
            sigma_min=args.sigma_min,
            sigma_data=1.0
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

