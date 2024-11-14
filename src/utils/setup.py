import os
import logging
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from torch.optim import AdamW, Adafactor
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler

from training.ema import EMAModel
from data.dataset import create_dataloader
from utils.validation import Validator
from data.tag_weighter import TagWeighter
from training.vae_finetuner import VAEFinetuner

logger = logging.getLogger(__name__)

def setup_torch_backends():
    """Configure PyTorch backend settings for optimal performance"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def setup_models(args, device, dtype):
    """Initialize and configure all models"""
    logger.info("\nLoading models...")
    
    # Load base pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        use_safetensors=True
    )
    
    # Move models to device
    unet = pipeline.unet.to(device)
    vae = pipeline.vae.to(device)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.finetune_vae:
            vae.enable_gradient_checkpointing()
    
    # Setup EMA if requested
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            unet,
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            min_decay=args.ema_min_decay,
            max_decay=args.ema_max_decay,
            update_every=args.ema_update_every,
            use_warmup=args.use_ema_warmup,
            grad_scale_factor=args.ema_grad_scale_factor
        )
    
    # Enable model compilation if requested
    if args.enable_compile:
        logger.info("Compiling UNet...")
        unet = torch.compile(unet, mode=args.compile_mode)
    
    return {
        "pipeline": pipeline,
        "unet": unet,
        "vae": vae,
        "ema": ema_model
    }

def setup_training(args, models, device, dtype):
    """Initialize all training components"""
    logger.info("\nSetting up training components...")
    
    # Create optimizer
    if args.use_adafactor:
        optimizer = Adafactor(
            models["unet"].parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=args.learning_rate
        )
    else:
        optimizer = AdamW(
            models["unet"].parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay
        )
    
    # Create dataloader
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir if not args.no_caching_latents else None,
        all_ar=args.all_ar
    )
    
    # Calculate number of update steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Initialize tag weighter if requested
    tag_weighter = None
    if args.use_tag_weighting:
        tag_weighter = TagWeighter(
            min_weight=args.min_tag_weight,
            max_weight=args.max_tag_weight
        )
    
    # Initialize VAE finetuner if requested
    vae_finetuner = None
    if args.finetune_vae:
        vae_finetuner = VAEFinetuner(
            vae=models["vae"],
            learning_rate=args.vae_learning_rate,
            device=device,
            dtype=dtype
        )
    
    # Initialize validator
    validator = Validator(
        pipeline=models["pipeline"],
        device=device,
        dtype=dtype
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "train_dataloader": train_dataloader,
        "tag_weighter": tag_weighter,
        "vae_finetuner": vae_finetuner,
        "validator": validator,
        "ema_model": models["ema"],
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "max_train_steps": max_train_steps
    }