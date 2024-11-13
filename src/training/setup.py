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
from training.utils import custom_collate
from models.model_validator import ModelValidator
import wandb
import numpy as np

logger = logging.getLogger(__name__)

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
        
        # Group dataset samples by aspect ratio and size
        def collate_fn(batch):
            # Filter out samples that are too small
            valid_batch = []
            for item in batch:
                latent_h, latent_w = item["latents"].shape[-2:]
                # Check if latent dimensions meet minimum size (32x32 for 256x256 pixel images)
                if latent_h >= 32 and latent_w >= 32:
                    valid_batch.append(item)
            
            if not valid_batch:
                raise ValueError("No valid samples in batch (all below minimum size)")
                
            # Sort remaining samples by size
            valid_batch.sort(key=lambda x: (x["latents"].shape[-2], x["latents"].shape[-1]))
            
            # Group by exact dimensions
            grouped = {}
            for item in valid_batch:
                size = (item["latents"].shape[-2], item["latents"].shape[-1])
                if size not in grouped:
                    grouped[size] = []
                grouped[size].append(item)
            
            # Take the largest group that fits in a batch
            largest_group = max(grouped.values(), key=len)
            
            # Create batch tensors with updated keys
            batch_dict = {
                "latents": torch.stack([x["latents"] for x in largest_group]),
                "text_embeddings": torch.stack([x["text_embeddings"] for x in largest_group]),
                "added_cond_kwargs": {
                    "text_embeds": torch.stack([
                        x["added_cond_kwargs"]["text_embeds"] for x in largest_group
                    ]),
                    "time_ids": torch.stack([
                        x["added_cond_kwargs"]["time_ids"] for x in largest_group
                    ])
                }
            }
            
            return batch_dict
        
        # Create dataloader with updated collate function
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Drop incomplete batches
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

def verify_training_components(train_components):
    """
    Verify that all required training components are present and properly configured
    
    Args:
        train_components (dict): Dictionary of training components
        
    Returns:
        bool: True if verification passes
    """
    required_components = [
        "dataset",
        "train_dataloader",
        "optimizer",
        "lr_scheduler",
        "tag_weighter",
        "num_update_steps_per_epoch",
        "num_training_steps"
    ]
    
    try:
        # Check for required components
        for component_name in required_components:
            if component_name not in train_components:
                raise ValueError(f"Missing required component: {component_name}")
            
        # Verify dataloader
        if len(train_components["train_dataloader"]) == 0:
            raise ValueError("Empty training dataloader")
            
        # Verify optimizer
        if len(list(train_components["optimizer"].param_groups)) == 0:
            raise ValueError("Optimizer has no parameter groups")
            
        return True
        
    except Exception as e:
        logger.error(f"Training component verification failed: {str(e)}")
        return False
