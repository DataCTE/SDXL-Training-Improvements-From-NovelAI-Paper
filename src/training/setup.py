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
    logger.info("Setting up training components...")
    
    try:
        # Create dataset and dataloader
        logger.info("Creating dataset...")
        dataset = CustomDataset(
            args.data_dir,
            models["vae"],
            models["tokenizer"],
            models["tokenizer_2"],
            models["text_encoder"],
            models["text_encoder_2"],
            cache_dir=args.cache_dir,
            batch_size=args.batch_size
        )
        
        # Group dataset samples by aspect ratio and size
        def collate_fn(batch):
            # Sort by height, width to group similar sizes
            batch.sort(key=lambda x: (x["latents"].shape[1], x["latents"].shape[2]))
            
            # Group by exact dimensions
            grouped = {}
            for item in batch:
                size = (item["latents"].shape[1], item["latents"].shape[2])
                if size not in grouped:
                    grouped[size] = []
                grouped[size].append(item)
            
            # Take the largest group that fits in a batch
            largest_group = max(grouped.values(), key=len)
            
            # Create batch tensors
            batch_dict = {
                "latents": torch.stack([x["latents"] for x in largest_group]),
                "text_embeddings": torch.stack([x["text_embeddings"] for x in largest_group]),
                "pooled_text_embeddings_2": torch.stack([x["pooled_text_embeddings_2"] for x in largest_group]),
                "tags": [x["tags"] for x in largest_group]
            }
            
            return batch_dict
        
        # Create dataloader with custom collate
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
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
            "validator": models.get("validator", None)
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
