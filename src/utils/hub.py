from huggingface_hub import HfApi
import logging
import traceback
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers.optimization import Adafactor
from diffusers.optimization import get_scheduler
from data.dataset import CustomDataset
from models.tag_weighter import TagBasedLossWeighter
from models.vae_finetuner import VAEFineTuner
from data.dataset import custom_collate

logger = logging.getLogger(__name__)

def push_to_hub(model_id, model_path, private=False, model_card=None):
    """
    Push model to Hugging Face Hub
    
    Args:
        model_id (str): Hugging Face Hub model ID
        model_path (str): Local path to model files
        private (bool): Whether to create a private repository
        model_card (str): Model card content
    """
    try:
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=model_id,
                private=private,
                repo_type="model",
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repository creation warning (may already exist): {str(e)}")
        
        # Upload model files
        logger.info(f"Uploading model to {model_id}...")
        api.upload_folder(
            repo_id=model_id,
            folder_path=model_path,
            commit_message="Upload model files"
        )
        
        # Upload model card if provided
        if model_card:
            logger.info("Uploading model card...")
            api.upload_file(
                repo_id=model_id,
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                commit_message="Update model card"
            )
        
        logger.info(f"Successfully pushed model to https://huggingface.co/{model_id}")
        
    except Exception as e:
        logger.error(f"Error pushing to Hub: {str(e)}")
        raise

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
        
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
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
        logger.info("Setting up learning rate scheduler...")
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
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
            "num_training_steps": num_training_steps
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
