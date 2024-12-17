import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from src.data.dataset import NovelAIDataset, NovelAIDatasetConfig
from src.data.sampler import AspectBatchSampler
from src.training.trainer import NovelAIDiffusionV3Trainer
from src.utils.model import setup_model
from src.config.config import Config
from src.utils.setup import setup_distributed, verify_memory_optimizations
from src.utils.metrics import setup_logging, log_system_info, cleanup_logging
from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter, TagWeightingConfig
from src.data.validation import validate_directories
import gc
import traceback
from src.config.arg_parser import parse_args

logger = logging.getLogger(__name__)


def train(config_path: str):
    """Main training function with improved setup and error handling."""
    is_main_process = True  # Move this to the top of the function
    
    try:
        # Parse arguments first
        args = parse_args()
        
        # Load config
        config = Config.from_yaml(args.config)
        
        # Debug: Validate config before proceeding
        logger.info("Validating configuration...")
        logger.info(f"Data config:")
        logger.info(f"- Image size: {config.data.image_size}")
        logger.info(f"- Min size: {config.data.min_image_size}")
        logger.info(f"- Max dim: {config.data.max_dim}")
        logger.info(f"- Bucket step: {config.data.bucket_step}")
        logger.info(f"- Min bucket size: {config.data.min_bucket_size}")
        logger.info(f"- Bucket tolerance: {config.data.bucket_tolerance}")
        logger.info(f"- Max aspect ratio: {config.data.max_aspect_ratio}")
        logger.info(f"- Cache dir: {config.data.cache_dir}")
        logger.info(f"- Text cache dir: {config.data.text_cache_dir}")
        logger.info(f"- Num workers: {config.data.num_workers}")
        logger.info(f"- Pin memory: {config.data.pin_memory}")
        logger.info(f"- Persistent workers: {config.data.persistent_workers}")
        
        logger.info(f"\nTraining config:")
        logger.info(f"- Batch size: {config.training.batch_size}")
        logger.info(f"- Learning rate: {config.training.learning_rate}")
        logger.info(f"- Num epochs: {config.training.num_epochs}")
        logger.info(f"- Mixed precision: {config.training.mixed_precision}")
        
        logger.info(f"\nSystem config:")
        logger.info(f"- Enable xformers: {config.system.enable_xformers}")
        logger.info(f"- Gradient checkpointing: {config.system.gradient_checkpointing}")
        logger.info(f"- Mixed precision: {config.system.mixed_precision}")
        
        # Validate required directories exist
        for dir_path in [
            config.paths.checkpoints_dir,
            config.paths.logs_dir,
            config.paths.output_dir,
            config.data.cache_dir,
            config.data.text_cache_dir
        ]:
            if not os.path.exists(dir_path):
                logger.info(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
        
        # Validate image directories
        for img_dir in config.data.image_dirs:
            if not os.path.exists(img_dir):
                raise ValueError(f"Image directory not found: {img_dir}")
            logger.info(f"Found image directory: {img_dir}")
        
        # Validate model paths
        def is_valid_model_name(name: str) -> bool:
            """Validate model name or path."""
            if os.path.exists(name):
                return True
            valid_prefixes = (
                "stabilityai/",
                "runwayml/",
                "CompVis/",
                "madebyollin/",
                "openai/",
                "facebook/"
            )
            return any(name.startswith(prefix) for prefix in valid_prefixes)

        if not is_valid_model_name(config.model.pretrained_model_name):
            logger.warning(f"Model path may not be valid: {config.model.pretrained_model_name}")
        
        # Validate numeric parameters
        if config.training.batch_size < 1:
            raise ValueError(f"Invalid batch size: {config.training.batch_size}")
        if config.training.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {config.training.learning_rate}")
        if config.training.num_epochs < 1:
            raise ValueError(f"Invalid number of epochs: {config.training.num_epochs}")
        if config.data.num_workers < 0:
            raise ValueError(f"Invalid number of workers: {config.data.num_workers}")
        
        logger.info("Configuration validation complete!")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure logging and wandb
        setup_logging(config, config.paths.logs_dir, is_main_process)
        
        # Log system information
        if is_main_process:
            log_system_info()
        
        # Validate directories
        valid_dirs, total_images = validate_directories(config)
        if is_main_process:
            logger.info(f"Found {total_images} valid images across {len(valid_dirs)} directories")
        config.data.image_dirs = valid_dirs
        
        # Setup models
        logger.info("Setting up models...")
        unet, vae = setup_model(args, device, config)
        
        # Apply memory optimizations and verify
        optimization_states = verify_memory_optimizations(
            model=unet,
            config=config,
            device=device,
            logger=logger
        )
        
        if not all(optimization_states.values()):
            logger.warning("Some memory optimizations failed to apply:")
            for opt, state in optimization_states.items():
                if not state:
                    logger.warning(f"- Failed to enable {opt}")
            logger.warning("Training will continue but may be less efficient")
        
        # Create trainer
        trainer = NovelAIDiffusionV3Trainer(
            config_path=config_path,
            model=unet,
            vae=vae,
            device=device
        )
        
        # Create dataset with text embedder and tag weighter
        dataset_config = NovelAIDatasetConfig(
            image_size=config.data.image_size,
            max_image_size=config.data.max_image_size,
            min_image_size=config.data.min_image_size,
            max_dim=config.data.max_dim,
            bucket_step=config.data.bucket_step,
            min_bucket_size=config.data.min_bucket_size,
            min_bucket_resolution=config.data.min_bucket_resolution,
            bucket_tolerance=config.data.bucket_tolerance,
            max_aspect_ratio=config.data.max_aspect_ratio,
            cache_dir=config.data.cache_dir,
            text_cache_dir=config.data.text_cache_dir,
            use_caching=config.data.use_caching,
            proportion_empty_prompts=config.data.proportion_empty_prompts,
            batch_size=config.training.batch_size,
            max_consecutive_batch_samples=2
        )
        
        dataset = NovelAIDataset(
            image_dirs=config.data.image_dirs,
            text_embedder=TextEmbedder(
                pretrained_model_name_or_path=config.model.pretrained_model_name,
                device=device,
                dtype=torch.bfloat16 if config.system.mixed_precision == "bf16" else torch.float32
            ),
            tag_weighter=TagWeighter(
                config=TagWeightingConfig(**config.tag_weighting.__dict__)
            ),
            vae=vae,
            config=dataset_config,
            device=device
        )
        
        if len(dataset) == 0:
            raise ValueError("Dataset contains no valid samples after initialization")
        
        # Create sampler
        sampler = AspectBatchSampler(
            dataset=dataset,
            batch_size=config.training.batch_size,
            drop_last=True,
            shuffle=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            collate_fn=trainer.collate_fn
        )
        
        # Assign dataloader to trainer
        trainer.dataloader = dataloader
        
        # Train
        logger.info("Starting training...")
        for epoch in range(config.training.num_epochs):
            trainer.train_epoch(epoch)
            
            # Save checkpoint
            if is_main_process:
                trainer.save_checkpoint(config.paths.checkpoints_dir, epoch)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        # Use cleanup function
        cleanup_logging(is_main_process)
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    from src.config.arg_parser import parse_args
    args = parse_args()
    train(args.config)