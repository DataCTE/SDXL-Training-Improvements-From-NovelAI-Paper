import os
import logging
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.data import NovelAIDataset, NovelAIDatasetConfig
from src.training.trainer import NovelAIDiffusionV3Trainer
from src.utils.model.model import setup_model, setup_text_encoders
from src.config.config import Config, TagWeighterConfig
from src.utils.system.setup import verify_memory_optimizations, setup_memory_optimizations
from src.utils.logging.metrics import setup_logging, log_system_info, cleanup_logging
from src.data.processors.utils.caption.tag_weighter import TagWeighter, parse_tags
import gc
import traceback
from src.config.arg_parser import parse_args
from src.config.config import DEFAULT_TARGET_RESOLUTIONS, DEFAULT_MAX_AR_ERROR
logger = logging.getLogger(__name__)


def train(config_path: str):
    """Main training function with improved setup and error handling."""
    is_main_process = True
    loop = None
    dataset = None
    
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
        logger.info(f"- Prediction type: {config.training.prediction_type}")
        logger.info(f"- SNR gamma: {config.training.snr_gamma}")
        
        logger.info(f"\nSystem config:")
        logger.info(f"- Enable xformers: {config.system.enable_xformers}")
        logger.info(f"- Gradient checkpointing: {config.system.gradient_checkpointing}")
        logger.info(f"- Mixed precision: {config.system.mixed_precision}")
        logger.info(f"- Channels last: {config.system.channels_last}")
        
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
        
        # Basic validation of image directories
        valid_image_dirs = []
        total_dirs = len(config.data.image_dirs)
        for img_dir in config.data.image_dirs:
            if not os.path.exists(img_dir):
                logger.warning(f"Image directory not found: {img_dir}")
                continue
            if not os.path.isdir(img_dir):
                logger.warning(f"Path is not a directory: {img_dir}")
                continue
            valid_image_dirs.append(img_dir)
            logger.info(f"Found valid image directory: {img_dir}")
            
        if not valid_image_dirs:
            raise ValueError("No valid image directories found")
            
        logger.info(f"Found {len(valid_image_dirs)} valid directories out of {total_dirs}")
        config.data.image_dirs = valid_image_dirs

        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure logging and wandb
        setup_logging(config, config.paths.logs_dir, is_main_process)
        
        # Log system information
        if is_main_process:
            log_system_info()
        
        # Setup models and encoders
        logger.info("Setting up models...")
        unet, vae = setup_model(args, device, config)
        
        # Initialize text encoders and tokenizers
        text_encoders, tokenizers = setup_text_encoders(
            config.model.pretrained_model_name,
            device=device,
            subfolder="text_encoder"
        )
        
        # Setup memory optimizations
        batch_size = config.training.batch_size
        micro_batch_size = batch_size // config.training.gradient_accumulation_steps
        
        memory_setup = setup_memory_optimizations(
            model=unet,
            config=config,
            device=device,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size
        )
        
        if not memory_setup:
            logger.warning("Memory optimizations setup failed")
        
        # Verify memory optimizations
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
        
        # Initialize tag weighter with proper configuration
        tag_weighter_config = TagWeighterConfig(
            default_weight=config.tag_weighting.default_weight,
            min_weight=config.tag_weighting.min_weight,
            max_weight=config.tag_weighting.max_weight,
            smoothing_factor=config.tag_weighting.smoothing_factor
        )
        
        TagWeighter(config=tag_weighter_config)
        
        # Create dataset config
        dataset_config = NovelAIDatasetConfig(
            image_dirs=config.data.image_dirs,
            image_size=config.data.image_size,
            max_size=config.data.max_image_size,
            min_size=config.data.min_image_size,
            max_dim=config.data.max_dim,
            step=config.data.bucket_step,
            min_resolution=config.data.min_bucket_resolution,
            max_ar=config.data.max_aspect_ratio,
            tolerance=config.data.bucket_tolerance,
            batch_size=config.training.batch_size,
            shuffle=config.data.shuffle,
            drop_last=config.data.drop_last if hasattr(config.data, 'drop_last') else False,
            debug_mode=config.data.debug_mode if hasattr(config.data, 'debug_mode') else False,
            prefetch_factor=config.data.prefetch_factor if hasattr(config.data, 'prefetch_factor') else None,
            target_resolutions=DEFAULT_TARGET_RESOLUTIONS,
            max_ar_error=DEFAULT_MAX_AR_ERROR
        )
        
        # Log dataset configuration
        logger.info(f"Initializing dataset with config:")
        logger.info(f"- Image directories: {dataset_config.image_dirs}")
        logger.info(f"- Batch size: {dataset_config.batch_size}")
        logger.info(f"- Image size: {dataset_config.image_size}")
        logger.info(f"- Max image size: {dataset_config.max_image_size}")
        
        # Create async event loop for dataset operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize dataset with async context
        async def init_dataset():
            # Create tag weighter if enabled
            tag_weighter = None
            if config.data.use_tag_weighting:
                tag_weighter = TagWeighter(
                    weight_ranges=config.data.tag_weight_ranges,
                    save_path=config.paths.tag_weights_path
                )
                if config.paths.tag_weights_path and os.path.exists(config.paths.tag_weights_path):
                    tag_weighter = TagWeighter.load(config.paths.tag_weights_path)
            
            return await NovelAIDataset.create(
                config=dataset_config,
                vae=vae,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                tag_weighter=tag_weighter
            )

        try:
            # Run dataset initialization in the event loop
            dataset = loop.run_until_complete(init_dataset())
            
            if len(dataset) == 0:
                raise ValueError("Dataset contains no valid samples after initialization")
                
            logger.info(f"Dataset initialized with {len(dataset)} samples")
            
            # Create trainer with async-aware dataset
            trainer = NovelAIDiffusionV3Trainer(
                config=config,
                model=unet,
                dataset=dataset,
                device=device
            )
            
            # Training loop with proper async cleanup
            for epoch in range(config.training.num_epochs):
                try:
                    trainer.train_epoch(epoch)
                    
                    # Save checkpoint
                    if is_main_process:
                        trainer.save_checkpoint(config.paths.checkpoints_dir, epoch)
                        
                except Exception as e:
                    logger.error(f"Error during epoch {epoch}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                finally:
                    # Ensure batch processor cleanup after each epoch
                    loop.run_until_complete(dataset.batch_processor.cleanup())
                    
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if dataset is not None:
                loop.run_until_complete(dataset.cleanup())
            if loop is not None and not loop.is_closed():
                loop.close()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        cleanup_logging(is_main_process)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    from src.config.arg_parser import parse_args
    args = parse_args()
    train(args.config)