import torch
import os
import wandb
import sys
import logging
import multiprocessing
from typing import Optional, Tuple, List
from pathlib import Path
import traceback
import torch.distributed as dist

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project components
from src.data.dataset import NovelAIDataset
from src.training.trainer import NovelAIDiffusionV3Trainer
from src.training.vae_trainer import VAETrainer
from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter, TagWeightingConfig
from src.data.dataset import NovelAIDatasetConfig
from src.config.config import Config
from src.config.arg_parser import parse_args
from src.utils.setup import setup_accelerator, setup_distributed
from src.utils.model import setup_model
from src.data.validation import validate_directories
from src.data import get_optimal_cpu_threads
from diffusers import UNet2DConditionModel, AutoencoderKL

def main():
    """Main training function."""
    try:
        args = parse_args()
        
        # Load and validate config
        config = Config.from_yaml(args.config)
        
        # Setup distributed training
        world_size, rank = setup_distributed(config)
        is_main_process = rank == 0
        
        # Configure logging
        setup_logging(config.paths.logs_dir, is_main_process)
        
        # Log system info
        if is_main_process:
            log_system_info()
        
        # Setup accelerator
        accelerator = setup_accelerator(config)
        device = accelerator.device
        
        # Setup models with memory tracking
        try:
            if is_main_process:
                logger.info("Setting up models...")
                initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Use setup_model from model.py but don't move to device
            unet, vae = setup_model(args, None, config)  # Pass None as device to prevent device movement
            
            if is_main_process and torch.cuda.is_available():
                mem_change = torch.cuda.memory_allocated() - initial_mem
                logger.info(f"Models loaded. Memory usage: {mem_change/1024**3:.1f}GB")
                
            # Verify models were created successfully
            if not isinstance(unet, UNet2DConditionModel):
                raise RuntimeError("Failed to create UNet model")
            if not isinstance(vae, AutoencoderKL):
                raise RuntimeError("Failed to create VAE model")
                
        except Exception as e:
            logger.error(f"Failed to setup models: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
        
        # Apply sync batch norm if needed
        if config.system.use_fsdp and config.system.sync_batch_norm:
            try:
                unet = accelerator.sync_batchnorm(unet)
                if args.train_vae:
                    vae = accelerator.sync_batchnorm(vae)
            except Exception as e:
                logger.error(f"Failed to apply sync batch norm: {e}")
                raise
        
        # Initialize appropriate trainer - let trainer handle device movement
        try:
            if args.train_vae:
                trainer = VAETrainer(
                    config=config,
                    accelerator=accelerator,
                    resume_from_checkpoint=args.resume_from_checkpoint
                )
            else:
                trainer = NovelAIDiffusionV3Trainer(
                    config_path=args.config,
                    model=unet,      # Pass model without device movement
                    vae=vae,         # Pass VAE without device movement
                    accelerator=accelerator
                )
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
        
        # Validate directories and update config
        valid_dirs, total_images = validate_directories(config)
        config.data.image_dirs = valid_dirs
        
        if is_main_process:
            logger.info(f"Found {total_images} valid images across {len(valid_dirs)} directories")

        # Setup dataset with memory tracking
        try:
            if is_main_process:
                logger.info("Setting up dataset...")
                initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
            dataset = NovelAIDataset(
                image_dirs=valid_dirs,
                text_embedder=TextEmbedder(
                    pretrained_model_name_or_path=config.model.pretrained_model_name,
                    device=device,
                    dtype=torch.bfloat16 if config.system.mixed_precision == "bf16" else torch.float32
                ),
                tag_weighter=TagWeighter(
                    config=TagWeightingConfig(
                        min_weight=config.tag_weighting.min_weight,
                        max_weight=config.tag_weighting.max_weight,
                        default_weight=config.tag_weighting.default_weight,
                        enabled=config.tag_weighting.enabled,
                        update_frequency=config.tag_weighting.update_frequency,
                        smoothing_factor=config.tag_weighting.smoothing_factor
                    )
                ),
                vae=vae,
                config=NovelAIDatasetConfig(
                    image_size=config.data.image_size,
                    min_size=config.data.min_size,
                    max_dim=config.data.max_dim,
                    bucket_step=config.data.bucket_step,
                    min_bucket_size=config.data.min_bucket_size,
                    bucket_tolerance=config.data.bucket_tolerance,
                    max_aspect_ratio=config.data.max_aspect_ratio,
                    cache_dir=config.data.cache_dir,
                    use_caching=config.data.use_caching,
                    proportion_empty_prompts=config.data.proportion_empty_prompts
                ),
                device=device
            )
            
            if len(dataset) == 0:
                raise ValueError("Dataset contains no valid samples after initialization")
                
            if is_main_process and torch.cuda.is_available():
                mem_change = torch.cuda.memory_allocated() - initial_mem
                logger.info(f"Dataset initialized. Memory usage: {mem_change/1024**3:.1f}GB")
            
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {str(e)}")
            raise
        
        # Prepare for distributed training
        trainer, dataset = accelerator.prepare(trainer, dataset)
        
        # Set optimal number of workers
        config.data.num_workers = get_optimal_cpu_threads()
        
        # Create dataloader with memory tracking
        try:
            if is_main_process:
                logger.info("Creating dataloader...")
                initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
            train_dataloader = trainer.create_dataloader(
                dataset=dataset,
                batch_size=config.training.batch_size,
                pin_memory=True
            )
            
            if is_main_process and torch.cuda.is_available():
                mem_change = torch.cuda.memory_allocated() - initial_mem
                logger.info(f"Dataloader created. Memory usage: {mem_change/1024**3:.1f}GB")
                
        except Exception as e:
            logger.error(f"Failed to create dataloader: {str(e)}")
            raise
        
        # Train
        try:
            if is_main_process:
                logger.info("Starting training...")
            
            for epoch in range(config.training.num_epochs):
                avg_loss = trainer.train_epoch(epoch, train_dataloader)
                
                if is_main_process:
                    logger.info(f"Epoch {epoch} completed with average loss: {avg_loss:.4f}")
                    trainer.save_checkpoint(config.paths.checkpoints_dir, epoch)
                    
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise e
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise