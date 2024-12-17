import torch
import os
import wandb
import sys
import logging
import multiprocessing
from typing import Optional, Tuple, List
from pathlib import Path
import traceback

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
from torch.distributed import dist

def main():
    try:
        args = parse_args()
        
        # Load config
        config = Config.from_yaml(args.config)
        
        # Create directories from config
        if dist.is_initialized() and dist.get_rank() == 0:
            for path in [
                config.paths.checkpoints_dir,
                config.paths.logs_dir,
                config.paths.output_dir,
                config.data.cache_dir,
                config.data.text_cache_dir
            ]:
                os.makedirs(path, exist_ok=True)
        
        # Setup distributed training
        world_size, rank = setup_distributed(config)
        is_main_process = rank == 0
        
        # Setup accelerator
        accelerator = setup_accelerator(config)
        device = accelerator.device
        
        # Initialize wandb if main process
        if is_main_process:
            try:
                wandb.init(
                    project="sdxl-finetune",
                    config={
                        "batch_size": config.training.batch_size,
                        "grad_accum_steps": config.training.gradient_accumulation_steps,
                        "effective_batch": config.training.batch_size * config.training.gradient_accumulation_steps * world_size,
                        "learning_rate": config.training.learning_rate,
                        "num_epochs": config.training.num_epochs,
                        "world_size": world_size,
                        "rank": rank,
                        "backend": config.system.backend,
                        "timestep_bias_strategy": config.training.timestep_bias_strategy,
                        "snr_gamma": config.training.snr_gamma
                    }
                )
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {str(e)}")
                raise
        
        # Setup models
        unet, vae = setup_model(args, device, config)
        
        if config.system.use_fsdp and config.system.sync_batch_norm:
            unet = accelerator.sync_batchnorm(unet)
            if args.train_vae:
                vae = accelerator.sync_batchnorm(vae)
        
        # Initialize appropriate trainer
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
                    model=unet,
                    vae=vae,
                    accelerator=accelerator
                )
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}")
            raise
        
        # Validate directories and update config
        valid_dirs, total_images = validate_directories(config)
        config.data.image_dirs = valid_dirs
        
        if is_main_process:
            logger.info(f"Found {total_images} valid images across {len(valid_dirs)} directories")

        # Setup dataset
        try:
            dataset = NovelAIDataset(
                image_dirs=valid_dirs,
                text_embedder=TextEmbedder(
                    pretrained_model_name_or_path=config.model.pretrained_model_name,
                    device=device,
                    dtype=torch.bfloat16
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
            
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {str(e)}")
            raise
        
        # Prepare for distributed training
        trainer, dataset = accelerator.prepare(trainer, dataset)
        
        # Set optimal number of workers
        config.data.num_workers = get_optimal_cpu_threads()
        
        # Create dataloader
        try:
            train_dataloader = trainer.create_dataloader(
                dataset=dataset,
                batch_size=config.training.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
        except Exception as e:
            logger.error(f"Failed to create dataloader: {str(e)}")
            raise
        
        # Training loop
        start_epoch = trainer.current_epoch
        save_dir = config.paths.checkpoints_dir
        
        try:
            if is_main_process:
                logger.info(f"Starting {'VAE' if args.train_vae else 'UNet'} training from epoch {start_epoch + 1}")
            
            for epoch in range(start_epoch, config.training.num_epochs + 1):
                if is_main_process:
                    logger.info(f"\nStarting epoch {epoch}")
                
                if hasattr(train_dataloader.sampler, "set_epoch"):
                    train_dataloader.sampler.set_epoch(epoch)
                
                epoch_loss = trainer.train_epoch(
                    epoch=epoch,
                    train_dataloader=train_dataloader
                )
                
                if is_main_process:
                    logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
                    
                    # Save checkpoint
                    trainer.save_checkpoint(save_dir, epoch)
                    
                    # Log metrics
                    prefix = "vae_" if args.train_vae else "unet_"
                    wandb.log({
                        f"{prefix}epoch": epoch,
                        f"{prefix}loss": epoch_loss,
                    }, step=trainer.global_step)
                
                if world_size > 1:
                    torch.distributed.barrier()
                
        except KeyboardInterrupt:
            if is_main_process:
                logger.info("\nTraining interrupted. Saving final checkpoint...")
                trainer.save_checkpoint(save_dir, epoch)
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        finally:
            # Cleanup
            if is_main_process:
                wandb.finish()
            accelerator.end_training()
            
            if world_size > 1:
                torch.distributed.destroy_process_group()
                
            if is_main_process:
                logger.info("Training completed!")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()