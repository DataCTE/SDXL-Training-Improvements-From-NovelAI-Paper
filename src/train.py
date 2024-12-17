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
from src.utils.metrics import setup_logging, log_system_info
from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter, TagWeightingConfig
from src.data.validation import validate_directories
import gc
import traceback
from src.config.arg_parser import parse_args

logger = logging.getLogger(__name__)


def train(config_path: str):
    """Main training function with improved setup and error handling."""
    try:
        # Parse arguments first
        args = parse_args()
        
        # Load config
        config = Config.from_yaml(args.config)
        
        # Setup distributed training if enabled
        if config.system.distributed_training:
            is_distributed = setup_distributed()
            if is_distributed:
                device = torch.device(f"cuda:{dist.get_rank()}")
                is_main_process = dist.get_rank() == 0
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                is_main_process = True
                # Update config to reflect non-distributed mode
                config.system.distributed_training = False
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            is_main_process = True

        # Configure logging
        setup_logging(config.paths.logs_dir, is_main_process)

        # Log system information
        if is_main_process:
            log_system_info()
        
        # Initialize wandb if enabled and configured
        if hasattr(config.training, 'use_wandb') and config.training.use_wandb and is_main_process:
            if not hasattr(config.training, 'wandb_project'):
                logger.warning("Wandb enabled but project name not configured. Disabling wandb.")
            else:
                wandb.init(
                    project=config.training.wandb_project,
                    name=config.training.wandb_run_name,
                    config=config.to_dict()
                )

        # Validate directories
        valid_dirs, total_images = validate_directories(config)
        if is_main_process:
            logger.info(f"Found {total_images} valid images across {len(valid_dirs)} directories")
        config.data.image_dirs = valid_dirs
        
        # Setup models with verification using args
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
            logger.warning("Some memory optimizations failed to apply")
        
        # Convert models to DDP if using distributed training
        if config.system.distributed_training:
            unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
            vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            
            unet = DistributedDataParallel(
                unet,
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                find_unused_parameters=False
            )
            vae = DistributedDataParallel(
                vae,
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                find_unused_parameters=False
            )
        
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
            min_size=config.data.min_size,
            max_dim=config.data.max_dim,
            bucket_step=config.data.bucket_step,
            min_bucket_size=config.data.min_bucket_size,
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
            image_dirs=valid_dirs,
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
            num_workers=config.system.num_workers,
            pin_memory=True,
            collate_fn=trainer.collate_fn
        )
        
        # Assign dataloader to trainer
        trainer.dataloader = dataloader
        
        # Train
        logger.info("Starting training...")
        for epoch in range(config.training.num_epochs):
            if config.system.distributed_training:
                sampler.set_epoch(epoch)
            
            trainer.train_epoch(epoch)
            
            # Save checkpoint on main process
            if is_main_process:
                trainer.save_checkpoint(config.paths.checkpoints_dir, epoch)
        
        logger.info("Training complete!")
        
        # Cleanup
        if config.training.use_wandb and is_main_process:
            wandb.finish()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    from src.config.arg_parser import parse_args
    args = parse_args()
    train(args.config)