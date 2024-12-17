import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from src.data.dataset import NovelAIDataset
from src.data.sampler import AspectBatchSampler
from src.training.trainer import NovelAIDiffusionV3Trainer
from src.utils.model import setup_model
from src.config.config import Config
from src.utils.setup import setup_distributed, setup_memory_optimizations, verify_memory_optimizations
from src.utils.metrics import log_system_info
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
        if config.system.distributed:
            setup_distributed()
            device = torch.device(f"cuda:{dist.get_rank()}")
            is_main_process = dist.get_rank() == 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            is_main_process = True

        # Log system information
        if is_main_process:
            log_system_info()
        
        # Initialize wandb if enabled
        if config.training.use_wandb and is_main_process:
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name,
                config=config.to_dict()
            )
        
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
        if config.system.distributed:
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
        
        # Create dataset and dataloader
        dataset = NovelAIDataset(
            config=config,
            device=device
        )
        
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
            if config.system.distributed:
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