import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
from pathlib import Path
from src.data.dataset import NovelAIDataset
from src.training.trainer import NovelAIDiffusionV3Trainer
from src.utils.model import setup_model
from src.config.config import Config
from src.utils.setup import setup_distributed
import gc

logger = logging.getLogger(__name__)

def train(config_path: str):
    """Main training function."""
    try:
        # Load config
        config = Config.from_yaml(config_path)
        
        # Setup distributed training if enabled
        if config.system.distributed:
            setup_distributed()
            device = torch.device(f"cuda:{dist.get_rank()}")
            is_main_process = dist.get_rank() == 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            is_main_process = True
        
        # Initialize wandb if enabled
        if config.training.use_wandb and is_main_process:
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name,
                config=config.to_dict()
            )
        
        # Setup models
        unet, vae = setup_model(None, device, config)
        
        # Convert models to DDP if using distributed training
        if config.system.distributed:
            # Convert batch norm to sync batch norm for distributed training
            unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
            vae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vae)
            
            # Wrap models in DDP
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
        
        # Create dataset
        dataset = NovelAIDataset(
            config=config,
            device=device
        )
        
        # Train
        logger.info("Starting training...")
        for epoch in range(config.training.num_epochs):
            if config.system.distributed:
                dataset.sampler.set_epoch(epoch)
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
        raise
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    train(sys.argv[1])