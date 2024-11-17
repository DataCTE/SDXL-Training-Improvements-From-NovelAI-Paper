"""Main entry point for SDXL training."""

import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import torch
import wandb

from src.config.args import parse_args
from src.training.trainer import SDXLTrainer
from src.data.setup_dataset import create_train_dataloader, create_validation_dataloader
from src.utils.hub import create_model_card, save_model_card, push_to_hub
from src.utils.logging import (
    setup_logging,
    log_system_info,
    setup_wandb,
    cleanup_wandb,
    log_memory_stats,
    log_model_gradients
)
from src.models.model_loader import load_models

logger = logging.getLogger(__name__)


def main() -> None:
    """Main training function with improved organization and error handling."""
    # Parse arguments and setup logging
    config = parse_args()
    setup_logging(config.system.verbose)
    
    try:
        # Log system information and setup
        log_system_info()
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models first
        models = load_models(config)
        
        # Setup data loaders
        train_dataloader = create_train_dataloader(
            data_dir=config.data.data_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            batch_size=config.training.batch_size,
            cache_dir=config.data.cache_dir,
            all_ar=config.data.all_ar,
            use_tag_weighting=config.data.use_tag_weighting
        )
        
        val_dataloader = None
        if config.data.validation_dir:
            val_dataloader = create_validation_dataloader(
                data_dir=config.data.validation_dir,
                vae=models["vae"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                batch_size=config.training.batch_size
            )
        
        # Initialize trainer with all required components
        trainer = SDXLTrainer(
            config=config,
            models=models,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device
        )
        
        # Setup wandb if enabled
        wandb_run = None
        if config.logging.use_wandb:
            wandb_run = setup_wandb(config)
        
        # Create output directory
        output_dir = Path(config.model.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(config.training.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.training.num_epochs}")
            
            # Train for one epoch
            trainer.train_epoch(epoch)
            metrics = trainer.metrics_manager.get_metrics()
            
            # Log metrics
            if config.logging.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()})
            
            # Run validation if available
            if val_dataloader:
                trainer.validate(epoch)
                val_metrics = trainer.metrics_manager.get_validation_metrics()
                if config.logging.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
            
            # Save checkpoint
            if config.logging.save_checkpoints and (epoch + 1) % config.logging.save_epochs == 0:
                trainer.save_checkpoint(
                    str(output_dir),
                    epoch
                )
            
            # Log memory stats and model gradients
            log_memory_stats(step=epoch)
            log_model_gradients(trainer.models["unet"], step=epoch)
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_checkpoint(str(final_model_path), config.training.num_epochs - 1)
        
        # Create and save model card
        model_card = create_model_card(config, trainer.metrics_manager.get_training_history())
        save_model_card(output_dir / "README.md", model_card)
        
        # Push to hub if configured
        if config.logging.push_to_hub:
            push_to_hub(
                output_dir,
                trainer.models["unet"],
                trainer.components.get("ema_model"),
                trainer.models["vae"],
                config
            )
        
        logger.info("Training completed successfully!")
        
    except (torch.cuda.CudaError, torch.cuda.OutOfMemoryError) as e:
        logger.error(f"CUDA error occurred: {str(e)}", exc_info=True)
        sys.exit(3)
    except (TypeError, AttributeError) as e:
        logger.error(f"Type or attribute error in training pipeline: {str(e)}", exc_info=True)
        sys.exit(5)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import required module: {str(e)}", exc_info=True)
        sys.exit(4)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    finally:
        # Cleanup
        if wandb_run is not None:
            cleanup_wandb(wandb_run)


if __name__ == "__main__":
    main()