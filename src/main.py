"""Main entry point for SDXL training with improvements from NovelAI paper."""

import logging
import sys
from pathlib import Path

from config.args import parse_args
from training.wrappers import train_sdxl, train_vae
from utils.logging import setup_logging
from models.model_loader import load_models
from data.image_processing.validation import ValidationConfig
from utils.progress import ProgressTracker
import traceback

logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        logger.info("Initializing training pipeline...")
        
        # Parse arguments
        logger.info("Parsing arguments...")
        config = parse_args()
        
        # Setup logging
        logger.info("Setting up logging...")
        setup_logging(
            log_dir=Path(config.output_dir) / "logs",
            log_level=logging.INFO
        )
        
        # Create validation config
        logger.info("Creating validation config...")
        validation_config = ValidationConfig(
            min_size=512,
            max_size=2048,
            min_aspect=0.4,
            max_aspect=2.5,
            check_content=True,
            device=config.device
        )
        
        # Load models
        logger.info("Loading models...")
        models = load_models(config)
        
        # Train VAE if enabled
        if config.vae_args.enable_vae_finetuning:
            logger.info("Starting VAE finetuning...")
            vae_trainer = train_vae(
                train_data_dir=config.train_data_dir,
                output_dir=config.output_dir,
                config=config.vae_args,
                validation_config=validation_config,
                wandb_run=config.wandb.use_wandb
            )
            vae_trainer.train()
            models["vae"] = vae_trainer.vae
        
        # Train SDXL
        logger.info("Starting SDXL training...")
        trainer = train_sdxl(
            train_data_dir=config.train_data_dir,
            output_dir=config.output_dir,
            pretrained_model_path=config.pretrained_model_path,
            models=models,
            validation_config=validation_config,
            config=config,
            wandb_run=config.wandb.use_wandb
        )
        
        trainer.train(save_dir=config.output_dir)
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_checkpoint(config.output_dir, config.num_epochs)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()