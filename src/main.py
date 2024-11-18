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
        with ProgressTracker("Initializing Training Pipeline", total=5) as progress:
            # Parse arguments
            progress.update(1, {"status": "Parsing arguments"})
            config = parse_args()
            
            # Setup logging
            progress.update(1, {"status": "Setting up logging"})
            setup_logging(
                log_dir=Path(config.output_dir) / "logs",
                log_level=logging.INFO
            )
            
            # Create validation config
            progress.update(1, {"status": "Creating validation config"})
            validation_config = ValidationConfig(
                min_size=512,
                max_size=2048,
                min_aspect=0.4,
                max_aspect=2.5,
                check_content=True,
                device=config.device
            )
            
            # Load models
            progress.update(1, {"status": "Loading models"})
            models = load_models(config)
            
            # Initialize training
            progress.update(1, {"status": "Initializing training"})
            
            # Train VAE if enabled
            if config.vae_args.enable_vae_finetuning:
                with ProgressTracker("VAE Finetuning", total=2) as vae_progress:
                    vae_progress.update(1, {"status": "Setting up VAE trainer"})
                    vae_trainer = train_vae(
                        train_data_dir=config.train_data_dir,
                        output_dir=config.output_dir,
                        config=config.vae_args,
                        validation_config=validation_config,
                        wandb_run=config.wandb.use_wandb
                    )
                    
                    vae_progress.update(1, {"status": "Running VAE training"})
                    vae_trainer.train()
                    models["vae"] = vae_trainer.vae
            
            # Train SDXL
            with ProgressTracker("SDXL Training", total=2) as sdxl_progress:
                sdxl_progress.update(1, {"status": "Setting up SDXL trainer"})
                trainer = train_sdxl(
                    train_data_dir=config.train_data_dir,
                    output_dir=config.output_dir,
                    pretrained_model_path=config.pretrained_model_path,
                    models=models,
                    validation_config=validation_config,
                    config=config,
                    wandb_run=config.wandb.use_wandb
                )
                
                sdxl_progress.update(1, {"status": "Running SDXL training"})
                trainer.train(save_dir=config.output_dir)
            
            # Save final model
            with ProgressTracker("Saving Final Model", total=1) as save_progress:
                trainer.save_checkpoint(config.output_dir, config.num_epochs)
                save_progress.update(1, {"status": "Model saved successfully"})
                
            logger.info("Training completed successfully!")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()