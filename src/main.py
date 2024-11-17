"""Main entry point for SDXL training with improvements from NovelAI paper."""

import logging
import sys
from pathlib import Path

from config.args import parse_args
from training.wrappers import train_sdxl, train_vae
from utils.logging import setup_logging
from models.model_loader import load_models

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    try:
        # Parse arguments
        config = parse_args()
        
        # Setup logging
        setup_logging(
            log_dir=Path(config.output_dir) / "logs",
            log_level=logging.INFO
        )
        
        # Load models
        logger.info("Loading models...")
        models = load_models(config)
        
        # Train VAE if enabled
        if config.vae_args.enable_vae_finetuning:
            logger.info("Starting VAE finetuning...")
            vae_trainer = train_vae(
                train_data_dir=config.data_dir,
                output_dir=config.output_dir,
                pretrained_vae_path=config.vae_args.vae_path,
                learning_rate=config.vae_args.learning_rate,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                mixed_precision=config.mixed_precision,
                use_8bit_adam=config.optimizer.use_8bit_adam,
                gradient_checkpointing=config.gradient_checkpointing,
                use_channel_scaling=config.vae_args.use_channel_scaling,
            )
            models["vae"] = vae_trainer.vae
        
        # Train SDXL
        logger.info("Starting SDXL training...")
        trainer = train_sdxl(
            train_data_dir=config.data_dir,
            output_dir=config.output_dir,
            pretrained_model_path=config.model_path,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            learning_rate=config.optimizer.learning_rate,
            mixed_precision=config.mixed_precision,
            use_8bit_adam=config.optimizer.use_8bit_adam,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            save_epochs=config.save_epochs,
            models=models
        )
        
        # Save final model
        trainer.save_checkpoint(config.output_dir, config.num_epochs)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        import traceback
        logger.error("Training failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()