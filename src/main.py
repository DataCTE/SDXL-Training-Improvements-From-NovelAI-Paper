"""Main entry point for SDXL training with improvements from NovelAI paper."""

import logging
import sys
from pathlib import Path

from config.args import parse_args
from training.wrappers import train_sdxl, train_vae
from utils.logging import setup_logging
from models.model_loader import load_models, save_diffusers_format, save_checkpoint
from data.image_processing.validation import ValidationConfig
from models.SDXL.pipeline import StableDiffusionXLPipeline
import traceback
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
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create validation config
        validation_config = ValidationConfig(
            min_size=config.validation_image_width,  # Use config values
            max_size=config.max_resolution,
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
                output_dir=str(output_dir / "vae"),
                config=config.vae_args,
                validation_config=validation_config
            )
            # Run VAE training
            vae_trainer.train()
            models["vae"] = vae_trainer.vae
            
            # Save VAE checkpoint
            vae_save_path = output_dir / "vae" / "final_vae.safetensors"
            save_checkpoint(
                pipeline=StableDiffusionXLPipeline(
                    vae=vae_trainer.vae,
                    text_encoder=models["text_encoder"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer=models["tokenizer"],
                    tokenizer_2=models["tokenizer_2"],
                    unet=models["unet"],
                    scheduler=models["scheduler"]
                ),
                checkpoint_path=str(vae_save_path),
                save_vae=True,
                use_safetensors=True
            )
        
        # Train SDXL
        logger.info("Starting SDXL training...")
        trainer = train_sdxl(
            train_data_dir=config.train_data_dir,
            output_dir=str(output_dir),
            pretrained_model_path=config.pretrained_model_path,
            models=models,
            validation_config=validation_config,
            config=config
        )
        
        # Run SDXL training
        trainer.train()
        
        # Create final pipeline
        final_pipeline = StableDiffusionXLPipeline(
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            unet=models["unet"],
            scheduler=models["scheduler"]
        )
        
        # Save final model in both formats
        final_checkpoint_path = output_dir / f"checkpoint_epoch_{config.num_epochs}.safetensors"
        final_diffusers_path = output_dir / f"epoch_{config.num_epochs}"
        
        # Save checkpoint
        save_checkpoint(
            pipeline=final_pipeline,
            checkpoint_path=str(final_checkpoint_path),
            save_vae=True,
            use_safetensors=True
        )
        
        # Save diffusers format
        save_diffusers_format(
            pipeline=final_pipeline,
            output_dir=str(final_diffusers_path),
            save_vae=True,
            use_safetensors=True
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")
        logger.info(f"Final diffusers format saved to: {final_diffusers_path}")
        
    except Exception as e:
        logger.error("Training failed with error: %s", str(e))
        logger.error("Full traceback:\n%s", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()