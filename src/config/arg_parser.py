import argparse
import os
import logging
import sys
import traceback

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with error handling."""
    try:
        parser = argparse.ArgumentParser(description="Train SDXL model")
        
        # Required arguments
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to configuration YAML file"
        )
        
        # Optional arguments
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="Path to checkpoint directory to resume training from"
        )
        
        parser.add_argument(
            "--train_vae",
            action="store_true",
            help="Train VAE instead of UNet"
        )
        
        parser.add_argument(
            "--vae_path",
            type=str,
            default=None,
            help="Path to pretrained VAE model"
        )
        
        parser.add_argument(
            "--unet_path",
            type=str,
            default=None,
            help="Path to pretrained UNet model"
        )
        
        # Parse and validate arguments
        args = parser.parse_args()
        
        # Validate config path
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
        # Validate checkpoint path if provided
        if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.resume_from_checkpoint}")
            
        # Validate VAE path if provided
        if args.vae_path and not os.path.exists(args.vae_path):
            raise FileNotFoundError(f"VAE model file not found: {args.vae_path}")
            
        # Validate UNet path if provided
        if args.unet_path and not os.path.exists(args.unet_path):
            raise FileNotFoundError(f"UNet model file not found: {args.unet_path}")
            
        # Log arguments if they're provided
        logger.info("Command line arguments:")
        logger.info(f"  Config: {args.config}")
        if args.resume_from_checkpoint:
            logger.info(f"  Resume from: {args.resume_from_checkpoint}")
        if args.train_vae:
            logger.info("  Training VAE")
            if args.vae_path:
                logger.info(f"  VAE path: {args.vae_path}")
        else:
            logger.info("  Training UNet")
            if args.unet_path:
                logger.info(f"  UNet path: {args.unet_path}")
        
        return args
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except argparse.ArgumentError as e:
        logger.error(f"Argument parsing error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1) 