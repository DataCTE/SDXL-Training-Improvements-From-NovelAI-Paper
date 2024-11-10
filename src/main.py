import argparse
import logging
import os
import sys
import wandb
import torch
import traceback
from pathlib import Path


from training.trainer import train
from training.utils import setup_torch_backends, cleanup
from models.setup import setup_models
from training.setup import setup_training
from training.utils import load_checkpoint, save_checkpoint, save_final_outputs
from utils.model_card import create_model_card, save_model_card, push_to_hub
from utils.logging import (
    setup_logging,
    log_system_info,
    log_training_setup,
    log_gpu_memory,
    setup_wandb,
    cleanup_wandb
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a Stable Diffusion XL model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of steps for scheduler warm up")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    
    # Model optimization
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--compile_mode", 
                       type=str, 
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       default='default')
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory at the expense of speed")
    
    # EMA and VAE settings
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--use_ema", action="store_true",
                       help="Enable Exponential Moving Average during training")
    parser.add_argument("--finetune_vae", action="store_true")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6)
    parser.add_argument("--vae_train_freq", type=int, default=10)
    
    # Tag weighting
    parser.add_argument("--min_tag_weight", type=float, default=0.1)
    parser.add_argument("--max_tag_weight", type=float, default=3.0)
    
    # Checkpointing
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training")
    parser.add_argument("--wandb_run_name", type=str)
    
    # Validation
    parser.add_argument("--validation_frequency", type=int, default=1)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--validation_prompts", type=str, nargs="+",
                       default=["a detailed portrait of a girl",
                               "completely black",
                               "a red ball on top of a blue cube, both infront of a green triangle"])
    
    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_private", action="store_true")
    
    return parser.parse_args()

def main(args):
    """Main training pipeline"""
    wandb_run = None
    try:
        # Setup logging
        setup_logging(log_dir=args.output_dir)
        log_system_info()
        
        # Initialize W&B
        wandb_run = setup_wandb(args)
        
        # Setup environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        logger.info(f"Using device: {device}, dtype: {dtype}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Initial CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        # === Initialize Components ===
        models = setup_models(args, device, dtype)
        train_components = setup_training(args, models, device, dtype)
        
        # Load checkpoint if resuming
        start_epoch = 0
        if args.resume_from_checkpoint:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
            training_state = load_checkpoint(args.resume_from_checkpoint, models, train_components)
            start_epoch = training_state["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
            
        # Log training setup
        log_training_setup(args, models, train_components)
        
        # === Training ===
        logger.info("\nStarting training...")
        training_history = train(args, models, train_components, device, dtype)
        
        # === Save Outputs ===
        logger.info("\nSaving final outputs...")
        save_final_outputs(args, models, training_history, train_components)
        
        # Create and save model card
        model_card = create_model_card(args, training_history)
        if model_card:
            save_model_card(model_card, args.output_dir)
            
            # Push to Hub if requested
            if args.push_to_hub:
                logger.info("\nPushing to HuggingFace Hub...")
                try:
                    push_to_hub(
                        args.hub_model_id,
                        args.output_dir,
                        args.hub_private,
                        model_card
                    )
                except Exception as e:
                    logger.error(f"Failed to push to Hub: {str(e)}")

        logger.info("\n=== Training Pipeline Completed Successfully ===")
        return True

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        logger.info(f"Saving emergency checkpoint to {emergency_dir}")
        try:
            save_checkpoint(models, train_components, args, -1, training_history, emergency_dir)
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        return False

    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    finally:
        cleanup_wandb(wandb_run)

if __name__ == "__main__":
    setup_torch_backends()
    args = parse_args()
    success = main(args)
    sys.exit(0 if success else 1) 