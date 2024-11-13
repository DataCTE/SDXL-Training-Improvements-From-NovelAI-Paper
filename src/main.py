import argparse
import logging
import os
import sys
import wandb
import torch
import traceback
from pathlib import Path


from training.trainer import train
from utils.setup import (
    setup_torch_backends,
    setup_models,
    setup_training
)
from utils.checkpoint import load_checkpoint, save_checkpoint, save_final_outputs
from utils.model_card import create_model_card, save_model_card, push_to_hub
from utils.logging import (
    setup_logging,
    log_system_info,
    log_training_setup,
    setup_wandb,
    cleanup_wandb
)

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler()
    ]
)

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
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps for learning rate scheduler")
    
    # AdamW optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="The beta1 parameter for the Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="The beta2 parameter for the Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                       help="Epsilon value for the Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                       help="Weight decay for the Adam optimizer")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                       help="Number of inference steps (default: 28 as per paper)")
    
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
    parser.add_argument("--use_tag_weighting", action="store_true", default=True,
                       help="Enable tag-based loss weighting during training")
    
    # Checkpointing
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training")
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Number of steps between logging updates")
    parser.add_argument("--save_epochs", type=int, default=1,
                       help="Number of epochs between saving checkpoints")
    
    # Validation
    parser.add_argument("--validation_frequency", type=int, default=1,
                       help="Number of epochs between validation runs")
    parser.add_argument("--validation_steps", type=int, default=0,
                       help="Number of steps between validation runs (0 to disable)")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--validation_prompts", type=str, nargs="+",
                       default=["a detailed portrait of a girl",
                               "completely black",
                               "a red ball on top of a blue cube"])
    
    # ZTSNR + V-Prediction settings
    parser.add_argument("--zsnr", action="store_true", default=True)
    parser.add_argument("--v_prediction", action="store_true", default=True)
    parser.add_argument("--sigma_min", type=float, default=0.029)
    parser.add_argument("--sigma_data", type=float, default=1.0)
    parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    parser.add_argument("--resolution_scaling", action="store_true", default=True)
    
    # CFG Rescale settings
    parser.add_argument("--rescale_cfg", action="store_true", default=True)
    parser.add_argument("--scale_method", type=str, default="karras")
    parser.add_argument("--rescale_multiplier", type=float, default=0.7)
    
    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_private", action="store_true")
    
    # No caching latents
    parser.add_argument('--no_caching_latents', action='store_true',
                       help='Disable caching of VAE latents')
    
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

def setup_wandb(args):
    """Initialize Weights & Biases logging"""
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            group="training",
        )
        
        # Define metric groupings
        wandb.define_metric("loss/*", summary="min")
        wandb.define_metric("model/*", summary="last")
        wandb.define_metric("noise/*", summary="mean")
        wandb.define_metric("lr/*", summary="last")
        
        # Create custom panels
        wandb.define_metric("loss/current", step_metric="step")
        wandb.define_metric("loss/average", step_metric="step")
        wandb.define_metric("loss/running", step_metric="step")
        wandb.define_metric("lr/textencoder", step_metric="step")
        wandb.define_metric("lr/unet", step_metric="step")
        
        # Add epoch-specific metric groupings
        wandb.define_metric("epoch", summary="max")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("epoch/progress", summary="last")
        wandb.define_metric("epoch/average_loss", summary="min")
        
        return wandb.run
    return None

if __name__ == "__main__":
    # Add multiprocessing start method configuration
    torch.multiprocessing.set_start_method('spawn')
    
    setup_torch_backends()
    args = parse_args()
    success = main(args)
    sys.exit(0 if success else 1) 