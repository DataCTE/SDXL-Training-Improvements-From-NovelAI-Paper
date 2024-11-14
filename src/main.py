import argparse
import logging
import os
import sys
import wandb
import torch
import traceback
from pathlib import Path


from training.trainer import train
from utils.checkpoint import load_checkpoint, save_checkpoint, save_final_outputs
from utils.model_card import create_model_card, save_model_card, push_to_hub
from utils.logging import (
    setup_logging,
    log_system_info,
    log_training_setup,
    setup_wandb,
    cleanup_wandb,
    log_memory_stats,
    log_model_gradients
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
    
    # EMA parameters
    parser.add_argument("--use_ema", action="store_true", default=True,
                       help="Use EMA model for training")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                       help="EMA decay rate")
    parser.add_argument("--ema_update_after_step", type=int, default=100,
                       help="Start EMA after this many steps")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0,
                       help="Inverse multiplicative factor of EMA warmup length")
    parser.add_argument("--ema_power", type=float, default=0.6667,
                       help="Power for decay rate schedule (default: 2/3)")
    parser.add_argument("--ema_min_decay", type=float, default=0.0,
                       help="Minimum EMA decay rate")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999,
                       help="Maximum EMA decay rate")
    parser.add_argument("--ema_update_every", type=int, default=1,
                       help="Update EMA every N steps")
    parser.add_argument("--use_ema_warmup", action="store_true", default=True,
                       help="Use EMA warmup")
    parser.add_argument("--ema_grad_scale_factor", type=float, default=0.5,
                       help="Factor for gradient-based update weighting")
    
    # V-prediction parameters
    parser.add_argument("--min_snr_gamma", type=float, default=5.0,
                      help="Minimum SNR value for loss weighting (default: 5.0)")
    parser.add_argument("--sigma_data", type=float, default=1.0,
                      help="Data standard deviation for v-prediction (default: 1.0)")
    parser.add_argument("--sigma_min", type=float, default=0.029,
                      help="Minimum sigma value for ZTSNR schedule")
    parser.add_argument("--rescale_multiplier", type=float, default=0.7,
                      help="Multiplier for CFG rescaling")
    
    # Training mode flags
    parser.add_argument("--zsnr", action="store_true", default=True,
                      help="Enable Zero Terminal SNR training")
    parser.add_argument("--v_prediction", action="store_true", default=True,
                      help="Enable v-prediction parameterization")
    parser.add_argument("--resolution_scaling", action="store_true", default=True,
                      help="Enable resolution-dependent sigma scaling")
    parser.add_argument("--rescale_cfg", action="store_true", default=True,
                      help="Enable CFG rescaling")
    parser.add_argument("--scale_method", type=str, default="karras",
                      choices=["karras", "simple"],
                      help="Method for CFG rescaling (karras or simple)")
    
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
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training",
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb_watch", action="store_true",
                       help="Enable W&B model parameter and gradient logging")
    parser.add_argument("--wandb_log_freq", type=int, default=10,
                       help="Log metrics every N steps")
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
    
    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_private", action="store_true")
    
    # No caching latents
    parser.add_argument('--no_caching_latents', action='store_true',
                       help='Disable caching of VAE latents')
    
    # Add to argument parser
    parser.add_argument('--all_ar', action='store_true',
                       help='Accept all aspect ratios without resizing')
    
    # Add num_workers argument
    parser.add_argument(
        '--num_workers',
        type=int,
        default=min(8, os.cpu_count() or 1),
        help='Number of workers for data loading'
    )
    
    # Performance and Training Settings
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Enable mixed precision training')
    return parser.parse_args()

def main(args):
    """Main training pipeline"""
    wandb_run = None
    try:
        # Set up logging
        setup_logging(os.path.join("logs", args.wandb_run_name) if args.wandb_run_name else "logs")
        log_system_info()
        
        # Set up Weights & Biases
        wandb_run = setup_wandb(args)
        
        # Log training setup and initial system state
        log_training_setup(args)
        if wandb_run and args.wandb_watch:
            wandb_run.watch(models.unet, log="all", log_freq=args.wandb_log_freq)
        
        # Load model and log initial memory stats
        models, train_components, training_history = load_checkpoint(args)
        if wandb_run:
            log_memory_stats(step=0)
        
        # Train model
        global_step = 0
        for epoch in range(args.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
            
            # Train for one epoch
            epoch_metrics = train(args, models, train_components)
            
            if wandb_run:
                # Log epoch metrics
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/loss": epoch_metrics["loss"],
                    "epoch/learning_rate": epoch_metrics["learning_rate"],
                    **{f"epoch/{k}": v for k, v in epoch_metrics.items() 
                       if k not in ["loss", "learning_rate"]}
                }, step=global_step)
                
                # Log model gradients
                log_model_gradients(models.unet, step=global_step)
                
                # Log memory stats
                log_memory_stats(step=global_step)
            
            global_step += len(train_components["train_dataloader"])

        # Save final outputs
        save_final_outputs(args, models, training_history, train_components)

        # Push to HuggingFace Hub
        if args.push_to_hub:
            push_to_hub(args, models)

        # Save model card
        save_model_card(args, models)
        
        # Log final metrics if using W&B
        if wandb_run:
            wandb.log({
                "training/completed": True,
                "training/total_steps": global_step,
                "training/final_loss": epoch_metrics["loss"],
                "training/final_lr": epoch_metrics["learning_rate"]
            })
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        if wandb_run:
            wandb.log({"training/failed": True, "training/error": str(e)})
        return False
        
    finally:
        if wandb_run:
            cleanup_wandb(wandb_run)

if __name__ == "__main__":
    try:
        # Try to set the start method only if it hasn't been set yet
        if not torch.multiprocessing.get_start_method(allow_none=True):
            torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        # If already set, just log and continue
        logger.info("Multiprocessing start method already set")

    args = parse_args()
    success = main(args)
    sys.exit(0 if success else 1)