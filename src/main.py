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
from training.trainer import get_cosine_schedule_with_warmup
from data.dataset import create_dataloader


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
        
        # Set device and dtype
        device = torch.device("cuda")
        dtype = torch.float16 if args.mixed_precision else torch.float32
        
        # Load model and initialize components
        models, optimizer_state, training_history = load_checkpoint(args)
        
        # Move models to device
        for model_name, model in models.items():
            if isinstance(model, torch.nn.Module):
                model.to(device=device, dtype=dtype)
                if args.gradient_checkpointing:
                    if hasattr(model, 'enable_gradient_checkpointing'):
                        model.enable_gradient_checkpointing()
                    elif hasattr(model, 'gradient_checkpointing_enable'):
                        model.gradient_checkpointing_enable()
        
        # Setup data loader first since we need it for lr_scheduler
        train_dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            all_ar=args.all_ar,
            cache_dir=args.cache_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"]
        )
        
        # Setup optimizer
        if args.use_adafactor:
            from transformers import Adafactor
            optimizer = Adafactor(
                models["unet"].parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                scale_parameter=True,
                relative_step=False
            )
        else:
            optimizer = torch.optim.AdamW(
                models["unet"].parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay
            )
        
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        
        # Setup learning rate scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.num_epochs * len(train_dataloader)
        )
        
        # Setup EMA
        if args.use_ema:
            from training.ema import EMAModel
            ema = EMAModel(
                models["unet"],
                model_path=args.model_path,
                decay=args.ema_decay,
                update_after_step=args.ema_update_after_step,
                update_every=args.ema_update_every,
                power=args.ema_power,
                min_decay=args.ema_min_decay,
                max_decay=args.ema_max_decay,
                use_ema_warmup=args.use_ema_warmup
            )
        else:
            ema = None
        
        # Setup tag weighter if enabled
        if args.use_tag_weighting:
            from data.tag_weighter import TagBasedLossWeighter
            tag_weighter = TagBasedLossWeighter(
                min_weight=args.min_tag_weight,
                max_weight=args.max_tag_weight
            )
        else:
            tag_weighter = None
        
        # Setup VAE finetuner if enabled
        if args.finetune_vae:
            from training.vae_finetuner import VAEFineTuner
            vae_finetuner = VAEFineTuner(
                models["vae"],
                learning_rate=args.vae_learning_rate,
                min_snr_gamma=args.min_snr_gamma,
                use_8bit_adam=args.use_8bit_adam,
                gradient_checkpointing=args.gradient_checkpointing,
                mixed_precision="fp16" if args.mixed_precision else "no"
            )
        else:
            vae_finetuner = None
        
        # Create components dictionary
        train_components = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "ema": ema,
            "train_dataloader": train_dataloader,
            "tag_weighter": tag_weighter,
            "vae_finetuner": vae_finetuner
        }
        
        # Log training setup and initial system state
        log_training_setup(args, models, train_components)
        
        if wandb_run and args.wandb_watch:
            wandb_run.watch(models["unet"], log="all", log_freq=args.wandb_log_freq)
        
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
                log_model_gradients(models["unet"], step=global_step)
                
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