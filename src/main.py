import argparse
import logging
import os
import sys
import wandb
import torch
import traceback
from pathlib import Path
from training.trainer import (
    setup_optimizer,
    setup_vae_finetuner,
    setup_ema,
    setup_validator,
    setup_tag_weighter,
    AverageMeter
)

# Optional: If you need the logging utility functions
from training.trainer import (
    _log_optimizer_config,
    _log_vae_config,
    _log_ema_config
)

import time
from collections import defaultdict
from tqdm import tqdm
import torch.cuda
import math
import re
from typing import Dict
# Required imports for the components
from torch.cuda.amp import autocast, GradScaler
from transformers import Adafactor
from utils.checkpoint import load_checkpoint, save_checkpoint, save_final_outputs
from utils.hub import create_model_card, save_model_card, push_to_hub
from utils.logging import (
    setup_logging,
    log_system_info,
    log_training_setup,
    setup_wandb,
    cleanup_wandb,
    log_memory_stats,
    log_model_gradients,

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
    
    # Training configuration
    parser.add_argument("--training_mode", type=str, default="v_prediction",
                      choices=["v_prediction", "epsilon"],
                      help="Training parameterization mode")
    parser.add_argument("--use_ztsnr", action="store_true",
                      help="Enable Zero Terminal SNR training")
    parser.add_argument("--rescale_cfg", action="store_true",
                      help="Enable CFG rescaling")
    parser.add_argument("--rescale_multiplier", type=float, default=0.7,
                      help="Multiplier for CFG rescaling")
    parser.add_argument("--resolution_scaling", action="store_true", default=True,
                      help="Enable resolution-based scaling")
    parser.add_argument("--min_snr_gamma", type=float, default=5.0,
                      help="Minimum SNR value for loss weighting")
    parser.add_argument("--sigma_data", type=float, default=1.0,
                      help="Data standard deviation")
    parser.add_argument("--sigma_min", type=float, default=0.029,
                      help="Minimum sigma value")
    parser.add_argument("--sigma_max", type=float, default=160.0,
                      help="Maximum sigma value")
    parser.add_argument("--scale_method", type=str, default="karras",
                      choices=["karras", "simple"],
                      help="Method for resolution and CFG scaling")
    parser.add_argument("--scale_factor", type=float, default=0.7,
                      help="Global scaling factor")
    
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
    parser.add_argument("--no_caching", action="store_true", help="Disable latent caching")
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
    
    # VAE finetuning parameters
    parser.add_argument("--finetune_vae", action="store_true",
                       help="Enable VAE finetuning")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6,
                       help="Learning rate for VAE finetuning")
    parser.add_argument("--vae_train_freq", type=int, default=10,
                       help="How often to update VAE during training")
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Use 8-bit Adam optimizer for memory efficiency")
    parser.add_argument("--adaptive_loss_scale", action="store_true",
                       help="Use adaptive loss scaling for VAE")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                       help="Weight for KL divergence loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.0,
                       help="Weight for perceptual loss")
    
    # Tag weighting
    parser.add_argument("--min_tag_weight", type=float, default=0.1)
    parser.add_argument("--max_tag_weight", type=float, default=3.0)
    parser.add_argument("--use_tag_weighting", action="store_true", default=True,
                       help="Enable tag-based loss weighting during training")
    
    # Checkpointing
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # Logging and monitoring
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose output during training")
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

def initialize_training_components(args, device, dtype, models):
    """Initialize all training components with proper error handling"""
    components = {}
    
    try:
        # Setup optimizer
        components["optimizer"] = setup_optimizer(args, models)
        
        # Setup data loader
        components["train_dataloader"] = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_caching_latents=args.no_caching,
            all_ar=args.all_ar,
            cache_dir=args.cache_dir,
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"]
        )
        
        # Setup learning rate scheduler
        components["lr_scheduler"] = get_cosine_schedule_with_warmup(
            optimizer=components["optimizer"],
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.num_epochs * len(components["train_dataloader"])
        )
        
        # Optional components
        components.update({
            "ema": setup_ema(args, models, device) if args.use_ema else None,
            "tag_weighter": setup_tag_weighter(args) if args.use_tag_weighting else None,
            "vae_finetuner": setup_vae_finetuner(args, models) if args.finetune_vae else None
        })
        
        # Training configuration
        components["training_config"] = {
            "mode": args.training_mode,
            "min_snr_gamma": args.min_snr_gamma,
            "sigma_data": args.sigma_data,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "scale_method": args.scale_method,
            "scale_factor": args.scale_factor
        }
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize training components: {str(e)}")
        raise

def train(args, models, components, device, dtype) -> Dict[str, float]:
    """Execute training steps with proper error handling and logging."""
    metrics = defaultdict(lambda: AverageMeter(name="default"))
    models["unet"].train()
    
    start_time = time.time()
    data_time = AverageMeter("data_time")
    batch_time = AverageMeter("batch_time")
    
    progress_bar = tqdm(
        components["train_dataloader"],
        desc=f"Training",
        dynamic_ncols=True
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            data_time.update(time.time() - start_time)
            batch_metrics = train_step(
                args=args,
                models=models,
                components=components,
                batch=batch,
                batch_idx=batch_idx,  # Added batch_idx here
                device=device,
                dtype=dtype
            )
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k].update(v)
            
            # Update progress bar
            progress_bar.set_postfix({
                k: f"{v.avg:.4f}" for k, v in metrics.items()
            })
            
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    return {k: v.avg for k, v in metrics.items()}

def train_step(args, models, components, batch, batch_idx, device, dtype) -> Dict[str, float]:
    """Execute single training step with proper error handling."""
    try:
        # Move batch to device
        batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        with autocast(enabled=args.mixed_precision):
            loss = models["unet"](batch)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if args.mixed_precision:
            components["scaler"].scale(loss).backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                components["scaler"].unscale_(components["optimizer"])
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
                components["scaler"].step(components["optimizer"])
                components["scaler"].update()
                components["optimizer"].zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
                components["optimizer"].step()
                components["optimizer"].zero_grad()
        
        # Update EMA if enabled
        if components.get("ema"):
            components["ema"].step(models["unet"])
        
        # Update learning rate
        if components.get("lr_scheduler"):
            components["lr_scheduler"].step()
        
        # Calculate metrics
        metrics = {
            "loss": loss.item(),
            "lr": components["optimizer"].param_groups[0]["lr"],
            "grad_norm": get_grad_norm(models["unet"])
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in training step: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def log_epoch_metrics(wandb_run, metrics: Dict[str, float], epoch: int, global_step: int) -> None:
    """Log epoch metrics to W&B with proper error handling."""
    try:
        # Prepare metrics for logging
        log_dict = {
            f"train/{k}": v for k, v in metrics.items()
        }
        
        # Add epoch info
        log_dict.update({
            "train/epoch": epoch,
            "train/global_step": global_step
        })
        
        # Log to W&B
        wandb_run.log(log_dict, step=global_step)
        
        # Log to console
        logger.info(
            f"Epoch {epoch} metrics: " + 
            ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
        
    except Exception as e:
        logger.error(f"Failed to log epoch metrics: {str(e)}")
        logger.error(traceback.format_exc())

def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm with proper error handling."""
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    except Exception as e:
        logger.error(f"Failed to calculate gradient norm: {str(e)}")
        return 0.0
    
def train_epoch(epoch, args, models, components, device, dtype, wandb_run, global_step):
    """Execute single training epoch with proper logging"""
    logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
    
    # Train for one epoch
    epoch_metrics = train(args, models, components, device, dtype)
    
    if wandb_run:
        log_epoch_metrics(wandb_run, epoch_metrics, epoch, global_step)
        log_model_gradients(models["unet"], step=global_step)
        log_memory_stats(step=global_step)
    
    return epoch_metrics

def run_validation(args, models, components, device, dtype, global_step) -> Dict[str, float]:
    """Run validation with proper error handling and metrics tracking."""
    try:
        logger.info(f"Running validation at step {global_step}")
        
        # Get validator from components
        validator = components.get("validator")
        if validator is None:
            logger.warning("No validator found in components, skipping validation")
            return {}

        # Set models to eval mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.eval()

        with torch.no_grad():
            # Run validation using configured prompts
            validation_metrics = validator.run_validation(
                prompts=args.validation_prompts,
                output_dir=os.path.join(args.output_dir, f"validation_{global_step}"),
                log_to_wandb=args.use_wandb,
                guidance_scale=args.validation_guidance_scale,
                num_inference_steps=args.validation_num_steps,
                height=args.validation_height,
                width=args.validation_width,
                num_images_per_prompt=args.validation_images_per_prompt,
                seed=args.validation_seed
            )

            # Add additional metrics if needed
            if components.get("ema"):
                validation_metrics["validation/ema_decay"] = components["ema"].cur_decay_value

            # Log validation results
            logger.info(
                "Validation Results: " +
                ", ".join(f"{k}: {v:.4f}" for k, v in validation_metrics.items())
            )

            return validation_metrics

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

    finally:
        # Set models back to train mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.train()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main(args):
    """Main training function with improved organization and error handling."""
    wandb_run = None
    try:
        # Initial setup and logging
        setup_logging(
            log_dir=os.path.join("logs", args.wandb_run_name) if args.wandb_run_name else "logs",
            log_level=logging.DEBUG if args.verbose else logging.INFO
        )
        log_system_info()
        
        # Initialize W&B if enabled
        if args.use_wandb:
            wandb_run = setup_wandb(args)
            logger.info("Weights & Biases initialized successfully")

        # Device and precision setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if args.mixed_precision else torch.float32
        logger.info(f"Using device: {device}, precision: {dtype}")

        # Model initialization and loading
        try:
            models, optimizer_state, training_history = load_checkpoint(args)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

        # Move models to device and configure
        for model_name, model in models.items():
            if isinstance(model, torch.nn.Module):
                model.to(device=device, dtype=dtype)
                if args.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
                    model.enable_gradient_checkpointing()
                if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                    model.enable_xformers_memory_efficient_attention()

        # Initialize training components
        components = initialize_training_components(args, device, dtype, models)
        
        # Load optimizer state if resuming
        if optimizer_state and components.get("optimizer"):
            components["optimizer"].load_state_dict(optimizer_state)
            logger.info("Optimizer state loaded successfully")

        # Setup W&B monitoring
        if wandb_run and args.wandb_watch:
            wandb_run.watch(
                models["unet"],
                log="all",
                log_freq=args.wandb_log_freq,
                log_graph=True
            )
            log_memory_stats(step=0)

        # Initialize metrics tracking
        metrics = {
            "loss": AverageMeter("loss"),
            "grad_norm": AverageMeter("grad_norm"),
            "lr": AverageMeter("learning_rate"),
            "vae_loss": AverageMeter("vae_loss") if args.finetune_vae else None,
            "batch_time": AverageMeter("batch_time"),
            "data_time": AverageMeter("data_time")
        }

        # Training loop
        global_step = training_history.get("global_step", 0)
        best_loss = float('inf')
        scaler = GradScaler() if args.mixed_precision else None
        
        logger.info("Starting training loop")
        for epoch in range(args.num_epochs):
            try:
                # Training epoch
                epoch_metrics = train_epoch(
                    epoch=epoch,
                    args=args,
                    models=models,
                    components=components,
                    device=device,
                    dtype=dtype,
                    metrics=metrics,
                    scaler=scaler,
                    wandb_run=wandb_run,
                    global_step=global_step
                )
                
                global_step += len(components["train_dataloader"])
                
                # Update best loss
                if epoch_metrics["loss"] < best_loss:
                    best_loss = epoch_metrics["loss"]
                    if args.save_checkpoints:
                        save_checkpoint(
                            args=args,
                            models=models,
                            optimizer=components["optimizer"],
                            epoch=epoch,
                            metrics=epoch_metrics,
                            is_best=True
                        )

                # Regular checkpoint saving
                if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
                    save_checkpoint(
                        args=args,
                        models=models,
                        optimizer=components["optimizer"],
                        epoch=epoch,
                        metrics=epoch_metrics
                    )

                # Validation
                if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
                    validation_metrics = run_validation(
                        args=args,
                        models=models,
                        components=components,
                        device=device,
                        dtype=dtype,
                        global_step=global_step
                    )
                    
                    if wandb_run:
                        wandb.log(
                            {f"validation/{k}": v for k, v in validation_metrics.items()},
                            step=global_step
                        )

            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {str(e)}")
                logger.error(traceback.format_exc())
                if wandb_run:
                    wandb.alert(
                        title=f"Training Error - Epoch {epoch}",
                        text=str(e)
                    )
                continue

        # Save final outputs
        try:
            save_final_outputs(
                args=args,
                models=models,
                training_history={
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "final_metrics": epoch_metrics
                },
                components=components
            )
            
            if args.push_to_hub:
                push_to_hub(args, models)
                
            save_model_card(args, models)
            logger.info("Training completed successfully")
            
            return True

        except Exception as e:
            logger.error(f"Error during final save: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        if wandb_run:
            wandb.alert(
                title="Training Failed",
                text=str(e)
            )
        return False

    finally:
        # Cleanup
        if wandb_run:
            cleanup_wandb(wandb_run)
        
        # Release memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        # Set multiprocessing start method
        if not torch.multiprocessing.get_start_method(allow_none=True):
            torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        logger.warning("Multiprocessing start method already set")

    # Parse arguments and run
    args = parse_args()
    success = main(args)
    sys.exit(0 if success else 1)