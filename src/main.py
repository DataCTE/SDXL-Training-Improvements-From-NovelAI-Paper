"""Main entry point for SDXL training."""

import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import torch
import wandb

from src.config.args import parse_args
from src.training.trainer import (
    train_epoch,
    initialize_training_components,
    run_validation,
    setup_optimizer,
    setup_vae_finetuner,
    setup_ema
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint, save_final_outputs
from src.utils.hub import create_model_card, save_model_card, push_to_hub
from src.utils.logging import (
    setup_logging,
    log_system_info,
    log_training_setup,
    setup_wandb,
    cleanup_wandb,
    log_memory_stats,
    log_model_gradients
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main training function with improved organization and error handling."""
    # Parse arguments and setup logging
    config = parse_args()
    setup_logging(config.system.verbose)
    
    try:
        # Log system information and setup
        log_system_info()
        
        # Get device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load model and initialize components
        models, train_components, training_history = load_checkpoint(
            config.model.model_path,
            config.model.vae_path if hasattr(config.model, 'vae_path') else None,
            dtype=dtype
        )
        
        # Initialize training components with loaded state
        training_state = initialize_training_components(config, models)
        training_state["device"] = device
        training_state["dtype"] = dtype
        training_state["train_components"] = train_components
        training_state["training_history"] = training_history
        
        # Log training setup with all required components
        log_training_setup(config, models, training_state)
        
        # Setup output directory
        output_dir = Path(config.model.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        wandb_run = None
        if config.logging.use_wandb:
            wandb_run = setup_wandb(config)
        
        # Setup training components
        setup_optimizer(config, models)
        if config.vae.finetune_vae:
            setup_vae_finetuner(config.vae, models)
        if config.ema.use_ema:
            setup_ema(config.ema, models["unet"])
        
        # Resume from checkpoint if specified
        if config.logging.resume_from_checkpoint:
            loaded_models, loaded_components, loaded_history = load_checkpoint(
                config.logging.resume_from_checkpoint,
                config.model.vae_path if hasattr(config.model, 'vae_path') else None,
                dtype=dtype
            )
            # Update models and training state with loaded components
            models.update(loaded_models)
            training_state["train_components"] = loaded_components
            training_state["training_history"] = loaded_history
        
        # Training loop
        for epoch in range(config.training.num_epochs):
            logger.info("Starting epoch %d/%d", epoch + 1, config.training.num_epochs)
            
            # Train for one epoch
            train_metrics = train_epoch(
                epoch=epoch,
                args=config,
                models=models,
                components=training_state,
                device=device,
                dtype=dtype,
                global_step=training_state.get("total_steps", 0)
            )
            
            # Log metrics if using wandb
            if config.logging.use_wandb:
                wandb.log({f"train/{k}": v for k, v in train_metrics.items()})
            
            # Run validation if available
            if "val_dataloader" in training_state:
                val_metrics = run_validation(
                    args=config,
                    models=models,
                    components=training_state,
                    device=device,
                    dtype=dtype,
                    global_step=training_state.get("total_steps", 0)
                )
                
                # Log validation metrics
                if config.logging.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
            
            # Save checkpoint if enabled
            if config.logging.save_checkpoints:
                checkpoint_path = output_dir / f"checkpoint-{epoch}.pt"
                save_checkpoint(
                    save_path=checkpoint_path,
                    models=models,
                    train_components=training_state,
                    training_history={"epoch": epoch, "total_steps": training_state.get("total_steps", 0)},
                    is_final=False
                )
            
            # Log memory stats and model gradients
            current_step = training_state.get("total_steps", 0)
            log_memory_stats(step=current_step)
            log_model_gradients(
                model=models["unet"],
                step=current_step
            )
        
        # Save final outputs
        final_model_path = output_dir / "final_model"
        save_final_outputs(
            final_model_path,
            training_state["model"],
            training_state["ema_model"] if config.ema.use_ema else None,
            training_state["vae"] if config.vae.use_vae else None
        )
        
        # Create and save model card
        model_card = create_model_card(
            config=config,
            training_history=training_state.get("training_history", {
                "loss_history": [],
                "validation_scores": [],
                "best_score": float('inf'),
                "total_steps": training_state.get("total_steps", 0)
            })
        )
        save_model_card(output_dir / "README.md", model_card)
        
        # Push to hub if configured
        if hasattr(config.logging, 'push_to_hub') and config.logging.push_to_hub:
            push_to_hub(
                output_dir,
                training_state["model"],
                training_state["ema_model"] if config.ema.use_ema else None,
                training_state["vae"] if config.vae.use_vae else None,
                config
            )
        
        logger.info("Training completed successfully!")
        
    except (torch.cuda.CudaError, torch.cuda.OutOfMemoryError) as e:
        logger.error("CUDA error occurred: %s", str(e), exc_info=True)
        sys.exit(3)
    except (TypeError, AttributeError) as e:
        logger.error("Type or attribute error in training pipeline: %s", str(e), exc_info=True)
        sys.exit(5)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Failed to import required module: %s", str(e), exc_info=True)
        sys.exit(4)
    except (RuntimeError, ValueError, OSError) as e:
        # RuntimeError: CUDA/model errors
        # ValueError: Invalid parameter values or model states
        # OSError: File operations, checkpoint loading/saving
        logger.error("Training failed: %s", str(e), exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
        
    finally:
        # Cleanup
        if config.logging.use_wandb and wandb_run is not None:
            cleanup_wandb(wandb_run)


if __name__ == "__main__":
    main()