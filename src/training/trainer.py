import torch
import time
from collections import defaultdict
from tqdm import tqdm
import wandb
import logging
import traceback
from safetensors.torch import load_file
from training.loss import get_sigmas, training_loss_v_prediction
from training.ema import EMAModel
from training.utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

def train_one_epoch(
    unet,
    train_dataloader,
    optimizer,
    lr_scheduler,
    ema_model,
    validator,
    tag_weighter,
    vae_finetuner,
    args,
    device,
    dtype,
    epoch,
    global_step,
    models,
    train_components,
    training_history
):
    """Single epoch training loop"""
    try:
        logger.debug("\n=== Starting new epoch ===")
        epoch_metrics = defaultdict(float)
        
        # Initialize metrics
        window_size = 100
        running_loss = 0.0
        loss_history = []
        grad_norm_value = 0.0
        
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix[0]:.4f}, LR: {postfix[1]:.2e}',
            postfix=[0.0, lr_scheduler.get_last_lr()[0]]
        )
        
        for step, batch in enumerate(train_dataloader):
            logger.debug(f"\n--- Step {step} ---")
            step_start = time.time()
            
            # Validate batch contents
            required_keys = ["latents", "text_embeddings", "added_cond_kwargs"]
            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                raise ValueError(f"Batch missing required keys: {missing_keys}")
            
            if "added_cond_kwargs" not in batch or not isinstance(batch["added_cond_kwargs"], dict):
                raise ValueError("Batch has invalid added_cond_kwargs format")
                
            required_cond_keys = ["text_embeds", "time_ids"]
            missing_cond_keys = [key for key in required_cond_keys if key not in batch["added_cond_kwargs"]]
            if missing_cond_keys:
                raise ValueError(f"added_cond_kwargs missing required keys: {missing_cond_keys}")
            
            # Move batch to device and handle tensor conversion
            latents = batch["latents"]
            if isinstance(latents, list):
                latents = torch.stack(latents)
            latents = latents.to(device, dtype=dtype)
            
            # Use pre-computed text embeddings
            text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
            
            # Get noise schedule
            sigmas = get_sigmas(args.num_inference_steps).to(device)
            sigma = sigmas[step % args.num_inference_steps].expand(latents.size(0))
            logger.debug(f"Sigma value: {sigma[0].item():.4f}")
            
            # Ensure added_cond_kwargs is properly formatted
            added_cond_kwargs = {
                "text_embeds": batch["added_cond_kwargs"]["text_embeds"].to(device, dtype=dtype),
                "time_ids": batch["added_cond_kwargs"]["time_ids"].to(device, dtype=dtype)
            } if "added_cond_kwargs" in batch else {
                "text_embeds": torch.zeros(latents.shape[0], 1280, device=device, dtype=dtype),
                "time_ids": torch.tensor(
                    [[1024, 1024, 1024, 1024, 0, 0]] * latents.shape[0],
                    device=device,
                    dtype=dtype
                )
            }
            
            # Training step
            with torch.amp.autocast('cuda', dtype=dtype):
                loss, step_metrics = training_loss_v_prediction(
                    model=unet,
                    x_0=latents,
                    sigma=sigma,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs
                )
                
                logger.debug(f"Raw loss: {loss.item():.6f}")
                loss = loss / args.gradient_accumulation_steps
                logger.debug(f"Loss after accumulation scaling: {loss.item():.6f}")
            
            # Apply tag-based weighting
            weighted_loss = tag_weighter.update_training_loss(loss, batch["tags"])
            logger.debug(f"Weighted loss: {weighted_loss.item():.6f}")
            weighted_loss.backward()
            
            # Update model if gradient accumulation complete
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Log pre-clipping gradients
                total_norm = 0.0
                for p in unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logger.debug(f"Pre-clipping gradient norm: {total_norm:.6f}")
                
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), 
                    args.max_grad_norm
                ).item()
                logger.debug(f"Post-clipping gradient norm: {grad_norm_value:.6f}")
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                if ema_model is not None:
                    ema_model.update_parameters(unet)
            
            # Update metrics
            running_loss = 0.9 * running_loss + 0.1 * weighted_loss.item() if step > 0 else weighted_loss.item()
            loss_history.append(weighted_loss.item())
            logger.debug(f"Running loss: {running_loss:.6f}")
            
            # Calculate averages
            if len(loss_history) > window_size:
                loss_history = loss_history[-window_size:]
            average_loss = sum(loss_history) / len(loss_history)
            logger.debug(f"Average loss: {average_loss:.6f}")
            
            # Log metrics to wandb
            if args.use_wandb and step % args.logging_steps == 0:
                metrics = {
                    # Loss metrics
                    "loss/current": weighted_loss.item(),
                    "loss/average": average_loss,
                    "loss/running": running_loss,
                    
                    # Gradient metrics
                    "gradients/norm": grad_norm_value,
                    
                    # Learning rates
                    "lr/unet": lr_scheduler.get_last_lr()[0],
                    "lr/textencoder": lr_scheduler.get_last_lr()[0],
                    
                    # Loss components
                    "loss/mse_mean": step_metrics['loss/mse_mean'],
                    "loss/mse_std": step_metrics['loss/mse_std'],
                    "loss/snr_mean": step_metrics['loss/snr_mean'],
                    "loss/min_snr_gamma_mean": step_metrics['loss/min_snr_gamma_mean'],
                    
                    # Model metrics
                    "model/v_pred_std": step_metrics['model/v_pred_std'],
                    "model/v_target_std": step_metrics['model/v_target_std'],
                    "model/alpha_t_mean": step_metrics['model/alpha_t_mean'],
                    
                    # Noise metrics
                    "noise/sigma_mean": step_metrics['noise/sigma_mean'],
                    "noise/x_t_std": step_metrics['noise/x_t_std'],
                    
                    # Step counter for x-axis
                    "step": global_step
                }
                
                wandb.log(metrics)
                
                # Debug print all metrics
                logger.debug("Wandb metrics:")
                for k, v in metrics.items():
                    logger.debug(f"  {k}: {v:.6f}")
            
            progress_bar.postfix[0] = running_loss
            progress_bar.postfix[1] = lr_scheduler.get_last_lr()[0]
            progress_bar.update(1)
            
            step_metrics['step_time'] = time.time() - step_start
            global_step += 1
        
        progress_bar.close()
        return epoch_metrics, global_step
    except Exception as e:
        # Save checkpoint even if training fails
        logger.error(f"Training failed in epoch {epoch+1}: {str(e)}")
        logger.error(traceback.format_exc())
        
        if models and train_components and training_history:
            logger.info("Attempting to save checkpoint after failure...")
            try:
                save_checkpoint(
                    checkpoint_dir=args.output_dir,
                    models=models,
                    train_components=train_components,
                    training_state=training_history
                )
            except Exception as save_error:
                logger.error(f"Failed to save checkpoint after training error: {str(save_error)}")
        else:
            logger.warning("Cannot save checkpoint: missing required components")
        
        raise  # Re-raise the original training error

def train(args, models, train_components, device, dtype):
    """
    Main training loop
    
    Args:
        args: Training arguments
        models (dict): Dictionary containing models
        train_components (dict): Dictionary containing training components
        device: Target device
        dtype: Training precision
    
    Returns:
        dict: Training history
    """
    # Unpack components
    unet = models["unet"]
    train_dataloader = train_components["train_dataloader"]
    optimizer = train_components["optimizer"]
    lr_scheduler = train_components["lr_scheduler"]
    ema_model = train_components["ema_model"]
    validator = train_components["validator"]
    tag_weighter = train_components["tag_weighter"]
    vae_finetuner = train_components.get("vae_finetuner")
    
    # Initialize training state
    global_step = 0
    training_history = {
        'epoch_losses': [],
        'learning_rates': [],
        'grad_norms': [],
        'validation_scores': [],
        'step_times': []
    }
    
    logger.info("\n=== Starting Training ===")
    
    try:    
        for epoch in range(args.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
            # Train one epoch
            epoch_metrics, global_step = train_one_epoch(
                unet=unet,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema_model=ema_model,
                validator=validator,
                tag_weighter=tag_weighter,
                vae_finetuner=vae_finetuner,
                args=args,
                device=device,
                dtype=dtype,
                epoch=epoch,
                global_step=global_step,
                models=models,
                train_components=train_components,
                training_history=training_history
            )
            
            # Log epoch metrics
            avg_loss = epoch_metrics['loss/total'] / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
            
            # Update training history
            training_history['epoch_losses'].append(avg_loss)
            training_history['learning_rates'].append(lr_scheduler.get_last_lr()[0])
            
            # Run validation
            if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
                logger.info("Running validation...")
                validation_metrics = validator.run_paper_validation(args.validation_prompts)
                training_history['validation_scores'].append(validation_metrics)
                
                if args.use_wandb:
                    wandb.log({
                        'validation': validation_metrics,
                        'epoch': epoch + 1
                    }, step=global_step)
            
            
            # Regular checkpoint saving during training
            if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
                logger.info("Saving checkpoint...")
                save_checkpoint(
                    models=models,
                    train_components=train_components,
                    args=args,
                    epoch=epoch,
                    training_history=training_history,
                    output_dir=args.output_dir
                )
            
            # Always save final checkpoint
            logger.info("Saving final checkpoint...")
            save_checkpoint(
                checkpoint_dir=args.output_dir,
                models=models,
                train_components=train_components,
                training_state=training_history
            )
            
            logger.info("\n=== Training Complete ===")
            return training_history
        
    except Exception as e:
        # Save checkpoint even if training fails
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        logger.info("Attempting to save checkpoint after failure...")
        try:
            save_checkpoint(
                checkpoint_dir=args.output_dir,
                models=models,
                train_components=train_components,
                training_state=training_history
            )
        except Exception as save_error:
            logger.error(f"Failed to save checkpoint after training error: {str(save_error)}")
        
        raise  # Re-raise the original training error