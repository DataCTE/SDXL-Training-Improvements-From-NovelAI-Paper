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
import math
import os

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
            step_start = time.time()
            
            # Validate batch contents
            required_keys = ["latents", "text_embeddings", "added_cond_kwargs"]
            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                raise ValueError(f"Batch missing required keys: {missing_keys}")
            
            # Move batch to device and handle tensor conversion
            latents = batch["latents"].to(device, dtype=dtype)
            text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
            
            # Handle added conditioning kwargs
            added_cond_kwargs = {
                "text_embeds": batch["added_cond_kwargs"]["text_embeds"].to(device, dtype=dtype),
                "time_ids": batch["added_cond_kwargs"]["time_ids"].to(device, dtype=dtype)
            }
            
            # Get noise schedule with resolution-aware sigma scaling
            _, _, height, width = latents.shape
            sigmas = get_sigmas(
                num_inference_steps=args.num_inference_steps,
                sigma_min=0.0292,
                height=height,
                width=width
            ).to(device)
            sigma = sigmas[step % args.num_inference_steps].expand(latents.size(0))
            
            # Log sigma distribution periodically
            if args.use_wandb and step % args.logging_steps == 0:
                timestep_histogram = torch.histc(sigma, bins=20)
                wandb.log({
                    "training/sigma_histogram": wandb.Histogram(timestep_histogram.cpu().numpy()),
                    "training/sigma_mean": sigma.mean().item(),
                    "training/sigma_std": sigma.std().item(),
                    "training/step": step,
                }, step=global_step)
            
            # Modified training step for BFloat16
            if dtype == torch.bfloat16:
                # Convert UNet to bfloat16 but keep embeddings in float32
                unet = unet.to(dtype=torch.bfloat16)
                loss, step_metrics = training_loss_v_prediction(
                    model=unet,
                    x_0=latents,
                    sigma=sigma,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs
                )
                
                # Skip batch if loss is unstable
                if torch.isnan(loss) or torch.isinf(loss) or loss > 1e5:
                    logger.warning(f"Skipping batch due to unstable loss: {loss.item()}")
                    continue
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            else:
                # Use AMP for float16/float32
                with torch.amp.autocast('cuda', dtype=dtype):
                    loss, step_metrics = training_loss_v_prediction(
                        model=unet,
                        x_0=latents,
                        sigma=sigma,
                        text_embeddings=text_embeddings,
                        added_cond_kwargs=added_cond_kwargs
                    )
                    
                    # Skip batch if loss was unstable
                    if loss is None:
                        continue  # This continue is valid as we're in the training loop
                    
                    loss = loss / args.gradient_accumulation_steps
                
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if dtype == torch.bfloat16:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), 
                        args.max_grad_norm
                    )
                    optimizer.step()
                else:
                    # Clip gradients with scaler
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        unet.parameters(),
                        args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                # Update EMA model if it exists
                if ema_model is not None:
                    ema_model.step(unet)
                    
                global_step += 1
                
                # Log EMA decay rate if using wandb
                if args.use_wandb and step % args.logging_steps == 0:
                    metrics["ema/decay"] = ema_model.cur_decay_value if ema_model else 0.0
                
                # Run validation at specified step intervals
                if validator and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    logger.info(f"\nRunning validation at step {global_step}")
                    validation_metrics = validator.run_validation(
                        prompts=args.validation_prompts,
                        output_dir=os.path.join(args.output_dir, f"validation_step_{global_step}"),
                        num_images_per_prompt=1,
                        guidance_scale=5.0,
                        num_inference_steps=28,
                        height=1024,
                        width=1024,
                        log_to_wandb=args.use_wandb
                    )
                    training_history['validation_scores'].append({
                        'step': global_step,
                        'metrics': validation_metrics
                    })
                    
                    if args.use_wandb:
                        wandb.log({
                            'validation': validation_metrics,
                            'step': global_step,
                            'validation/images': wandb.Image(
                                os.path.join(args.output_dir, f"validation_step_{global_step}")
                            )
                        }, step=global_step)
            
            # Update metrics
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
            loss_history.append(loss.item())
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
                    "loss/current": loss.item(),
                    "loss/average": average_loss,
                    "loss/running": running_loss,
                    
                    # Gradient metrics
                    "gradients/norm": grad_norm_value,
                    
                    # Learning rates
                    "lr/unet": lr_scheduler.get_last_lr()[0],
                    
                    # Step metrics from loss function
                    **step_metrics,  # This will include all metrics from training_loss_v_prediction
                    
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
            
        progress_bar.close()
        return epoch_metrics, global_step
        
    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        raise

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train(args, models, train_components, device, dtype):
    """Main training loop"""
    try:
        # Initialize training components
        unet = models["unet"]
        train_dataloader = train_components["train_dataloader"]
        optimizer = train_components["optimizer"]
        lr_scheduler = train_components["lr_scheduler"]
        
        # Get dataset from dataloader
        dataset = train_dataloader.dataset
        
        # Initialize EMA if enabled
        ema_model = None
        if args.use_ema:
            ema_model = EMAModel(
                model=unet,
                decay=args.ema_decay,  # Add this to args if not present
                device=device,
                dtype=dtype
            )
            train_components["ema_model"] = ema_model
        
        validator = train_components.get("validator")
        tag_weighter = train_components.get("tag_weighter")
        vae_finetuner = train_components.get("vae_finetuner")
        
        # Initialize training history
        training_history = {
            'epoch_losses': [],
            'learning_rates': [],
            'validation_scores': [],
            'global_step': 0
        }
        
        global_step = 0
        
        # Main epoch loop
        for epoch in range(args.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
            
            # Shuffle dataset before each epoch
            dataset.shuffle_samples()
            
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
            
            # Update training history
            training_history['epoch_losses'].append(epoch_metrics['loss/total'] / len(train_dataloader))
            training_history['learning_rates'].append(lr_scheduler.get_last_lr()[0])
            training_history['global_step'] = global_step
            
            # Save regular checkpoint
            if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
                save_checkpoint(
                    models=models,
                    train_components=train_components,
                    args=args,
                    epoch=epoch,
                    training_history=training_history,
                    output_dir=os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                )
        
        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        save_checkpoint(
            models=models,
            train_components=train_components,
            args=args,
            epoch=args.num_epochs-1,
            training_history=training_history,
            output_dir=os.path.join(args.output_dir, "final")
        )
        
        logger.info("\n=== Training Complete ===")
        return training_history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save emergency checkpoint
        try:
            save_checkpoint(
                models=models,
                train_components=train_components,
                args=args,
                epoch=-1,
                training_history=training_history,
                output_dir=os.path.join(args.output_dir, "emergency")
            )
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {str(save_error)}")
        
        raise