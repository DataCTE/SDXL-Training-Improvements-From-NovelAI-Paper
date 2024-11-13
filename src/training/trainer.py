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
import numpy as np


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
    training_history
):
    """Single epoch training loop with detailed logging"""
    try:
        logger.debug(f"\n=== Starting epoch {epoch+1} ===")
        
        # Add epoch to wandb at start of epoch
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0]
            }, step=global_step)
        
        # Initialize metrics
        window_size = 100
        running_loss = 0.0
        loss_history = []
        grad_norm_value = 0.0
        tag_metrics = defaultdict(list)
        
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix[0]:.4f}, LR: {postfix[1]:.2e}',
            postfix=[0.0, lr_scheduler.get_last_lr()[0]]
        )
        
        # Set training mode and ensure gradient checkpointing
        unet.train()
        if args.gradient_checkpointing:
            if hasattr(unet, "gradient_checkpointing_enable"):
                unet.gradient_checkpointing_enable()
            
            for encoder in [models.get("text_encoder"), models.get("text_encoder_2")]:
                if encoder is not None and encoder.training:
                    if hasattr(encoder, "gradient_checkpointing_enable"):
                        encoder.gradient_checkpointing_enable()

        for step, batch in enumerate(train_dataloader):
            step_start = time.time()
            
            # Validate batch contents
            required_keys = ["latents", "text_embeddings", "added_cond_kwargs", "tags", "special_tags", "tag_weights"]
            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                raise ValueError(f"Batch missing required keys: {missing_keys}")
            
            # Move batch to device
            latents = batch["latents"].to(device, dtype=dtype)
            text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
            added_cond_kwargs = {
                k: v.to(device, dtype=dtype) if torch.is_tensor(v) else v
                for k, v in batch["added_cond_kwargs"].items()
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
            
            # Calculate loss with tag weighting
            loss, step_metrics = training_loss_v_prediction(
                model=unet,
                x_0=latents,
                sigma=sigma,
                text_embeddings=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                tag_weighter=tag_weighter,
                batch_tags={
                    'tags': batch['tags'],
                    'tag_weights': batch['tag_weights'],
                    'special_tags': batch['special_tags']
                }
            )
            
            # Skip batch if loss is unstable
            if torch.isnan(loss) or torch.isinf(loss) or loss > 1e5:
                logger.warning(f"Skipping batch due to unstable loss: {loss.item()}")
                continue
            
            # Backward pass with gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), 
                    args.max_grad_norm
                )
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                # Update EMA model if it exists
                if ema_model is not None:
                    ema_model.step(unet)
                    
                    # Log EMA metrics
                    if args.use_wandb and step % args.logging_steps == 0:
                        wandb.log({
                            "ema/decay": ema_model.get_decay_value(),
                            "ema/num_updates": ema_model.num_updates
                        }, step=global_step)
                
                global_step += 1
            
            # Update metrics
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
            loss_history.append(loss.item())
            
            # Update tag metrics
            for key, value in step_metrics.items():
                if key.startswith('tags/'):
                    tag_metrics[key].append(value)
            
            # Calculate averages for progress bar
            if len(loss_history) > window_size:
                loss_history = loss_history[-window_size:]
            average_loss = sum(loss_history) / len(loss_history)
            
            # Update progress bar
            progress_bar.set_postfix([average_loss, lr_scheduler.get_last_lr()[0]])
            progress_bar.update(1)
            
            # Detailed logging
            if args.use_wandb and step % args.logging_steps == 0:
                metrics = {
                    "training/loss": average_loss,
                    "training/running_loss": running_loss,
                    "training/grad_norm": grad_norm_value,
                    "training/learning_rate": lr_scheduler.get_last_lr()[0],
                    "training/step_time": time.time() - step_start,
                    # Tag metrics averages
                    "tags/weight_mean": np.mean(tag_metrics["tags/weight_mean"]),
                    "tags/niji_ratio": np.mean(tag_metrics["tags/niji_count"]) / args.batch_size,
                    "tags/quality_6_ratio": np.mean(tag_metrics["tags/quality_6_count"]) / args.batch_size,
                    "tags/stylize_mean": np.mean(tag_metrics["tags/stylize_mean"]),
                    "tags/chaos_mean": np.mean(tag_metrics["tags/chaos_mean"])
                }
                wandb.log(metrics, step=global_step)
        
        # End of epoch logging
        epoch_metrics = {
            "epoch/average_loss": np.mean(loss_history),
            "epoch/final_lr": lr_scheduler.get_last_lr()[0],
            "epoch/total_steps": step + 1,
            "epoch/tag_stats": {
                k: np.mean(v) for k, v in tag_metrics.items()
            }
        }
        
        if args.use_wandb:
            wandb.log(epoch_metrics, step=global_step)
        
        return global_step

    except Exception as e:
        logger.error(f"Error in epoch {epoch+1}: {str(e)}")
        logger.error(traceback.format_exc())
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
    training_history = {
        'loss_history': [],
        'validation_scores': [],
        'ema_validation_scores': [],  # Add EMA validation tracking
        'best_score': float('inf'),
        'best_ema_score': float('inf')  # Add EMA best score tracking
    }
    
    global_step = 0
    start_epoch = 0
    
    # Monitor memory usage with gradient checkpointing
    if args.gradient_checkpointing and torch.cuda.is_available():
        logger.info("\nInitial memory usage with gradient checkpointing:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        logger.info(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        if args.use_wandb:
            wandb.log({
                "memory/allocated_gb": torch.cuda.memory_allocated()/1e9,
                "memory/cached_gb": torch.cuda.memory_reserved()/1e9
            })
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, models, train_components)
        start_epoch = checkpoint_data['epoch'] + 1
        global_step = checkpoint_data.get('global_step', 0)
        if 'training_history' in checkpoint_data:
            training_history.update(checkpoint_data['training_history'])
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Log epoch start
        logger.info(f"\nStarting epoch {epoch+1}/{args.num_epochs}")
        if args.use_wandb:
            wandb.log({"train/epoch": epoch}, step=global_step)
        
        global_step = train_one_epoch(
            unet=models["unet"],
            train_dataloader=train_components["train_dataloader"],
            optimizer=train_components["optimizer"],
            lr_scheduler=train_components["lr_scheduler"],
            ema_model=train_components["ema_model"],
            validator=train_components["validator"],
            tag_weighter=train_components["tag_weighter"],
            vae_finetuner=train_components["vae_finetuner"],
            args=args,
            device=device,
            dtype=dtype,
            epoch=epoch,
            global_step=global_step,
            models=models,
            training_history=training_history
        )
        
        # Run end-of-epoch validation for both regular and EMA models
        if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
            logger.info("\nRunning end-of-epoch validation...")
            
            # Regular model validation
            validation_metrics = train_components["validator"].run_validation(
                model=models["unet"],
                prompts=args.validation_prompts,
                output_dir=os.path.join(args.output_dir, f"validation_epoch_{epoch+1}"),
                num_images_per_prompt=1,
                guidance_scale=5.0,
                num_inference_steps=28,
                height=1024,
                width=1024,
                log_to_wandb=args.use_wandb
            )
            
            # EMA model validation if enabled
            if train_components["ema_model"] is not None:
                ema_validation_metrics = train_components["validator"].run_validation(
                    model=train_components["ema_model"].get_model(),
                    prompts=args.validation_prompts,
                    output_dir=os.path.join(args.output_dir, f"ema_validation_epoch_{epoch+1}"),
                    num_images_per_prompt=1,
                    guidance_scale=5.0,
                    num_inference_steps=28,
                    height=1024,
                    width=1024,
                    log_to_wandb=args.use_wandb
                )
                
                if args.use_wandb:
                    wandb.log({
                        "validation/epoch": epoch + 1,
                        "validation/regular": validation_metrics,
                        "validation/ema": ema_validation_metrics
                    }, step=global_step)
        
        # Save checkpoint at epoch end if requested
        if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            save_checkpoint(
                models=models,
                train_components=train_components,
                args=args,
                epoch=epoch,
                global_step=global_step,
                training_history=training_history,
                output_dir=checkpoint_dir
            )
    
    return training_history