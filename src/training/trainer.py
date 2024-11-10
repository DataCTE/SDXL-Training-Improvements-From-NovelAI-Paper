import torch
import time
from collections import defaultdict
from tqdm import tqdm
import wandb
import logging
import traceback
from safetensors.torch import load_file

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
    global_step
):
    """Single epoch training loop"""
    epoch_metrics = defaultdict(float)
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(train_dataloader):
        step_start = time.time()
        
        # Process batch
        latents = batch["latents"].to(device, dtype=dtype)
        text_embeddings = torch.cat([
            batch["text_embeddings"].to(device, dtype=dtype),
            batch["text_embeddings_2"].to(device, dtype=dtype)
        ], dim=-1)
        pooled_embeds = batch["pooled_text_embeddings_2"].to(device, dtype=dtype)
        
        # Get noise schedule
        sigmas = get_sigmas(args.num_inference_steps).to(device)
        sigma = sigmas[step % args.num_inference_steps].expand(latents.size(0))
        
        # Training step
        with torch.amp.autocast('cuda', dtype=dtype):
            loss, step_metrics = training_loss_v_prediction(
                unet,
                latents,
                sigma,
                text_embeddings,
                {
                    "text_embeds": pooled_embeds,
                    "time_ids": torch.tensor([1024, 1024, 1024, 1024, 0, 0], 
                                          device=device, dtype=dtype).repeat(latents.shape[0], 1)
                }
            )
            loss = loss / args.gradient_accumulation_steps
        
        # Apply tag-based weighting
        weighted_loss = tag_weighter.update_training_loss(loss, batch["tags"])
        weighted_loss.backward()
        
        # Update model if gradient accumulation complete
        if (step + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unet.parameters(), 
                args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            ema_model.update_parameters(unet)
            
            # VAE finetuning step
            if vae_finetuner and (global_step % args.vae_train_freq == 0):
                vae_loss = vae_finetuner.training_step(
                    batch["images"].to(device),
                    batch["latents"].to(device)
                )
                step_metrics['vae_loss'] = vae_loss
        
        # Update metrics
        for k, v in step_metrics.items():
            epoch_metrics[k] += v
            
        # Log step metrics
        if args.use_wandb and step % args.logging_steps == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/weighted_loss': weighted_loss.item(),
                'train/grad_norm': grad_norm.item(),
                'train/lr': lr_scheduler.get_last_lr()[0],
                **{f'train/{k}': v for k, v in step_metrics.items()}
            }, step=global_step)
        
        progress_bar.update(1)
        global_step += 1
    
    progress_bar.close()
    return epoch_metrics, global_step

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
            global_step=global_step
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
        
        # Save checkpoint
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
    
    logger.info("\n=== Training Complete ===")
    return training_history
