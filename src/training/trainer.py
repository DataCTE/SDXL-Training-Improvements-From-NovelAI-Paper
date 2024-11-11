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
from models.reward_model import RewardModel
from pathlib import Path
import torch.nn.functional as F
from ..utils.image_utils import encode_images, add_noise
from ..utils.text_utils import encode_prompts

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
            
            # Get noise schedule
            sigmas = get_sigmas(args.num_inference_steps).to(device)
            sigma = sigmas[step % args.num_inference_steps].expand(latents.size(0))
            
            # Properly format added conditioning
            added_cond_kwargs = {
                "text_embeds": batch["added_cond_kwargs"]["text_embeds"].to(device, dtype=dtype),
                "time_ids": batch["added_cond_kwargs"]["time_ids"].to(device, dtype=dtype)
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
                
                # Skip batch if loss function returned None
                if loss is None:
                    logger.debug("Skipping batch due to invalid aspect ratio")
                    continue
                
                logger.debug(f"Raw loss: {loss.item():.6f}")
                loss = loss / args.gradient_accumulation_steps
                logger.debug(f"Loss after accumulation scaling: {loss.item():.6f}")
            
            # Apply tag-based weighting only if enabled
            if args.use_tag_weighting and tag_weighter is not None:
                if "tags" in batch:
                    weighted_loss = tag_weighter.update_training_loss(loss, batch["tags"])
                    logger.debug(f"Weighted loss: {weighted_loss.item():.6f}")
                else:
                    logger.warning("Tag weighter provided but no tags found in batch")
                    weighted_loss = loss
            else:
                weighted_loss = loss
                logger.debug("Tag weighting disabled or no weighter available")
            
            if args.use_itercomp and models["reward_models"] is not None:
                # Calculate composition-aware rewards
                rewards = []
                for reward_model in models["reward_models"].values():
                    reward = reward_model(text_embeddings, latents)
                    rewards.append(reward)
                    
                # Combine with existing loss
                reward_loss = sum(rewards) / len(rewards)
                weighted_loss = weighted_loss + args.reward_weight * reward_loss
            
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

def train_reward_models(reward_models, train_dataloader, args):
    """Train composition-aware reward models on model gallery preferences"""
    logger.info("Training composition-aware reward models...")
    
    for reward_type, reward_model in reward_models.items():
        logger.info(f"Training {reward_type} reward model")
        reward_model.train()
        
        optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        for batch in train_dataloader:
            # Get winning and losing images from model gallery preferences
            winning_images = batch["winning_images"].to(args.device) 
            losing_images = batch["losing_images"].to(args.device)
            prompts = batch["prompts"]
            
            # Calculate reward scores
            winning_scores = reward_model(prompts, winning_images)
            losing_scores = reward_model(prompts, losing_images)
            
            # Compute preference loss
            loss = -torch.log(torch.sigmoid(winning_scores - losing_scores)).mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.use_wandb:
                wandb.log({
                    f"reward_model/{reward_type}_loss": loss.item()
                })
                
        reward_model.eval()
    
    logger.info("Finished training reward models")

def expand_model_gallery(models, train_dataloader, args):
    """Expand model gallery with samples from current model"""
    logger.info("Expanding model gallery with new samples...")
    
    base_model = models["unet"]
    base_model.eval()
    
    gallery_samples = []
    with torch.no_grad():
        for batch in train_dataloader:
            # Generate images with current model
            prompts = batch["prompts"]
            generated = generate_images(
                models,
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )
            
            # Get reward scores
            reward_scores = {}
            for reward_type, reward_model in models["reward_models"].items():
                scores = reward_model(prompts, generated).cpu()
                reward_scores[reward_type] = scores
                
            # Store samples and scores
            gallery_samples.append({
                "images": generated.cpu(),
                "prompts": prompts,
                "reward_scores": reward_scores
            })
            
            if args.use_wandb:
                wandb.log({
                    f"gallery/reward_{reward_type}": scores.mean()
                    for reward_type, scores in reward_scores.items()
                })
    
    # Update gallery dataset
    update_gallery_dataset(gallery_samples, args)
    logger.info("Finished expanding model gallery")

def update_gallery_dataset(new_samples, args):
    """Update the gallery dataset with new samples"""
    gallery_path = Path(args.output_dir) / "model_gallery"
    gallery_path.mkdir(exist_ok=True)
    
    # Load existing gallery if any
    existing_samples = []
    if (gallery_path / "samples.pt").exists():
        existing_samples = torch.load(gallery_path / "samples.pt")
        
    # Combine with new samples
    all_samples = existing_samples + new_samples
    
    # Save updated gallery
    torch.save(all_samples, gallery_path / "samples.pt")
    
    logger.info(f"Updated gallery dataset with {len(new_samples)} new samples")

def generate_images(models, prompts, num_inference_steps=50, guidance_scale=7.5):
    """Generate images using the current model"""
    # Set up pipeline components
    unet = models["unet"]
    vae = models["vae"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    
    # Encode text
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(unet.device))[0]
    
    # Generate latents
    latents = torch.randn(
        (len(prompts), unet.config.in_channels, 64, 64),
        device=unet.device
    )
    
    # Denoise latents
    for t in range(num_inference_steps):
        with torch.no_grad():
            noise_pred = unet(latents, t, text_embeddings).sample
            latents = diffusion_step(latents, noise_pred, t, guidance_scale)
    
    # Decode latents
    with torch.no_grad():
        images = vae.decode(latents).sample
        
    return images

def iterative_feedback_learning(models, train_dataloader, args):
    """Implement iterative feedback learning loop"""
    for iteration in range(args.num_iterations):
        logger.info(f"Starting iteration {iteration+1}/{args.num_iterations}")
        
        # Train reward models
        if iteration > 0:  # Skip first iteration
            train_reward_models(models["reward_models"], train_dataloader)
            
        # Train base diffusion model
        train(models, train_dataloader, args)
        
        # Expand model gallery
        if iteration < args.num_iterations - 1:
            expand_model_gallery(models, train_dataloader)
            
    return models

def diffusion_step(latents, noise_pred, timestep, guidance_scale=7.5):
    """
    Perform a single diffusion step with classifier-free guidance
    
    Args:
        latents: Current latent state
        noise_pred: Predicted noise from UNet
        timestep: Current timestep
        guidance_scale: Scale factor for classifier-free guidance (default: 7.5)
    
    Returns:
        Updated latents after denoising step
    """
    # Get alphas for current timestep
    alpha_t = get_alpha_schedule()[timestep]
    alpha_prev = get_alpha_schedule()[timestep - 1] if timestep > 0 else 1.0
    
    # Calculate coefficients
    c0 = 1 / torch.sqrt(alpha_t)
    c1 = (1 - alpha_t) / torch.sqrt(1 - alpha_prev)
    
    # Apply classifier-free guidance
    if guidance_scale > 1.0:
        # Split noise prediction into unconditional and conditional parts
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        # Combine using guidance scale
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
    # Update latents using predicted noise
    latents = c0 * latents - c1 * noise_pred
    
    return latents

def get_alpha_schedule(num_train_timesteps=1000):
    """
    Get the alpha (noise level) schedule for the diffusion process
    
    Args:
        num_train_timesteps: Number of diffusion timesteps
    
    Returns:
        Tensor containing alpha values for each timestep
    """
    # Linear schedule from β1 to β2
    beta_start = 0.00085
    beta_end = 0.012
    betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
    
    # Calculate alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return alphas_cumprod

def train_with_itercomp(models, train_dataloader, args):
    """
    Train using IterComp approach while preserving existing functionality
    """
    if args.use_itercomp:
        logger.info("Starting IterComp training...")
        return iterative_feedback_learning(models, train_dataloader, args)
    else:
        logger.info("Starting standard training...")
        return train(models, train_dataloader, args)

def training_step(unet, batch, args, reward_models=None):
    """
    Single training step with optional IterComp reward guidance
    
    Args:
        unet: UNet model
        batch: Training batch containing images and prompts
        args: Training arguments
        reward_models: Optional dict of reward models for IterComp
    
    Returns:
        Total loss combining base loss and optional reward loss
    """
    # Compute base diffusion loss
    loss = compute_base_loss(unet, batch, args)
    
    # Add IterComp reward guidance if enabled
    if reward_models is not None:
        reward_loss = compute_reward_loss(reward_models, batch)
        loss = loss + args.reward_weight * reward_loss
        
    return loss

def compute_base_loss(unet, batch, args):
    """
    Compute standard diffusion training loss
    
    Args:
        unet: UNet model
        batch: Training batch
        args: Training arguments
    
    Returns:
        Base diffusion loss
    """
    # Get images and prompts from batch
    images = batch["images"].to(args.device)
    prompts = batch["prompts"]
    
    # Encode images to latent space
    latents = encode_images(images)
    
    # Add noise to latents
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, args.num_train_timesteps, (latents.shape[0],), device=latents.device)
    noisy_latents = add_noise(latents, noise, timesteps)
    
    # Get text embeddings
    text_embeddings = encode_prompts(prompts)
    
    # Predict noise
    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
    
    # Calculate loss
    if args.loss_fn == "l2":
        loss = F.mse_loss(noise_pred, noise)
    elif args.loss_fn == "l1":
        loss = F.l1_loss(noise_pred, noise)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
        
    return loss

def compute_reward_loss(reward_models, batch):
    """
    Compute composition-aware reward loss from multiple reward models
    
    Args:
        reward_models: Dict of reward models for different aspects
        batch: Training batch
    
    Returns:
        Combined reward loss
    """
    total_reward_loss = 0
    
    # Get generated and reference images
    generated_images = batch["generated_images"] 
    reference_images = batch["reference_images"]
    prompts = batch["prompts"]
    
    # Calculate reward loss for each type of reward model
    for reward_type, reward_model in reward_models.items():
        # Get reward scores
        generated_score = reward_model(prompts, generated_images)
        reference_score = reward_model(prompts, reference_images)
        
        # Calculate loss based on reward difference
        reward_loss = torch.mean(reference_score - generated_score)
        total_reward_loss += reward_loss
        
    return total_reward_loss