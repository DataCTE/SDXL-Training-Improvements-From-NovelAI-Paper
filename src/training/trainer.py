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
from models.reward_model import (
    get_model_manager,
    CLIPFeatureExtractor,
    DETRObjectDetector,
    AttributeBindingRewardModel,
    SpatialRewardModel,
    NonSpatialRewardModel,
    collect_preferences,
    train_reward_model
)
from pathlib import Path
import torch.nn.functional as F
from utils.image_utils import encode_images, add_noise
from utils.text_utils import encode_prompts

logger = logging.getLogger(__name__)

# Get global model manager instance
model_manager = get_model_manager()

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
    """Train for one epoch with optional reward guidance"""
    try:
        unet.train()
        running_loss = 0.0
        loss_history = []
        window_size = 100
        epoch_metrics = defaultdict(float)
        
        # Add gradient norm tracking
        max_grad_norm = 1.0
        grad_norms = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Calculate loss with reward guidance if enabled
            loss, step_metrics = training_step(
                unet=unet,
                batch=batch,
                args=args,
                reward_models=models.get("reward_models")
            )
            
            # Skip bad batches
            if loss is None:
                continue
                
            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"Skipping step {step} due to invalid loss value")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), 
                    max_grad_norm
                )
                grad_norms.append(grad_norm.item())
                
                # Skip step if gradients are invalid
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Skipping step {step} due to invalid gradients")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                if ema_model is not None:
                    ema_model.update_parameters(unet)
            
            # Update metrics
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
            loss_history.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{running_loss:.6f}",
                **{k: f"{v:.6f}" for k, v in step_metrics.items()}
            })
            
            # Update epoch metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] += v
            
            global_step += 1
            
        # Average epoch metrics
        num_steps = len(train_dataloader)
        epoch_metrics = {k: v / num_steps for k, v in epoch_metrics.items()}
        
        return epoch_metrics, global_step
        
    except Exception as e:
        logger.error("Error in train_one_epoch")
        logger.error(traceback.format_exc())
        raise

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
    
    try:
        total_metrics = defaultdict(float)
        
        for reward_type, reward_model in reward_models.items():
            logger.info(f"Training {reward_type} reward model")
            
            # Collect preference pairs
            preference_pairs = collect_preferences(
                reward_type=reward_type,
                train_dataloader=train_dataloader,
                model_manager=model_manager,
                args=args
            )
            
            # Train model on preferences
            reward_loss = train_reward_model(
                reward_model=reward_model,
                preference_pairs=preference_pairs,
                args=args
            )
            
            total_metrics[f"{reward_type}_loss"] = reward_loss
            logger.info(f"{reward_type} training completed with loss: {reward_loss:.6f}")
            
            if args.use_wandb:
                wandb.log({
                    f"reward_model/{reward_type}_loss": reward_loss,
                    "reward_model/train_step": args.global_step
                })
                
        return total_metrics
        
    except Exception as e:
        logger.error("Error in reward model training")
        logger.error(traceback.format_exc())
        raise

def expand_model_gallery(models, train_dataloader, args):
    """Expand model gallery with samples from current model"""
    logger.info("Expanding model gallery with new samples...")
    
    try:
        base_model = models["unet"]
        base_model.eval()
        
        gallery_samples = []
        with torch.no_grad():
            for batch in train_dataloader:
                # Generate images
                prompts = batch["prompts"]
                generated = generate_images(models, prompts, args)
                
                # Extract features using model manager
                clip_features = model_manager.clip.extract_features(generated)
                object_detections = model_manager.detr.detect_objects(generated)
                
                # Calculate reward scores
                reward_scores = {}
                for reward_type, reward_model in models["reward_models"].items():
                    if reward_type == "attribute_binding":
                        scores = reward_model(clip_features, prompts)
                    elif reward_type == "spatial":
                        scores = reward_model(object_detections, prompts)
                    elif reward_type == "non_spatial":
                        scores = reward_model(clip_features, object_detections, prompts)
                        
                    reward_scores[reward_type] = scores.cpu()
                    logger.debug(f"{reward_type} scores mean: {scores.mean():.6f}")
                
                # Store samples and scores
                gallery_samples.append({
                    "images": generated.cpu(),
                    "prompts": prompts,
                    "clip_features": clip_features.cpu(),
                    "object_detections": object_detections,
                    "reward_scores": reward_scores
                })
                
                if args.use_wandb:
                    wandb.log({
                        f"gallery/reward_{reward_type}": scores.mean()
                        for reward_type, scores in reward_scores.items()
                    })
        
        # Update gallery dataset
        update_gallery_dataset(gallery_samples, args)
        logger.info("Successfully expanded model gallery")
        
    except Exception as e:
        logger.error("Error expanding model gallery")
        logger.error(traceback.format_exc())
        raise

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
    """Train with iterative compositional learning"""
    logger.info("Starting IterComp training")
    training_history = defaultdict(list)
    global_step = 0
    
    try:
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            models["unet"].parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # Initialize learning rate scheduler
        num_training_steps = len(train_dataloader) * args.num_epochs * args.num_iterations
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=args.min_learning_rate
        )
        
        for iteration in range(args.num_iterations):
            logger.info(f"\n=== Starting IterComp Iteration {iteration + 1}/{args.num_iterations} ===")
            iteration_metrics = defaultdict(float)
            
            # Train base model with reward guidance
            for epoch in range(args.num_epochs):
                epoch_metrics, global_step = train_one_epoch(
                    unet=models["unet"],
                    train_dataloader=train_dataloader,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema_model=None,
                    validator=None,
                    tag_weighter=None,
                    vae_finetuner=None,
                    args=args,
                    device=args.device,
                    dtype=args.dtype,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    train_components={"optimizer": optimizer, "lr_scheduler": lr_scheduler},
                    training_history=training_history
                )
                
                # Log metrics
                if args.use_wandb:
                    wandb.log({
                        f"itercomp/iteration": iteration,
                        **{f"itercomp/{k}": v for k, v in epoch_metrics.items()},
                        "global_step": global_step
                    })
                
                # Update iteration metrics
                for k, v in epoch_metrics.items():
                    iteration_metrics[k] += v
            
            # Update reward models if not final iteration
            if iteration < args.num_iterations - 1:
                update_reward_models(
                    models["reward_models"],
                    train_dataloader,
                    args,
                    iteration_metrics
                )
            
            # Save checkpoint
            if args.save_every_iteration:
                save_checkpoint(
                    models=models,
                    training_history=training_history,
                    iteration=iteration,
                    args=args
                )
            
            # Record iteration history
            training_history["iterations"].append({
                "iteration": iteration,
                "metrics": iteration_metrics,
                "global_step": global_step
            })
        
        logger.info("IterComp training completed successfully")
        return training_history, global_step
        
    except Exception as e:
        logger.error("Error in IterComp training")
        logger.error(traceback.format_exc())
        raise

def update_reward_models(reward_models, train_dataloader, args, metrics):
    """Update reward models with new preferences"""
    logger.info("Starting reward model update")
    
    try:
        for reward_type, reward_model in reward_models.items():
            logger.debug(f"Updating {reward_type} reward model")
            
            # Collect preferences using model manager features
            preference_pairs = collect_preferences(
                reward_type=reward_type,
                train_dataloader=train_dataloader,
                model_manager=model_manager,
                args=args
            )
            
            # Update model
            reward_loss = train_reward_model(
                reward_model=reward_model,
                preference_pairs=preference_pairs,
                args=args
            )
            
            logger.debug(f"{reward_type} reward model updated with loss: {reward_loss:.6f}")
            
            if args.use_wandb:
                wandb.log({
                    f"reward_model/{reward_type}_loss": reward_loss,
                })
                
    except Exception as e:
        logger.error("Error updating reward models")
        logger.error(traceback.format_exc())
        raise

def training_step(unet, batch, args, reward_models=None):
    """
    Single training step combining NAI V3 improvements with IterComp reward guidance
    """
    try:
        # Validate batch contents
        required_keys = ["latents", "text_embeddings"]
        for key in required_keys:
            if key not in batch:
                raise KeyError(f"Missing required key '{key}' in batch. Available keys: {batch.keys()}")

        # Get base diffusion loss with NAI V3 improvements
        base_loss, base_metrics = training_loss_v_prediction(
            model=unet,
            x_0=batch["latents"],
            sigma=batch.get("sigma", None),  # Handle missing sigma
            text_embeddings=batch["text_embeddings"],
            added_cond_kwargs=batch.get("added_cond_kwargs")
        )
        
        if base_loss is None:  # Skip batch if loss calculation failed
            logger.warning("Skipping batch due to invalid dimensions or aspect ratio")
            return None, {}

        total_loss = base_loss
        metrics = {"base_loss": base_loss.item()}
        metrics.update(base_metrics)
        
        # Add reward guidance if enabled
        if reward_models is not None:
            reward_loss, reward_metrics = compute_reward_guidance(
                reward_models=reward_models,
                batch=batch,
                args=args
            )
            if reward_loss is not None:
                total_loss = total_loss + args.reward_weight * reward_loss
                metrics.update(reward_metrics)
            
        return total_loss, metrics
        
    except Exception as e:
        logger.error("Error in training step")
        logger.error(f"Batch keys: {batch.keys()}")
        logger.error(traceback.format_exc())
        raise

def compute_reward_guidance(reward_models, batch, args):
    """Compute reward guidance loss if possible"""
    try:
        if "images" not in batch or "prompts" not in batch:
            return None, {}

        # Extract features once for all reward models
        clip_features = model_manager.clip.extract_features(
            batch["images"], 
            text=batch["prompts"]
        )["image_features"]
        
        object_detections = model_manager.detr.detect_objects(batch["images"])
        text_embeddings = batch["text_embeddings"]
        
        # Calculate rewards for each aspect
        reward_losses = {}
        reward_metrics = {}
        
        for reward_type, reward_model in reward_models.items():
            if isinstance(reward_model, AttributeBindingRewardModel):
                reward = reward_model(clip_features, text_embeddings)
            elif isinstance(reward_model, SpatialRewardModel):
                reward = reward_model(object_detections, text_embeddings)
            else:  # NonSpatialRewardModel
                reward = reward_model(clip_features, object_detections, text_embeddings)
            
            reward_losses[reward_type] = reward
            reward_metrics[f"{reward_type}_reward"] = reward.item()
        
        # Combine rewards
        total_reward_loss = sum(reward_losses.values()) / len(reward_losses)
        reward_metrics["reward_loss"] = total_reward_loss.item()
        
        return total_reward_loss, reward_metrics
        
    except Exception as e:
        logger.warning(f"Failed to compute reward guidance: {str(e)}")
        return None, {}

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