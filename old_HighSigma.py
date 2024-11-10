import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import logging
import argparse
import math
from PIL import Image
from bitsandbytes.optim import AdamW8bit
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers.optimization import Adafactor
import torchvision.models as models
from models.model_validator import ModelValidator
import torchvision.transforms.functional as TF
import os
import wandb
import json
import traceback
from collections import defaultdict
import time
import shutil
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# Global bfloat16 settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, sigma_max=20000.0):
    """
    Generate sigmas using a schedule that supports Zero Terminal SNR (ZTSNR)
    Args:
        num_inference_steps: Number of inference steps
        sigma_min: Minimum sigma value (≈0.0292 from paper)
        sigma_max: Maximum sigma value (set to 20000 for practical ZTSNR)
    Returns:
        Tensor of sigma values
    """
    # Use rho=7.0 as specified in the paper
    rho = 7.0  
    t = torch.linspace(1, 0, num_inference_steps)
    # Karras schedule with ZTSNR modifications
    sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigmas

def v_prediction_scaling_factors(sigma, sigma_data=1.0):
    """
    Compute scaling factors for v-prediction as described in paper section 2.1
    """
    # α_t = 1/√(1 + σ²) from paper
    alpha_t = 1 / torch.sqrt(1 + sigma**2)
    
    # Scaling factors from paper appendix A.1
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = -sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return alpha_t, c_skip, c_out, c_in

def get_ztsnr_schedule(num_steps=28, sigma_min=0.0292, sigma_max=20000.0, rho=7.0):
    """
    Generate ZTSNR noise schedule as described in paper section 2.2
    Using practical implementation from appendix A.2
    """
    t = torch.linspace(1, 0, num_steps)
    sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigmas



def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs):
    """
    Training loss using v-prediction with MinSNR weighting (sections 2.1 and 2.4)
    
    Args:
        unet (UNet2DConditionModel): The model being trained
        batch (dict): Training batch containing:
            - latents: [B, 4, H, W] latent representations
            - text_embeddings: [B, 77, 768] CLIP-L embeddings
            - text_embeddings_2: [B, 77, 1280] CLIP-G embeddings
            - pooled_text_embeddings_2: [B, 1280] Pooled CLIP-G embeddings
            - tags: List of image tags for weighting
        args: Training arguments
        device: Target device
        step (int): Current step number
        sigmas (tensor): Noise schedule
        optimizer: Model optimizer
        lr_scheduler: Learning rate scheduler
        ema_model: EMA model instance
        tag_weighter: Tag-based loss weighting instance
        vae_finetuner: VAE finetuning instance
        dtype: Training precision (default: bfloat16)
    
    Returns:
        dict: Step metrics including loss values and model stats
   
    """
    try:
        logger.debug("\n=== Starting v-prediction training loss calculation ===")
        logger.debug("Initial input shapes and values:")
        logger.debug(f"x_0: shape={x_0.shape}, dtype={x_0.dtype}, device={x_0.device}")
        logger.debug(f"sigma: shape={sigma.shape}, dtype={sigma.dtype}, range=[{sigma.min():.6f}, {sigma.max():.6f}]")
        logger.debug(f"text_embeddings: shape={text_embeddings.shape}, dtype={text_embeddings.dtype}")
        
        # Get noise and scaling factors
        logger.debug("\nGenerating noise and computing scaling factors:")
        noise = torch.randn_like(x_0)
        logger.debug(f"noise: shape={noise.shape}, dtype={noise.dtype}, std={noise.std():.6f}")
        
        alpha_t, c_skip, c_out, c_in = v_prediction_scaling_factors(sigma)
        logger.debug(f"alpha_t: range=[{alpha_t.min():.6f}, {alpha_t.max():.6f}]")
        logger.debug(f"c_skip: range=[{c_skip.min():.6f}, {c_skip.max():.6f}]")
        logger.debug(f"c_out: range=[{c_out.min():.6f}, {c_out.max():.6f}]")
        logger.debug(f"c_in: range=[{c_in.min():.6f}, {c_in.max():.6f}]")
        
        # Compute noisy sample x_t = x_0 + σε
        logger.debug("\nComputing noisy sample:")
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        logger.debug(f"x_t: shape={x_t.shape}, range=[{x_t.min():.6f}, {x_t.max():.6f}]")
        
        # Compute v-target = α_t * ε - (1 - α_t) * x_0
        logger.debug("\nComputing v-target:")
        v_target = alpha_t.view(-1, 1, 1, 1) * noise - (1 - alpha_t).view(-1, 1, 1, 1) * x_0
        logger.debug(f"v_target: shape={v_target.shape}, range=[{v_target.min():.6f}, {v_target.max():.6f}]")
        
        # Get model prediction
        logger.debug("\nGetting model prediction:")
        v_pred = model(
            x_t,
            sigma,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        logger.debug(f"v_pred: shape={v_pred.shape}, range=[{v_pred.min():.6f}, {v_pred.max():.6f}]")
        
        # MinSNR weighting
        logger.debug("\nComputing MinSNR weights:")
        snr = 1 / (sigma**2)  # SNR = 1/σ²
        gamma = 1.0  # SNR clipping value
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        logger.debug(f"SNR: range=[{snr.min():.6f}, {snr.max():.6f}]")
        logger.debug(f"min_snr_gamma: range=[{min_snr_gamma.min():.6f}, {min_snr_gamma.max():.6f}]")
        
        # Compute weighted MSE loss
        logger.debug("\nComputing final loss:")
        mse_loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = (min_snr_gamma.view(-1, 1, 1, 1) * mse_loss).mean()
        
        # Collect detailed metrics
        loss_metrics = {
            'loss/total': loss.item(),
            'loss/mse_mean': mse_loss.mean().item(),
            'loss/mse_std': mse_loss.std().item(),
            'loss/snr_mean': snr.mean().item(),
            'loss/min_snr_gamma_mean': min_snr_gamma.mean().item(),
            'model/v_pred_std': v_pred.std().item(),
            'model/v_target_std': v_target.std().item(),
            'model/alpha_t_mean': alpha_t.mean().item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item()
        }
        
        logger.debug("Loss metrics:")
        for key, value in loss_metrics.items():
            logger.debug(f"{key}: {value:.6f}")
        
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Log state of variables at time of error
        logger.error("\nVariable state at error:")
        local_vars = locals()
        for name, value in local_vars.items():
            if isinstance(value, torch.Tensor):
                try:
                    logger.error(f"{name}: shape={value.shape}, dtype={value.dtype}, "
                               f"range=[{value.min():.6f}, {value.max():.6f}]")
                except:
                    logger.error(f"{name}: <tensor stats unavailable>")
            else:
                logger.error(f"{name}: type={type(value)}")
        raise


class PerceptualLoss:
    def __init__(self):
        # Use pre-trained VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to("cuda")
        # Convert VGG to bfloat16
        self.vgg = self.vgg.to(dtype=torch.bfloat16)
        self.vgg.requires_grad_(False)
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def get_features(self, x):
        # Ensure input is in bfloat16
        x = x.to(dtype=torch.bfloat16)
        x = self.normalize(x)
        features = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def __call__(self, pred, target):
        # Ensure inputs are in bfloat16
        pred = pred.to(dtype=torch.bfloat16)
        target = target.to(dtype=torch.bfloat16)
        
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)

        loss = 0.0
        for key in pred_features:
            loss += F.mse_loss(pred_features[key], target_features[key])
        return loss

def custom_collate(batch):
    """
    Custom collate function for DataLoader that handles both single and batched samples.
    
    Args:
        batch: List of dictionaries containing dataset items
            For batch_size=1: List with single dictionary
            For batch_size>1: List of multiple dictionaries
    
    Returns:
        For batch_size=1: Original dictionary without any stacking
        For batch_size>1: Dictionary with properly stacked tensors and lists
    """
    # Get batch size from first item
    batch_size = len(batch)
    
    if batch_size == 1:
        # For single samples, return the dictionary directly without any stacking
        return batch[0]
    else:
        # For batched samples, we need to stack the tensors properly
        elem = batch[0]  # Get first item to determine dictionary structure
        collated = {}
        
        for key in elem:
            if key == "tags":
                # Tags are lists of strings, so we keep them as a list of lists
                collated[key] = [d[key] for d in batch]
            elif key == "target_size":
                # target_size needs to remain as separate tensors for each batch item
                collated[key] = [d[key] for d in batch]
            else:
                try:
                    # Try to stack tensors along a new batch dimension
                    collated[key] = torch.stack([d[key] for d in batch])
                except:
                    # If stacking fails, keep as list (for non-tensor data)
                    collated[key] = [d[key] for d in batch]
        
        return collated




def setup_models(args, device, dtype):
    """Initialize and configure all models"""
    logger.info("Setting up models...")
    
    models = {
        "unet": UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=dtype
        ).to(device),
        
        "vae": AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=dtype
        ),
        
        "tokenizer": CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer"
        ),
        
        "tokenizer_2": CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer_2"
        ),
        
        "text_encoder": CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
            torch_dtype=dtype
        ),
        
        "text_encoder_2": CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype
        )
    }
    
    # Enable optimizations
    models["unet"].enable_gradient_checkpointing()
    models["unet"].enable_xformers_memory_efficient_attention()
    models["vae"].enable_xformers_memory_efficient_attention()
    
    return models

def setup_training(args, models, device, dtype):
    """Initialize all training components"""
    logger.info("Setting up training components...")
    
    # Create dataset and dataloader
    dataset = CustomDataset(
        args.data_dir,
        models["vae"],
        models["tokenizer"],
        models["tokenizer_2"],
        models["text_encoder"],
        models["text_encoder_2"],
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # Initialize optimizer
    optimizer = (
        Adafactor(
            models["unet"].parameters(),
            lr=args.learning_rate * args.batch_size,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False
        ) if args.use_adafactor else
        AdamW8bit(
            models["unet"].parameters(),
            lr=args.learning_rate * args.batch_size,
            betas=(0.9, 0.999)
        )
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = args.num_epochs * num_update_steps_per_epoch

    # Setup other components
    return {
        "train_dataloader": train_dataloader,
        "optimizer": optimizer,
        "lr_scheduler": get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        ),
        "ema_model": AveragedModel(
            models["unet"],
            avg_fn=lambda avg, new, _: args.ema_decay * avg + (1 - args.ema_decay) * new
        ),
        "validator": ModelValidator(
            model=models["unet"],
            vae=models["vae"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            device=device
        ),
        "tag_weighter": TagBasedLossWeighter(
            min_weight=args.min_tag_weight,
            max_weight=args.max_tag_weight
        ),
        "vae_finetuner": VAEFineTuner(
            vae=models["vae"],
            learning_rate=args.vae_learning_rate
        ) if args.finetune_vae else None
    }

def save_checkpoint(models, train_components, args, epoch, training_history, output_dir):
    """
    Save a training checkpoint including models, optimizer state, and training history
    
    Args:
        models (dict): Dictionary containing all models:
            - unet: UNet2DConditionModel
            - vae: AutoencoderKL
            - text_encoder: CLIPTextModel
            - text_encoder_2: CLIPTextModel
        train_components (dict): Dictionary containing training components:
            - optimizer: Optimizer
            - lr_scheduler: LRScheduler
            - ema_model: EMA model
        args: Training arguments
        epoch (int): Current epoch number
        training_history (dict): Dictionary containing training metrics
        output_dir (str): Base directory for saving checkpoints
    """
    try:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_{epoch:04d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save models with diffusers-compatible format
        for name, model in models.items():
            if hasattr(model, 'save_pretrained'):
                model_dir = os.path.join(checkpoint_dir, name)
                logger.info(f"Saving {name} to {model_dir}")
                model.save_pretrained(model_dir, safe_serialization=True)
        
        # Save optimizer state
        optimizer = train_components["optimizer"]
        torch.save(
            optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt")
        )
        
        # Save scheduler state
        lr_scheduler = train_components["lr_scheduler"]
        torch.save(
            lr_scheduler.state_dict(),
            os.path.join(checkpoint_dir, "scheduler.pt")
        )
        
        # Save EMA model if present
        if "ema_model" in train_components:
            ema_model = train_components["ema_model"]
            torch.save(
                ema_model.state_dict(),
                os.path.join(checkpoint_dir, "ema.pt")
            )
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "training_history": training_history,
            "args": vars(args)  # Convert args to dict for saving
        }
        torch.save(
            training_state,
            os.path.join(checkpoint_dir, "training_state.pt")
        )
        
        logger.info("Checkpoint saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def train(args, models, train_components, device, dtype):
    """
    Main training loop with comprehensive progress tracking and validation
    
    Args:
        args: Training arguments including:
            - num_epochs: Number of training epochs
            - batch_size: Batch size per device
            - learning_rate: Initial learning rate
            - validation_frequency: Epochs between validations
            - logging_steps: Steps between metric logging
            - save_epochs: Epochs between checkpoints
        models: Dictionary containing:
            - unet: UNet2DConditionModel
            - vae: AutoencoderKL
            - tokenizer: CLIPTokenizer
            - tokenizer_2: CLIPTokenizer
            - text_encoder: CLIPTextModel
            - text_encoder_2: CLIPTextModel
        train_components: Dictionary containing:
            - train_dataloader: DataLoader
            - optimizer: Optimizer
            - lr_scheduler: LRScheduler
            - ema_model: EMA model
            - validator: ModelValidator
            - tag_weighter: TagBasedLossWeighter
            - vae_finetuner: VAEFineTuner (optional)
        device: Target device (cuda/cpu)
        dtype: Training precision (typically bfloat16)
            
    Returns:
        dict: Training history containing:
            - epoch_losses: List of average losses per epoch
            - learning_rates: Learning rate schedule
            - grad_norms: Gradient norms during training
            - validation_scores: Validation metrics
    """
    
    # Unpack models and components for cleaner access
    unet = models["unet"]
    train_dataloader = train_components["train_dataloader"]
    optimizer = train_components["optimizer"]
    lr_scheduler = train_components["lr_scheduler"]
    ema_model = train_components["ema_model"]
    validator = train_components["validator"]
    tag_weighter = train_components["tag_weighter"]
    vae_finetuner = train_components.get("vae_finetuner")

    # Initialize training state tracking
    training_history = {
        'epoch_losses': [],
        'learning_rates': [],
        'grad_norms': [],
        'validation_scores': [],
        'step_times': []
    }
    
    total_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    logger.info("\n=== Starting Training ===")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    for epoch in range(args.num_epochs):
        # Initialize epoch tracking
        epoch_metrics = defaultdict(float)
        epoch_start_time = time.time()
        step_times = []
        
        logger.info(f"\n=== Starting Epoch {epoch+1}/{args.num_epochs} ===")
        
        for step, batch in enumerate(train_dataloader):
            step_start = time.time()
            global_step = epoch * len(train_dataloader) + step
            
            # === Process Batch ===
            # Shape: [batch_size, 4, 64, 64]
            latents = batch["latents"].to(device).to(dtype=torch.bfloat16)
            
            # Shape: [batch_size, 77, 768] - First encoder (CLIP-L)
            text_embeddings_1 = batch["text_embeddings"].to(device).to(dtype=torch.bfloat16)
            
            # Shape: [batch_size, 77, 1280] - Second encoder (CLIP-G)
            text_embeddings_2 = batch["text_embeddings_2"].to(device).to(dtype=torch.bfloat16)
            
            # Shape: [batch_size, 77, 2048] - Combined embeddings
            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
            
            # Shape: [batch_size, 1280] - Pooled embeddings
            pooled_embeds = batch["pooled_text_embeddings_2"].to(device).to(dtype=torch.bfloat16)
            
            # === Training Step ===
            with torch.amp.autocast('cuda', dtype=dtype):
                # Shape: [num_inference_steps]
                sigmas = get_sigmas(args.num_inference_steps).to(device)
                sigma = sigmas[step % args.num_inference_steps]
                sigma = sigma.expand(latents.size(0))
                
                # Run training step
                loss, step_metrics = training_loss_v_prediction(
                    unet,
                    latents,  # [B, 4, 64, 64]
                    sigma,    # [B]
                    text_embeddings,  # [B, 77, 2048]
                    {
                        "text_embeds": pooled_embeds,  # [B, 1280]
                        "time_ids": torch.tensor([
                            1024, 1024,  # Original dims
                            1024, 1024,  # Target dims
                            0, 0,        # Crop coords
                        ], device=device, dtype=dtype).repeat(latents.shape[0], 1)
                    }
                )
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
            
            # === Update Model ===
            weighted_loss = tag_weighter.update_training_loss(loss, batch["tags"])
            weighted_loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients and update model
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), 
                    args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema_model.update_parameters(unet)
                
                # Track gradient norm
                training_history['grad_norms'].append(grad_norm.item())
                
                # VAE finetuning step if enabled
                if vae_finetuner and (global_step % args.vae_train_freq == 0):
                    logger.debug("Running VAE finetuning step")
                    try:
                        vae_loss = vae_finetuner.training_step(
                            batch["images"].to(device),  # Original images
                            batch["latents"].to(device)  # Encoded latents
                        )
                        epoch_metrics['vae_loss'] += vae_loss
                        
                        if args.use_wandb:
                            wandb.log({
                                'vae_loss': vae_loss
                            }, step=global_step)
                            
                    except Exception as e:
                        logger.error(f"VAE finetuning step failed: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")

            
            progress_bar.update(1)
        
        # === End of Epoch ===
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch metrics
        logger.info(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        logger.info("Average metrics:")
        for k, v in epoch_metrics.items():
            avg_value = v / len(train_dataloader)
            logger.info(f"- {k}: {avg_value:.4f}")
        
        # Update training history
        training_history['epoch_losses'].append(
            epoch_metrics['loss/total'] / len(train_dataloader)
        )
        training_history['learning_rates'].append(
            lr_scheduler.get_last_lr()[0]
        )
        training_history['step_times'].extend(step_times)
        
        # Run validation if scheduled
        if (epoch + 1) % args.validation_frequency == 0:
            logger.info("\nRunning validation...")
            validation_metrics = validator.run_paper_validation(
                args.validation_prompts
            )
            training_history['validation_scores'].append(validation_metrics)
            
            if args.use_wandb:
                wandb.log({
                    'validation': validation_metrics
                }, step=global_step)
        
        # Save checkpoint if scheduled
        if (epoch + 1) % args.save_epochs == 0:
            logger.info("\nSaving checkpoint...")
            save_checkpoint(
                unet, optimizer, lr_scheduler, epoch, 
                training_history, args.output_dir
            )
    
    progress_bar.close()
    return training_history

def save_training_config(args, output_dir, training_history):
    """Save training configuration and history"""
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f)

def create_model_card(args, training_history):
    """
    Create a detailed model card for the trained model following HuggingFace's best practices
    
    Args:
        args: Training arguments containing model configuration and training settings
        training_history (dict): Dictionary containing training metrics and history
        
    Returns:
        str: Formatted markdown text for the model card
    """
    try:
        # Get final metrics
        final_loss = training_history['epoch_losses'][-1] if training_history['epoch_losses'] else None
        final_validation = training_history['validation_scores'][-1] if training_history['validation_scores'] else None
        
        # Format training parameters
        training_params = {
            "Learning Rate": args.learning_rate,
            "Batch Size": args.batch_size,
            "Number of Epochs": args.num_epochs,
            "Gradient Accumulation Steps": args.gradient_accumulation_steps,
            "Max Gradient Norm": args.max_grad_norm,
            "EMA Decay": args.ema_decay,
            "Optimizer": "Adafactor" if args.use_adafactor else "AdamW8bit"
        }
        
        # Create markdown content
        model_card = f"""
# {os.path.basename(args.model_path)} Fine-tuned SDXL Model

This is a fine-tuned version of [{args.model_path}](https://huggingface.co/{args.model_path}) trained with High Sigma training techniques.

## Model Details

- **Base Model**: {args.model_path}
- **Training Type**: High Sigma Fine-tuning
- **Training Data**: Custom dataset from {args.data_dir}
- **Developed by**: [Your Organization]
- **Trained by**: [Your Name/Organization]
- **License**: [License Type]

## Training Details

### Training Parameters
"""

        # Add training parameters
        for param, value in training_params.items():
            model_card += f"- **{param}**: {value}\n"
        
        # Add performance metrics if available
        if final_loss or final_validation:
            model_card += "\n### Performance Metrics\n"
            if final_loss:
                model_card += f"- **Final Training Loss**: {final_loss:.4f}\n"
            if final_validation:
                model_card += "- **Validation Results**:\n"
                for metric, value in final_validation.items():
                    if isinstance(value, (int, float)):
                        model_card += f"  - {metric}: {value:.4f}\n"
                    else:
                        model_card += f"  - {metric}: {value}\n"

        # Add training curves if using wandb
        if args.use_wandb:
            model_card += """
### Training Curves
Training curves and additional metrics are available in the [Weights & Biases run](INSERT_WANDB_RUN_LINK).
"""

        # Add usage information
        model_card += """
## Usage

This model can be used with the standard Stable Diffusion XL pipeline. Here's an example:
```python
from diffusers import StableDiffusionXLPipeline
import torch
model_id = "path/to/model"
pipe = StableDiffusionXLPipeline.from_pretrained(
model_id,
torch_dtype=torch.float16,
use_safetensors=True
).to("cuda")
prompt = "your prompt here"
image = pipe(prompt).images[0]
```

## Training Process

This model was trained using High Sigma training techniques, which includes:
- Training with high noise levels (σ ≈ 20000)
- Zero Terminal SNR optimization
- High-resolution coherence preservation
- Tag-based loss weighting

## Limitations & Biases

Please note:
- This model inherits biases from its base model and training data
- The model's performance may vary depending on the prompt complexity
- [Add any specific limitations or known issues]

## License
[Specify License Information]

## Citation
If you use this model in your research, please cite:
```
bibtex
@misc{your-model-name,
author = {Your Name},
title = {Model Name},
year = {2024},
publisher = {Hugging Face},
journal = {Hugging Face Model Hub},
}
```
"""

        return model_card
        
    except Exception as e:
        logger.error(f"Failed to create model card: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return """
# Model Card Generation Failed

There was an error generating the complete model card. Please check the logs for details.

## Basic Information
- **Base Model**: {args.model_path}
- **Training Type**: High Sigma Fine-tuning
"""

def save_model_card(model_card, output_dir):
    """
    Save the model card to the output directory
    
    Args:
        model_card (str): The formatted model card markdown text
        output_dir (str): Directory to save the model card
    """
    try:
        card_path = os.path.join(output_dir, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        logger.info(f"Model card saved to {card_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model card: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def push_to_hub(
    hub_model_id,
    output_dir,
    private=False,
    model_card=None,
    commit_message=None
):
    """
    Push the trained model to HuggingFace Hub
    
    Args:
        hub_model_id (str): Full model ID on HuggingFace (e.g., 'username/model-name')
        output_dir (str): Local directory containing the model files
        private (bool): Whether to create a private repository
        model_card (str, optional): Model card content in markdown format
        commit_message (str, optional): Custom commit message
    """
    try:
        from huggingface_hub import (
            HfApi, 
            create_repo,
            upload_folder
        )
        
        logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
        api = HfApi()
        
        # Create or get repository
        try:
            logger.info("Creating repository...")
            create_repo(
                hub_model_id,
                private=private,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repository creation warning (might already exist): {e}")
        
        # Prepare commit message
        if commit_message is None:
            commit_message = f"Upload {hub_model_id} with High Sigma training"
        
        # Save model card if provided
        if model_card is not None:
            card_path = os.path.join(output_dir, "README.md")
            logger.info(f"Saving model card to {card_path}")
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(model_card)
        
        # Upload the model
        logger.info(f"Uploading files from {output_dir}")
        upload_folder(
            folder_path=output_dir,
            repo_id=hub_model_id,
            commit_message=commit_message,
            ignore_patterns=[
                "*.pt",          # Ignore pytorch state files
                "*.bin",         # Prefer safetensors
                "checkpoint_*",  # Ignore intermediate checkpoints
                "events.out.*",  # Ignore tensorboard files
                "*.ckpt",       # Ignore old checkpoint format
                "*.pth",        # Ignore other pytorch files
                "__pycache__",  # Ignore python cache
                ".git",         # Ignore git files
                "wandb",        # Ignore wandb files
                "logs",         # Ignore log files
                "cache"         # Ignore cache directories
            ]
        )
        
        # Get repository URL
        repo_url = f"https://huggingface.co/{hub_model_id}"
        logger.info(f"Model successfully pushed to {repo_url}")
        
        return repo_url
        
    except ImportError as e:
        logger.error("huggingface_hub package not found. Please install with:")
        logger.error("pip install huggingface_hub")
        raise
        
    except Exception as e:
        logger.error(f"Failed to push to hub: {str(e)}")
        logger.error(f"Model ID: {hub_model_id}")
        logger.error(f"Output directory: {output_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def verify_hub_token():
    """
    Verify that a HuggingFace token is available and valid
    
    Returns:
        bool: True if token is valid
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.whoami()
        return True
        
    except ImportError:
        logger.error("huggingface_hub package not found")
        return False
        
    except Exception as e:
        logger.error(f"Invalid or missing HuggingFace token: {e}")
        logger.error("Please run `huggingface-cli login` or set the HUGGING_FACE_HUB_TOKEN environment variable")
        return False

def save_final_outputs(args, models, training_history):
    """Save final model, config and training history"""
    logger.info("Saving final outputs...")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    for name, model in models.items():
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(
                os.path.join(output_dir, name),
                safe_serialization=True
            )

    # Save config and history
    save_training_config(args, output_dir, training_history)
    
    # Create model card if pushing to hub
    if args.push_to_hub:
        model_card = create_model_card(args, training_history)
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)


def verify_checkpoint_directory(checkpoint_dir):
    """
    Verify that a checkpoint directory contains all required model directories in diffusers format
    
    Args:
        checkpoint_dir (str): Directory to verify
        
    Returns:
        bool, dict: Tuple containing:
            - bool: True if directory contains all required model directories
            - dict: Status of optional files
    """
    # Optional state files - used for training resumption
    optional_files = [
        "training_state.pt",
        "optimizer.pt",
        "scheduler.pt",
        "ema.safetensors",
        "ema.pt"
    ]
    
    # Required model directories - needed for model loading
    required_dirs = [
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2"
    ]
    
    # Check for either safetensors or pt files in model directories
    def has_model_files(model_dir):
        return (
            os.path.exists(os.path.join(model_dir, "model.safetensors")) or
            os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
        )
    
    # Track optional file status
    optional_status = {
        f: os.path.exists(os.path.join(checkpoint_dir, f))
        for f in optional_files
    }
    
    # Check required directories and their contents
    missing_dirs = []
    for d in required_dirs:
        dir_path = os.path.join(checkpoint_dir, d)
        if not os.path.exists(dir_path):
            missing_dirs.append(d)
        elif not has_model_files(dir_path):
            missing_dirs.append(f"{d} (no model files)")
    
    if missing_dirs:
        logger.warning("Checkpoint directory missing required model directories:")
        logger.warning(f"Missing or invalid directories: {missing_dirs}")
        return False, optional_status
    
    # Log optional file status
    logger.info("Optional file status:")
    for f, exists in optional_status.items():
        logger.info(f"- {f}: {'Found' if exists else 'Missing'}")
    
    return True, optional_status

def load_checkpoint(checkpoint_dir, models, train_components):
    """
    Load a saved checkpoint in diffusers format with safetensors support
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoint
        models (dict): Dictionary containing models to load
        train_components (dict): Dictionary containing training components
            
    Returns:
        dict: Training state (if available) or None
    """
    try:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Verify directory structure
        is_valid, optional_status = verify_checkpoint_directory(checkpoint_dir)
        if not is_valid:
            raise ValueError("Invalid checkpoint directory structure")
        
        # Load models (required)
        model_components = {
            "unet": ("unet", UNet2DConditionModel),
            "vae": ("vae", AutoencoderKL),
            "text_encoder": ("text_encoder", CLIPTextModel),
            "text_encoder_2": ("text_encoder_2", CLIPTextModel)
        }
        
        for model_key, (subfolder, model_class) in model_components.items():
            model_dir = os.path.join(checkpoint_dir, subfolder)
            logger.info(f"Loading {model_key} from {model_dir}")
            models[model_key] = model_class.from_pretrained(
                model_dir,
                use_safetensors=True,
                torch_dtype=models[model_key].dtype
            )
        
        # Load training state if available
        training_state = None
        if optional_status["training_state.pt"]:
            logger.info("Loading training state")
            training_state = torch.load(
                os.path.join(checkpoint_dir, "training_state.pt"),
                map_location='cpu'
            )
            
            # Load optimizer state if available
            if optional_status["optimizer.pt"]:
                logger.info("Loading optimizer state")
                train_components["optimizer"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "optimizer.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load scheduler state if available
            if optional_status["scheduler.pt"]:
                logger.info("Loading scheduler state")
                train_components["lr_scheduler"].load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "scheduler.pt"),
                        map_location='cpu'
                    )
                )
            
            # Load EMA state if available
            if "ema_model" in train_components:
                if optional_status["ema.safetensors"]:
                    logger.info("Loading EMA state from safetensors")
                    from safetensors.torch import load_file
                    ema_state = load_file(
                        os.path.join(checkpoint_dir, "ema.safetensors")
                    )
                    train_components["ema_model"].load_state_dict(ema_state)
                elif optional_status["ema.pt"]:
                    logger.info("Loading EMA state from pytorch")
                    train_components["ema_model"].load_state_dict(
                        torch.load(
                            os.path.join(checkpoint_dir, "ema.pt"),
                            map_location='cpu'
                        )
                    )
        
        return training_state
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        logger.error(f"Checkpoint directory: {checkpoint_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--finetune_vae", action="store_true", help="Enable VAE finetuning")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6, help="VAE learning rate")
    parser.add_argument("--vae_train_freq", type=int, default=10, help="VAE training frequency")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--use_adafactor", action="store_true", help="Use Adafactor instead of AdamW8bit")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients"
    )
    # Removed compile_mode argument
    parser.add_argument("--enable_compile", action="store_true", help="Enable model compilation")
    parser.add_argument("--compile_mode", type=str, choices=['default', 'reduce-overhead', 'max-autotune'], default='default', help="Torch compile mode")
    
    parser.add_argument("--save_checkpoints", action="store_true", help="Save checkpoints after each epoch")
    parser.add_argument("--min_tag_weight", type=float, default=0.1, help="Minimum tag-based loss weight")
    parser.add_argument("--max_tag_weight", type=float, default=3.0, help="Maximum tag-based loss weight")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HuggingFace Hub model ID")
    parser.add_argument("--hub_private", action="store_true", help="Make the HuggingFace repo private")
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=1,  # Run validation every epoch by default
        help="Number of epochs between validation runs"
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=[
            "a detailed portrait of a girl",
            "completely black",
            "a red ball on top of a blue cube, both infront of a green triangle"
        ],
        help="Prompts to use for validation"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip validation entirely"
    )

    args = parser.parse_args()
    return args


def main(args):
    """
    Main entry point for training
    
    Args:
        args: Parsed command line arguments including:
            - use_wandb: Enable Weights & Biases logging
            - wandb_project: W&B project name
            - wandb_run_name: W&B run name
            - model_path: Path to base model
            - output_dir: Directory for saving outputs
            - push_to_hub: Whether to push to HuggingFace Hub
    """
    wandb_run = None
    try:
        # === Setup Logging ===
        logger.info("\n=== Starting Training Pipeline ===")
        logger.info(f"Arguments: {args}")
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                resume=True  # Enable run resumption
            )
            logger.info(f"Initialized W&B run: {wandb_run.name}")

        # === Setup Environment ===
        # Set up device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        logger.info(f"Using device: {device}")
        logger.info(f"Using dtype: {dtype}")

        # Clear CUDA cache and set memory allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                logger.info(f"Initial CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        # === Model Setup ===
        logger.info("\nSetting up models and components...")
        
        # Load and setup models
        models = setup_models(args, device, dtype)
        logger.info("Models loaded successfully")
        
        # Initialize training components
        train_components = setup_training(args, models, device, dtype)
        logger.info("Training components initialized")
        
        # Load checkpoint if resuming
        if args.resume_from_checkpoint:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
            training_state = load_checkpoint(
                args.resume_from_checkpoint,
                models,
                train_components
            )
            start_epoch = training_state["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 0
            
        # === Training Loop ===
        logger.info("\nStarting training...")
        training_history = train(
            args=args,
            models=models,
            train_components=train_components,
            device=device,
            dtype=dtype
        )
        logger.info("Training completed successfully")

        # === Save Outputs ===
        logger.info("\nSaving final outputs...")
        save_final_outputs(
            args=args,
            models=models,
            training_history=training_history,
            train_components=train_components
        )
        
        # Create and save model card
        model_card = create_model_card(args, training_history)
        save_model_card(model_card, args.output_dir)
        
        # Push to Hub if requested
        if args.push_to_hub:
            logger.info("\nPushing to HuggingFace Hub...")
            push_to_hub(
                args.hub_model_id,
                args.output_dir,
                args.hub_private,
                model_card
            )

        logger.info("\n=== Training Pipeline Completed Successfully ===")
        return True

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        # Save emergency checkpoint
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        logger.info(f"Saving emergency checkpoint to {emergency_dir}")
        try:
            save_checkpoint(
                models,
                train_components,
                args,
                -1,  # Special epoch number for interrupted training
                training_history,
                emergency_dir
            )
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        return False

    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    finally:
        # Cleanup
        try:
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'memory_stats'):
                    logger.info(f"Final CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            # Close wandb run if it exists
            if wandb_run is not None:
                wandb_run.finish()
            
            # Remove temporary files
            if hasattr(args, 'cache_dir') and os.path.exists(args.cache_dir):
                logger.info(f"Cleaning up cache directory: {args.cache_dir}")
                shutil.rmtree(args.cache_dir, ignore_errors=True)
                
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    success = main(args)
    sys.exit(0 if success else 1)