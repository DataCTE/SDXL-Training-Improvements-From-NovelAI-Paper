import os
import math
import time
import logging
import traceback
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from training import get_sigmas, training_loss_v_prediction
from utils import save_checkpoint, load_checkpoint
from utils.logging import log_metrics_batch

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    tag_weighter: Optional[Any],
    device: torch.device,
    dtype: torch.dtype,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    mixed_precision: bool = True,
    min_snr_gamma: float = 5.0,
    sigma_data: float = 1.0,
    use_ztsnr: bool = True,
    rescale_cfg: bool = True,
    rescale_multiplier: float = 0.7,
    scale_method: str = "karras",
    resolution_scaling: bool = True,
    use_tag_weighting: bool = True,
    vae_finetuner: Optional[Any] = None,
    vae_train_freq: int = 10,
    ema_model: Optional[Any] = None,
    verbose: bool = False
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch with improved v-prediction, ZTSNR, VAE finetuning and EMA support"""
    model.train()
    scaler = GradScaler(enabled=mixed_precision)
    metrics = {
        'batch_time': AverageMeter('Batch Time', ':6.3f'),
        'data_time': AverageMeter('Data Loading Time', ':6.3f'),
        'loss': AverageMeter('Training Loss', ':.4e'),
        'vae_loss': AverageMeter('VAE Loss', ':.4e') if vae_finetuner else None,
        'grad_norm': AverageMeter('Gradient Norm', ':.4e'),
        'lr': AverageMeter('Learning Rate', ':.2e'),
        'bucket_size': AverageMeter('Bucket Size', ':6.0f'),
        'ema_decay': AverageMeter('EMA Decay', ':.4e') if ema_model else None
    }
    
    end = time.time()
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(train_dataloader):
        try:
            # Log input tensor shapes and bucket information
            if verbose:
                logger.info(f"\nBatch {batch_idx}/{len(train_dataloader)}:")
                bucket_size = batch.get('bucket_size', (None, None))
                logger.info(f"Bucket size (h,w): {bucket_size}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"Input {k}: shape={v.shape}, dtype={v.dtype}")

            metrics['data_time'].update(time.time() - end)
            
            # Device transfer with shape logging and prefetch next batch
            latents = batch['latents'].to(device, dtype=dtype, non_blocking=True)
            text_embeddings = batch['text_embeddings'].to(device, dtype=dtype, non_blocking=True)
            added_cond_kwargs = {k: v.to(device, dtype=dtype, non_blocking=True) 
                               if torch.is_tensor(v) else v 
                               for k, v in batch.get('added_cond_kwargs', {}).items()}
            
            # Start VAE finetuning step in parallel with UNet step
            vae_future = None
            if vae_finetuner and batch_idx % vae_train_freq == 0:
                # Create a CUDA stream for VAE computation
                vae_stream = torch.cuda.Stream()
                with torch.cuda.stream(vae_stream):
                    vae_loss = vae_finetuner.training_step(batch)
                    metrics['vae_loss'].update(vae_loss.item())
                    if verbose:
                        logger.info(f"VAE Loss: {vae_loss.item():.4f}")

            # Forward pass with mixed precision for UNet
            with autocast(enabled=mixed_precision):
                # Get loss weights from tag weighter if available
                tag_weights = None
                if use_tag_weighting and tag_weighter is not None:
                    tag_weights = tag_weighter.get_weights(batch.get('tags', None))

                # Calculate training loss with all improvements
                loss = training_loss_v_prediction(
                    model=model,
                    x=latents,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    min_snr_gamma=min_snr_gamma,
                    sigma_data=sigma_data,
                    use_ztsnr=use_ztsnr,
                    rescale_cfg=rescale_cfg,
                    rescale_multiplier=rescale_multiplier,
                    scale_method=scale_method,
                    resolution_scaling=resolution_scaling,
                    tag_weights=tag_weights
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling for UNet
            scaler.scale(loss).backward()
            
            # Step optimization if gradient accumulation complete
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Synchronize VAE stream if it was used
                if vae_finetuner and batch_idx % vae_train_freq == 0:
                    torch.cuda.current_stream().wait_stream(vae_stream)

                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_grad_norm
                )
                metrics['grad_norm'].update(grad_norm.item())

                # Optimizer and scheduler steps
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update EMA model if available
                if ema_model is not None:
                    ema_model.step(model)
                    if metrics['ema_decay'] is not None:
                        metrics['ema_decay'].update(ema_model._get_decay_rate())

            # Update metrics
            metrics['loss'].update(loss.item() * gradient_accumulation_steps)
            metrics['lr'].update(optimizer.param_groups[0]['lr'])
            if 'bucket_size' in batch:
                metrics['bucket_size'].update(np.prod(batch['bucket_size']))
            metrics['batch_time'].update(time.time() - end)
            
            # Log progress
            if verbose and (batch_idx + 1) % 10 == 0:
                log_metrics_batch(metrics, batch_idx, len(train_dataloader))
            
            end = time.time()

            # Clean up any unused memory
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    return metrics['loss'].avg, {k: v.avg for k, v in metrics.items() if v is not None}

def train(args, models, train_components, device, dtype):
    """Main training loop with improved v-prediction, ZTSNR, VAE finetuning and EMA support"""
    # Model setup
    model = models["unet"]
    if args.use_ema:
        ema_model = models.get("ema")
        if ema_model:
            ema_model.to(device=device, dtype=dtype)
            logger.info("EMA model initialized and moved to device")
            if args.ema_decay:
                ema_model.decay = args.ema_decay
            if args.ema_update_after_step:
                ema_model.update_after_step = args.ema_update_after_step
            if args.ema_power:
                ema_model.power = args.ema_power
            if args.ema_min_decay:
                ema_model.min_decay = args.ema_min_decay
            if args.ema_max_decay:
                ema_model.max_decay = args.ema_max_decay
            if args.ema_update_every:
                ema_model.update_every = args.ema_update_every
            if args.use_ema_warmup:
                ema_model.use_warmup = True
    
    # Optimizer and components setup
    optimizer = train_components["optimizer"]
    lr_scheduler = train_components["lr_scheduler"]
    train_dataloader = train_components["train_dataloader"]
    tag_weighter = train_components["tag_weighter"]
    
    # Initialize training history
    training_history = {
        'loss_history': [],
        'validation_scores': [],
        'ema_validation_scores': [],
        'best_score': float('inf'),
        'best_ema_score': float('inf')
    }
    
    global_step = 0
    start_epoch = 0
    
    # Set up mixed precision training
    scaler = None
    if args.mixed_precision:
        scaler = GradScaler()
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("\nGradient checkpointing enabled")
        if torch.cuda.is_available():
            logger.info("Initial memory usage with gradient checkpointing:")
            logger.info(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            logger.info(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            
            if args.use_wandb:
                log_metrics_batch({
                    "memory/allocated_gb": torch.cuda.memory_allocated()/1e9,
                    "memory/cached_gb": torch.cuda.memory_reserved()/1e9
                }, step=global_step)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, models, train_components)
        start_epoch = checkpoint_data['epoch'] + 1
        global_step = checkpoint_data.get('global_step', 0)
        if 'training_history' in checkpoint_data:
            training_history.update(checkpoint_data['training_history'])
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    # Enable model compilation if requested
    if args.enable_compile:
        logger.info(f"Compiling model with mode: {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Training loop
    logger.info("\nStarting training loop...")
    for epoch in range(start_epoch, args.num_epochs):
        # Log epoch start
        logger.info(f"\nStarting epoch {epoch+1}/{args.num_epochs}")
        if args.use_wandb:
            log_metrics_batch({"train/epoch": epoch}, step=global_step)
        
        # Train one epoch
        avg_loss, metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            tag_weighter=tag_weighter,
            device=device,
            dtype=dtype,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            mixed_precision=args.mixed_precision,
            min_snr_gamma=args.min_snr_gamma,
            sigma_data=args.sigma_data,
            use_ztsnr=args.use_ztsnr,
            rescale_cfg=args.rescale_cfg,
            rescale_multiplier=args.rescale_multiplier,
            scale_method=args.scale_method,
            resolution_scaling=args.resolution_scaling,
            use_tag_weighting=args.use_tag_weighting,
            vae_finetuner=train_components.get("vae_finetuner"),
            vae_train_freq=args.vae_train_freq,
            ema_model=ema_model,
            verbose=args.verbose
        )
        
        # Run end-of-epoch validation
        if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
            logger.info("\nRunning end-of-epoch validation...")
            
            # Regular model validation
            validation_metrics = train_components["validator"].run_validation(
                prompts=args.validation_prompts,
                output_dir=os.path.join(args.output_dir, f"validation_epoch_{epoch+1}"),
                log_to_wandb=args.use_wandb,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale if hasattr(args, 'guidance_scale') else 5.0,
                num_inference_steps=args.num_inference_steps,
                height=1024,
                width=1024
            )
            
            # EMA validation if enabled
            if args.use_ema and ema_model is not None:
                # Temporarily swap UNet with EMA model
                original_unet = train_components["validator"].pipeline.unet
                train_components["validator"].pipeline.unet = ema_model.averaged_model
                
                ema_validation_metrics = train_components["validator"].run_validation(
                    prompts=args.validation_prompts,
                    output_dir=os.path.join(args.output_dir, f"ema_validation_epoch_{epoch+1}"),
                    log_to_wandb=args.use_wandb,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale if hasattr(args, 'guidance_scale') else 5.0,
                    num_inference_steps=args.num_inference_steps,
                    height=1024,
                    width=1024,
                    prefix="ema_"
                )
                
                # Restore original UNet
                train_components["validator"].pipeline.unet = original_unet
        
        # Save checkpoint
        if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                models=models,
                train_components=train_components,
                training_history=training_history,
                global_step=global_step
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update global step
        global_step += len(train_dataloader)
        
        # Push to hub if requested
        if args.push_to_hub and (epoch + 1) % args.save_epochs == 0:
            logger.info("\nPushing to Hub...")
            save_model_card(
                repo_id=args.hub_model_id,
                images=None,  # Add sample images if desired
                base_model=args.model_path,
                train_text_encoder=True,
                prompt=args.validation_prompts[0],
                repo_folder=args.output_dir,
            )
            push_to_hub(
                repo_id=args.hub_model_id,
                output_dir=args.output_dir,
                commit_message=f"Epoch {epoch+1}",
                private=args.hub_private
            )
    
    return training_history

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