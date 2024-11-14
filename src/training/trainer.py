import os
import math
import time
import logging
import traceback
from collections import defaultdict

import torch
import wandb
import numpy as np
from tqdm import tqdm

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
    model, optimizer, lr_scheduler, train_dataloader, 
    tag_weighter, device, dtype, 
    gradient_accumulation_steps=1, 
    max_grad_norm=1.0, 
    mixed_precision=True,
    min_snr_gamma=5.0,
    sigma_data=1.0,
    sigma_min=0.01,
    resolution_scaling=1.0,
    rescale_cfg=None,
    scale_method="linear",
    rescale_multiplier=1.0
):
    model.train()
    total_loss = 0.0
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    end = time.time()
    global_step = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Unpack batch with efficient unpacking
        latents = batch['latents'].to(device, dtype=dtype, non_blocking=True)
        text_embeddings = batch['text_embeddings'].to(device, dtype=dtype, non_blocking=True)
        added_cond_kwargs = {
            k: v.to(device, dtype=dtype, non_blocking=True) 
            if torch.is_tensor(v) else v 
            for k, v in batch['added_cond_kwargs'].items()
        }
        
        # Get resolution-dependent sigma
        height, width = latents.shape[2:4]
        sigma = get_sigmas(height=height * 8, width=width * 8)[0].to(device, dtype=dtype)
        
        # Forward pass and loss computation
        with torch.amp.autocast('cuda', enabled=mixed_precision):
            loss = training_loss_v_prediction(
                model, 
                latents, 
                sigma, 
                text_embeddings, 
                added_cond_kwargs=added_cond_kwargs,
                tag_weighter=tag_weighter,
                batch_tags=batch,
                min_snr_gamma=min_snr_gamma,
                sigma_data=sigma_data,
                sigma_min=sigma_min,
                resolution_scaling=resolution_scaling,
                rescale_cfg=rescale_cfg,
                scale_method=scale_method,
                rescale_multiplier=rescale_multiplier
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation and optimization
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update tracking metrics
            total_loss += loss.item() * gradient_accumulation_steps
            losses.update(loss.item() * gradient_accumulation_steps)
            
            # Log metrics efficiently
            if wandb.run:
                log_metrics_batch({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "performance/batch_time": batch_time.avg,
                    "performance/data_time": data_time.avg
                }, step=global_step)
            
            global_step += 1
        
        # Measure batch time
        batch_time.update(time.time() - end)
        end = time.time()
    
    return global_step


def train(args, models, train_components, device, dtype):
    """Main training loop with improved v-prediction and ZTSNR support"""
    model = models["unet"]
    if args.use_ema:
        ema_model = models["ema"]
    
    optimizer = train_components["optimizer"]
    lr_scheduler = train_components["lr_scheduler"]
    train_dataloader = train_components["train_dataloader"]
    tag_weighter = train_components["tag_weighter"]
    
    training_history = {
        'loss_history': [],
        'validation_scores': [],
        'ema_validation_scores': [],
        'best_score': float('inf'),
        'best_ema_score': float('inf')
    }
    
    global_step = 0
    start_epoch = 0
    
    # Monitor memory usage with gradient checkpointing
    if args.gradient_checkpointing and torch.cuda.is_available():
        logger.info("\nInitial memory usage with gradient checkpointing:")
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
    
    # Use the validator that was already initialized in train_components
    validator = train_components["validator"]
    
    # Update UNet in validator to use training UNet
    validator.pipeline.unet = models["unet"]

    # Training loop
    logger.info("\nStarting training loop...")
    for epoch in range(start_epoch, args.num_epochs):
        # Log epoch start
        logger.info(f"\nStarting epoch {epoch+1}/{args.num_epochs}")
        if args.use_wandb:
            log_metrics_batch({"train/epoch": epoch}, step=global_step)
        
        # Train one epoch
        global_step = train_one_epoch(
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
            sigma_min=args.sigma_min,
            resolution_scaling=args.resolution_scaling,
            rescale_cfg=args.rescale_cfg,
            scale_method=args.scale_method,
            rescale_multiplier=args.rescale_multiplier
        )
        
        # Run end-of-epoch validation
        if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
            logger.info("\nRunning end-of-epoch validation...")
            
            # Regular model validation (using training model)
            validation_metrics = train_components["validator"].run_validation(
                prompts=args.validation_prompts,
                output_dir=os.path.join(args.output_dir, f"validation_epoch_{epoch+1}"),
                log_to_wandb=args.use_wandb,
                num_images_per_prompt=1,
                guidance_scale=5.0,
                num_inference_steps=28,
                height=1024,
                width=1024
            )
            
            # EMA validation if enabled
            if train_components["ema_model"] is not None:
                # Temporarily swap UNet with EMA model
                original_unet = train_components["validator"].pipeline.unet
                train_components["validator"].pipeline.unet = train_components["ema_model"].averaged_model
                
                ema_validation_metrics = train_components["validator"].run_validation(
                    prompts=args.validation_prompts,
                    output_dir=os.path.join(args.output_dir, f"ema_validation_epoch_{epoch+1}"),
                    log_to_wandb=args.use_wandb,
                    num_images_per_prompt=1,
                    guidance_scale=5.0,
                    num_inference_steps=28,
                    height=1024,
                    width=1024
                )
                
                # Restore original UNet
                train_components["validator"].pipeline.unet = original_unet
                
                if args.use_wandb:
                    log_metrics_batch({
                        "validation/epoch": epoch + 1,
                        "validation/metrics": validation_metrics,
                        "validation/ema_metrics": ema_validation_metrics
                    }, step=global_step)
            else:
                if args.use_wandb:
                    log_metrics_batch({
                        "validation/epoch": epoch + 1,
                        "validation/metrics": validation_metrics
                    }, step=global_step)
            
            # Update training history
            training_history['validation_scores'].append(validation_metrics)
            if train_components["ema_model"] is not None:
                training_history['ema_validation_scores'].append(ema_validation_metrics)
        
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