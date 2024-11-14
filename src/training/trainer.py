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
    vae_finetuner: Optional[Any] = None,
    vae_train_freq: int = 10,
    verbose: bool = False
) -> Tuple[float, Dict[str, float]]:
    model.train()
    scaler = GradScaler(enabled=mixed_precision)
    metrics = {
        'batch_time': AverageMeter('Batch Time', ':6.3f'),
        'data_time': AverageMeter('Data Loading Time', ':6.3f'),
        'loss': AverageMeter('Training Loss', ':.4e'),
        'vae_loss': AverageMeter('VAE Loss', ':.4e') if vae_finetuner else None,
        'grad_norm': AverageMeter('Gradient Norm', ':.4e'),
        'lr': AverageMeter('Learning Rate', ':.2e')
    }
    
    end = time.time()
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(train_dataloader):
        try:
            # Log input tensor shapes
            logger.info(f"\nBatch {batch_idx}/{len(train_dataloader)}:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"Input {k}: shape={v.shape}, dtype={v.dtype}")

            metrics['data_time'].update(time.time() - end)
            
            # Device transfer with shape logging
            latents = batch['latents'].to(device, dtype=dtype, non_blocking=True)
            text_embeddings = batch['text_embeddings'].to(device, dtype=dtype, non_blocking=True)
            added_cond_kwargs = {k: v.to(device, dtype=dtype, non_blocking=True) 
                               if torch.is_tensor(v) else v 
                               for k, v in batch.get('added_cond_kwargs', {}).items()}
            
            # Resolution and sigma logging
            height, width = latents.shape[2:4]
            logger.info(f"Resolution: {height*8}x{width*8} (latents: {height}x{width})")
            
            sigma = get_sigmas(height=height*8, width=width*8)[0].to(device, dtype=dtype)
            logger.info(f"Sigma: {sigma.item():.6f}")
            
            with autocast(device_type='cuda', dtype=dtype, enabled=mixed_precision):
                loss = training_loss_v_prediction(
                    model=model,
                    x_0=latents,
                    sigma=sigma,
                    text_embeddings=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    tag_weighter=tag_weighter,
                    batch_tags=batch,
                    min_snr_gamma=min_snr_gamma,
                    sigma_data=sigma_data,
                    verbose=verbose
                )
                loss = loss / gradient_accumulation_steps
            
            if not torch.isfinite(loss):
                logger.error(f"Loss is {loss.item()}")
                continue
                
            scaler.scale(loss).backward()
            
            if vae_finetuner and batch_idx % vae_train_freq == 0:
                vae_loss = vae_finetuner.training_step(batch)
                metrics['vae_loss'].update(vae_loss.item())
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                metrics['grad_norm'].update(grad_norm.item())
                logger.info(f"Gradient norm: {grad_norm.item():.6f}")
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                lr_scheduler.step()
                metrics['lr'].update(optimizer.param_groups[0]['lr'])
            
            metrics['loss'].update(loss.item() * gradient_accumulation_steps)
            metrics['batch_time'].update(time.time() - end)
            
            if batch_idx % 10 == 0:
                logger.info("Metrics:")
                for name, meter in metrics.items():
                    if meter is not None:
                        logger.info(f"{name}: {meter}")
            
            end = time.time()
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error("\nShape mismatch detected:")
                err_msg = str(e)
                param = err_msg.split('size mismatch for ')[1].split(':')[0] 
                expected = err_msg.split('copying a param with shape ')[1].split(',')[0]
                actual = err_msg.split('the shape in current model is ')[1]
                logger.error(f"Parameter: {param}")
                logger.error(f"Expected shape: {expected}")
                logger.error(f"Actual shape: {actual}")
            raise
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}:")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            continue

    return metrics['loss'].avg, {k: v.avg if v is not None else 0.0 for k, v in metrics.items()}

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
            vae_finetuner=train_components.get("vae_finetuner"),
            vae_train_freq=args.vae_train_freq,
            verbose=args.verbose
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