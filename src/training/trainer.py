import os
import math
import time
import logging
import traceback
from collections import defaultdict
from typing import Union, Optional, Any, Tuple, Dict
import numpy as np
from dataclasses import dataclass, field
from threading import Lock


import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.hub import save_model_card, push_to_hub
from training import get_sigmas, training_loss_v_prediction
from utils import save_checkpoint, load_checkpoint
from utils.logging import log_metrics_batch

logger = logging.getLogger(__name__)


@dataclass
class AverageMeter:
    """Thread-safe average meter with enhanced functionality and performance"""
    name: str
    fmt: str = ':f'
    window_size: Optional[int] = None
    
    # Initialize private attributes
    _val: float = field(default=0, init=False)
    _sum: float = field(default=0, init=False)
    _count: int = field(default=0, init=False)
    _avg: float = field(default=0, init=False)
    _history: list = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._val = 0
            self._sum = 0
            self._count = 0
            self._avg = 0
            self._history.clear()
    
    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """Thread-safe update with support for tensors and arrays"""
        if isinstance(val, (torch.Tensor, np.ndarray)):
            val = float(val.detach().cpu().item() if torch.is_tensor(val) else val.item())
            
        with self._lock:
            self._val = val
            self._sum += val * n
            self._count += n
            self._avg = self._sum / self._count
            
            if self.window_size:
                self._history.append(val)
                if len(self._history) > self.window_size:
                    self._history.pop(0)
                self._avg = np.mean(self._history)
    
    @property
    def val(self) -> float:
        """Current value"""
        with self._lock:
            return self._val
    
    @property
    def avg(self) -> float:
        """Running average"""
        with self._lock:
            return self._avg
    
    @property
    def sum(self) -> float:
        """Total sum"""
        with self._lock:
            return self._sum
    
    @property
    def count(self) -> int:
        """Number of updates"""
        with self._lock:
            return self._count
    
    @property
    def std(self) -> float:
        """Standard deviation of values"""
        with self._lock:
            if not self._history:
                return 0.0
            return float(np.std(self._history) if len(self._history) > 1 else 0.0)
    
    def get_stats(self) -> dict:
        """Get all statistics"""
        with self._lock:
            return {
                'current': self._val,
                'average': self._avg,
                'sum': self._sum,
                'count': self._count,
                'std': self.std
            }
    
    def state_dict(self) -> dict:
        """Get state for checkpointing"""
        with self._lock:
            return {
                'val': self._val,
                'sum': self._sum,
                'count': self._count,
                'avg': self._avg,
                'history': self._history.copy()
            }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint"""
        with self._lock:
            self._val = state_dict['val']
            self._sum = state_dict['sum']
            self._count = state_dict['count']
            self._avg = state_dict['avg']
            self._history = state_dict.get('history', []).copy()
    
    def __str__(self) -> str:
        """String representation with formatting"""
        with self._lock:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(
                name=self.name, 
                val=self._val, 
                avg=self._avg
            )
    
    def __repr__(self) -> str:
        """Detailed representation"""
        with self._lock:
            return (f"AverageMeter(name='{self.name}', "
                   f"val={self._val:.4f}, "
                   f"avg={self._avg:.4f}, "
                   f"count={self._count})")


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    tag_weighter: Optional[Any],
    **kwargs
) -> Tuple[float, Dict[str, float]]:
    """Optimized training loop with improved performance and memory management"""
    from contextlib import nullcontext
    import gc
    
    def setup_training():
        """Initialize training components"""
        model.train()
        if kwargs.get('use_gradient_checkpointing', True):
            model.enable_gradient_checkpointing()
        
        # Initialize scaler
        scaler = kwargs.get('scaler')
        if scaler is None and kwargs.get('mixed_precision', True):
            scaler = torch.amp.GradScaler('cuda')
        return scaler
    
    def process_batch_to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Efficiently move batch data to device"""
        return {
            'latents': batch['latents'].to(device, dtype=dtype, non_blocking=True),
            'text_embeddings': batch['text_embeddings'].to(device, dtype=dtype, non_blocking=True),
            'added_cond_kwargs': {
                k: v.to(device, dtype=dtype, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.get('added_cond_kwargs', {}).items()
            }
        }
    
    def run_vae_step(batch: Dict[str, Any], vae_stream: torch.cuda.Stream) -> Optional[float]:
        """Execute VAE training step in parallel"""
        if not (vae_finetuner := kwargs.get('vae_finetuner')):
            return None
            
        with torch.cuda.stream(vae_stream):
            vae_loss = vae_finetuner.training_step(batch)
            return vae_loss.item()
    
    def compute_loss(batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss with all improvements"""
        tag_weights = None
        if kwargs.get('use_tag_weighting', True) and tag_weighter is not None:
            tag_weights = tag_weighter.get_weights(batch.get('tags', None))
            
        return training_loss_v_prediction(
            model=model,
            x=batch_data['latents'],
            text_embeddings=batch_data['text_embeddings'],
            added_cond_kwargs=batch_data['added_cond_kwargs'],
            min_snr_gamma=kwargs.get('min_snr_gamma', 5.0),
            sigma_data=kwargs.get('sigma_data', 1.0),
            use_ztsnr=kwargs.get('use_ztsnr', True),
            rescale_cfg=kwargs.get('rescale_cfg', True),
            rescale_multiplier=kwargs.get('rescale_multiplier', 0.7),
            scale_method=kwargs.get('scale_method', "karras"),
            resolution_scaling=kwargs.get('resolution_scaling', True),
            tag_weights=tag_weights
        )
    
    def optimization_step(loss: torch.Tensor, batch_idx: int, metrics: Dict[str, AverageMeter]):
        """Execute optimization step with gradient handling"""
        grad_acc_steps = kwargs.get('gradient_accumulation_steps', 1)
        
        # Scale and backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_acc_steps == 0:
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                kwargs.get('max_grad_norm', 1.0)
            )
            metrics['grad_norm'].update(grad_norm.item())
            
            # Optimizer steps
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA
            if ema_model := kwargs.get('ema_model'):
                ema_model.step(model)
                if metrics['ema_decay'] is not None:
                    metrics['ema_decay'].update(ema_model._get_decay_rate())
    
    def update_metrics(metrics: Dict[str, AverageMeter], loss: torch.Tensor, 
                      batch: Dict[str, Any], batch_time: float):
        """Update training metrics"""
        grad_acc_steps = kwargs.get('gradient_accumulation_steps', 1)
        metrics['loss'].update(loss.item() * grad_acc_steps)
        metrics['lr'].update(optimizer.param_groups[0]['lr'])
        if 'bucket_size' in batch:
            metrics['bucket_size'].update(np.prod(batch['bucket_size']))
        metrics['batch_time'].update(batch_time)
    
    # Initialize training
    scaler = setup_training()
    metrics = {
        'batch_time': AverageMeter('Batch Time', ':6.3f'),
        'data_time': AverageMeter('Data Loading Time', ':6.3f'),
        'loss': AverageMeter('Training Loss', ':.4e'),
        'vae_loss': AverageMeter('VAE Loss', ':.4e') if kwargs.get('vae_finetuner') else None,
        'grad_norm': AverageMeter('Gradient Norm', ':.4e'),
        'lr': AverageMeter('Learning Rate', ':.2e'),
        'bucket_size': AverageMeter('Bucket Size', ':6.0f'),
        'ema_decay': AverageMeter('EMA Decay', ':.4e') if kwargs.get('ema_model') else None
    }
    
    # Training loop
    end = time.time()
    optimizer.zero_grad(set_to_none=True)
    vae_stream = torch.cuda.Stream() if kwargs.get('vae_finetuner') else None
    
    for batch_idx, batch in enumerate(train_dataloader):
        try:
            metrics['data_time'].update(time.time() - end)
            
            # Process batch
            batch_data = process_batch_to_device(batch)
            
            # VAE step
            if vae_stream and batch_idx % kwargs.get('vae_train_freq', 10) == 0:
                if vae_loss := run_vae_step(batch, vae_stream):
                    metrics['vae_loss'].update(vae_loss)
            
            # Forward pass
            with autocast(enabled=kwargs.get('mixed_precision', True)):
                loss = compute_loss(batch_data)
                loss = loss / kwargs.get('gradient_accumulation_steps', 1)
            
            # Optimization
            optimization_step(loss, batch_idx, metrics)
            
            # Metrics
            update_metrics(metrics, loss, batch, time.time() - end)
            
            # Logging
            if kwargs.get('verbose', False) and (batch_idx + 1) % 10 == 0:
                log_metrics_batch(metrics, batch_idx, len(train_dataloader))
            
            end = time.time()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Batch {batch_idx} error: {str(e)}")
            logger.error(traceback.format_exc())
            continue
            
        finally:
            if kwargs.get('use_cpu_offload', False):
                model.to('cpu')
    
    return metrics['loss'].avg, {k: v.avg for k, v in metrics.items() if v is not None}

def train(args, models, train_components, device, dtype):
    """Optimized training loop with improved organization and performance"""
    from contextlib import nullcontext
    import gc
    from contextlib import contextmanager
    
    def setup_training():
        """Initialize training components and state"""
        nonlocal model, ema_model, global_step, start_epoch
        
        # Setup EMA if enabled
        if args.use_ema and (ema_model := models.get("ema")):
            ema_model.to(device=device, dtype=dtype)
            for attr in ['decay', 'update_after_step', 'power', 'min_decay', 
                        'max_decay', 'update_every']:
                setattr(ema_model, attr, getattr(args, f'ema_{attr}'))
            ema_model.use_warmup = True
            logger.info("EMA model initialized")
        
        # Enable gradient checkpointing
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
            log_memory_usage(global_step)
        
        # Resume from checkpoint
        if args.resume_from_checkpoint:
            checkpoint_data = load_checkpoint(args.resume_from_checkpoint, models, train_components)
            start_epoch = checkpoint_data['epoch'] + 1
            global_step = checkpoint_data.get('global_step', 0)
            training_history.update(checkpoint_data.get('training_history', {}))
            logger.info(f"Resumed from epoch {start_epoch}")
            
        # Initialize validator
        setup_validator()
        
        # Compile model if requested
        if args.enable_compile:
            logger.info(f"Compiling model: {args.compile_mode}")
            model = torch.compile(model, mode=args.compile_mode)
    
    def setup_validator():
        """Initialize validation components"""
        if not args.skip_validation and "validator" not in train_components:
            from inference.text_to_image import SDXLInference
            validator = SDXLInference(None, device, dtype)
            validator.pipeline = models.get("pipeline")
            train_components["validator"] = validator
            logger.info("Validator initialized")
    
    def run_validation(epoch):
        """Execute validation step with metrics logging"""
        if not args.skip_validation and (epoch + 1) % args.validation_frequency == 0:
            if validator := train_components.get("validator"):
                logger.info("\nRunning validation...")
                
                # Regular validation
                validation_metrics = validate_model(validator, "validation", epoch)
                log_metrics(validation_metrics, "val", global_step)
                
                # EMA validation
                if args.use_ema and ema_model:
                    with temporarily_swap_unet(validator, ema_model.averaged_model):
                        ema_metrics = validate_model(validator, "ema_validation", epoch)
                        log_metrics(ema_metrics, "ema_val", global_step)
    
    def validate_model(validator, prefix, epoch):
        """Run validation with current model"""
        return validator.run_validation(
            prompts=args.validation_prompts,
            output_dir=os.path.join(args.output_dir, f"{prefix}_epoch_{epoch+1}"),
            log_to_wandb=args.use_wandb,
            num_images_per_prompt=1,
            guidance_scale=getattr(args, 'guidance_scale', 5.0),
            num_inference_steps=args.num_inference_steps,
            height=1024, width=1024,
            prefix=prefix.split('_')[0] + "_" if prefix.startswith("ema") else ""
        )
    
    @contextmanager
    def temporarily_swap_unet(validator, new_unet):
        """Safely swap UNet models"""
        original_unet = validator.pipeline.unet
        validator.pipeline.unet = new_unet
        try:
            yield
        finally:
            validator.pipeline.unet = original_unet
    
    def log_metrics(metrics, prefix, step):
        """Log metrics to console and wandb"""
        for name, value in metrics.items():
            logger.info(f"{prefix.title()} {name}: {value:.4f}")
            training_history[f"{prefix}/{name}"].append(value)
            if args.use_wandb:
                wandb.log({f"{prefix}/{name}": value}, step=step)
    
    def save_progress(epoch):
        """Save checkpoints and push to hub"""
        if args.save_checkpoints and (epoch + 1) % args.save_epochs == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(
                checkpoint_path, epoch=epoch, models=models,
                train_components=train_components,
                training_history=training_history,
                global_step=global_step
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            if args.push_to_hub:
                push_to_hub_with_card(epoch)
    
    def push_to_hub_with_card(epoch):
        """Push model to hub with card"""
        logger.info("\nPushing to Hub...")
        save_model_card(
            repo_id=args.hub_model_id,
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
    
    def log_memory_usage(step):
        """Log memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1e9
            cached = torch.cuda.memory_reserved()/1e9
            logger.info(f"Memory usage - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            if args.use_wandb:
                wandb.log({
                    "memory/allocated_gb": allocated,
                    "memory/cached_gb": cached
                }, step=step)
    
    # Main training loop
    model = models["unet"]
    ema_model = None
    global_step = 0
    start_epoch = 0
    training_history = defaultdict(list)
    
    # Initialize components
    setup_training()
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
    
    logger.info("\nStarting training loop...")
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        if args.use_wandb:
            wandb.log({"train/epoch": epoch}, step=global_step)
        
        # Train epoch
        avg_loss, metrics = train_one_epoch(
            model=model, optimizer=train_components["optimizer"],
            lr_scheduler=train_components["lr_scheduler"],
            train_dataloader=train_components["train_dataloader"],
            tag_weighter=train_components["tag_weighter"],
            device=device, dtype=dtype,
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
            verbose=args.verbose,
            scaler=scaler
        )
        
        # Log metrics
        log_metrics({"loss": avg_loss, **metrics}, "train", global_step)
        
        # Validation
        run_validation(epoch)
        
        # Save progress
        save_progress(epoch)
        
        # Update learning rate and step
        train_components["lr_scheduler"].step()
        if args.use_wandb:
            wandb.log({
                "train/learning_rate": train_components["lr_scheduler"].get_last_lr()[0]
            }, step=global_step)
        
        global_step += len(train_components["train_dataloader"])
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final summary
    logger.info("\nTraining completed!")
    logger.info(f"Final loss: {avg_loss:.4f}")
    logger.info("Final metrics:")
    for metric_name, values in training_history.items():
        if values:
            logger.info(f"{metric_name}: {values[-1]:.4f}")
    
    return training_history

