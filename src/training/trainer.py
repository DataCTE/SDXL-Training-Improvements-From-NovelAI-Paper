import logging
import gc
import os
import math
import time
from typing import Dict, Any, Union, Optional, List
from functools import lru_cache
from multiprocessing import Manager
from dataclasses import dataclass
from collections import defaultdict
from contextlib import nullcontext

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import Adafactor
from bitsandbytes.optim import AdamW8bit
from diffusers import AutoencoderKL

from src.inference.text_to_image import SDXLInference
from src.training.ema import EMAModel
from src.training.loss import training_loss_v_prediction, get_cosine_schedule_with_warmup
from src.data.tag_weighter import TagBasedLossWeighter
from src.training.vae_finetuner import VAEFineTuner
from src.data.dataset.dataset import CustomDataset
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _get_optimizer_config(
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
) -> Dict[str, Any]:
    """Cache optimizer configurations."""
    base_config = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    if optimizer_type.lower() == "adamw":
        return {**base_config, "betas": (adam_beta1, adam_beta2), "eps": adam_epsilon}
    elif optimizer_type.lower() == "adafactor":
        return {
            **base_config,
            "scale_parameter": True,
            "relative_step": False,
            "warmup_init": False,
        }
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def setup_optimizer(args, models) -> torch.optim.Optimizer:
    """Set up optimizer with proper configuration and memory optimizations."""
    try:
        # Validate UNet model
        if not hasattr(models["unet"], "parameters"):
            raise ValueError("UNet model is not properly initialized")

        # Get trainable parameters with optimized list comprehension
        params_to_optimize = [p for p in models["unet"].parameters() if p.requires_grad]

        # Add text encoder parameters if needed
        if getattr(args, "train_text_encoder", False):
            if all(k in models for k in ["text_encoder", "text_encoder_2"]):
                text_params = [
                    p
                    for model_key in ["text_encoder", "text_encoder_2"]
                    for p in models[model_key].parameters()
                    if p.requires_grad
                ]
                params_to_optimize.extend(text_params)

        # Validate parameters
        if not params_to_optimize:
            raise ValueError("No trainable parameters found")

        # Log total number of trainable parameters
        total_params = sum(p.numel() for p in params_to_optimize)
        logger.info("Total trainable parameters: %s", format(total_params, ","))

        # Get cached optimizer config
        opt_config = _get_optimizer_config(
            getattr(args.optimizer, "optimizer_type", "adamw"),
            args.training.learning_rate,
            args.optimizer.weight_decay,
            args.optimizer.adam_beta1,
            args.optimizer.adam_beta2,
            args.optimizer.adam_epsilon,
        )

        # Initialize optimizer based on type
        if args.optimizer.use_8bit_adam:
            try:
                optimizer = AdamW8bit(params_to_optimize, **opt_config)
            except ImportError as e:
                logger.warning("Failed to import 8-bit AdamW: %s. Falling back to regular AdamW.", str(e))
                args.optimizer.use_8bit_adam = False
                optimizer = torch.optim.AdamW(params_to_optimize, **opt_config)
        elif args.optimizer.use_adafactor:
            optimizer = Adafactor(
                params_to_optimize,
                lr=opt_config["lr"],
                weight_decay=opt_config["weight_decay"],
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
            )
        else:
            optimizer = torch.optim.AdamW(params_to_optimize, **opt_config)

        return optimizer

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup optimizer: {str(e)}") from e

def setup_vae_finetuner(args, models) -> Optional[VAEFineTuner]:
    """
    Set up the VAE FineTuner if fine-tuning is enabled in the arguments.

    Args:
        args: A configuration object containing various training parameters.
        models: A dictionary of models, expected to include the 'vae' model.

    Returns:
        An instance of VAEFineTuner if fine-tuning is enabled, otherwise None.

    Raises:
        Exception: If the setup of the VAE FineTuner fails.
    """
    if not args.vae.finetune_vae:
        return None

    try:
        vae_finetuner = VAEFineTuner(
            vae=models["vae"],
            device=args.training.device,
            mixed_precision=args.training.mixed_precision,
            use_amp=args.training.use_amp,
            learning_rate=args.vae.vae_learning_rate,
            adam_beta1=args.optimizer.adam_beta1,
            adam_beta2=args.optimizer.adam_beta2,
            adam_epsilon=args.optimizer.adam_epsilon,
            weight_decay=args.optimizer.weight_decay,
            use_8bit_adam=args.optimizer.use_8bit_adam,
            use_channel_scaling=args.vae.vae_use_channel_scaling,
            adaptive_loss_scale=args.vae.adaptive_loss_scale,
            kl_weight=args.vae.kl_weight,
            perceptual_weight=args.vae.perceptual_weight,
            initial_scale_factor=args.vae.vae_initial_scale_factor,
            decay=args.vae.vae_decay,
            update_after_step=args.vae.vae_update_after_step,
        )
        return vae_finetuner
    except Exception as e:
        logger.error("Failed to setup VAE finetuner: %s", str(e))
        raise

@lru_cache(maxsize=1)
def _get_ema_config(
    decay: float = 0.9999,
    update_every: int = 10,
    device: Union[str, torch.device] = None,
    model_path: str = None,
) -> Dict[str, Any]:
    """Get basic EMA configuration."""
    return {
        "decay": decay,
        "update_every": update_every,
        "device": device,
        "model_path": model_path,
    }


def setup_ema(args, model, device=None):
    """Setup EMA model with proper error handling"""
    try:
        # Use provided device or default to CUDA/CPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decay = args.ema.ema_decay

        # Basic validation
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay:.6f}")

        ema_config = _get_ema_config(
            decay=decay,
            update_every=args.ema.ema_update_every,
            device=device,
            model_path=args.model.model_path,
        )

        if args.ema.use_ema:
            ema = EMAModel(model, **ema_config)
            return ema
        return None

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup EMA model: {str(e)}") from e


def setup_validator(args, models, device, dtype) -> Optional[Any]:
    """Initialize validation components."""
    try:
        if getattr(args.system, "skip_validation", False):
            return None

        validator = SDXLInference(None, device, dtype)
        validator.pipeline = models.get("pipeline")

        return validator

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup validator: {str(e)}") from e


@lru_cache(maxsize=32)
def _get_tag_weighter_config(
    base_weight: float,
    min_weight: float,
    max_weight: float,
    window_size: int,
    no_cache: bool = False,
) -> Dict[str, Any]:
    """Cache tag weighter configuration."""
    return {
        "base_weight": base_weight,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "window_size": window_size,
        "no_cache": no_cache,
    }


def setup_tag_weighter(args) -> Optional[Any]:
    """Initialize tag weighting system."""
    try:
        if not args.tag_weighting.use_tag_weighting:
            return None

        weighter_config = _get_tag_weighter_config(
            base_weight=getattr(args.tag_weighting, "tag_base_weight", 1.0),
            min_weight=args.tag_weighting.min_tag_weight,
            max_weight=args.tag_weighting.max_tag_weight,
            window_size=getattr(args.tag_weighting, "tag_window_size", 100),
            no_cache=args.data.no_caching,
        )
        weighter = TagBasedLossWeighter(config=weighter_config)

        return weighter

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup tag weighter: {str(e)}") from e


@dataclass
class AverageMeter:
    """Fully pickleable average meter."""

    name: str
    fmt: str = ":f"
    window_size: Optional[int] = None

    def __post_init__(self):
        self._val: float = 0
        self._sum: float = 0
        self._count: int = 0
        self._avg: float = 0
        self._history: List[float] = []

    def __getstate__(self):
        """Get state for pickling."""
        return {
            "name": self.name,
            "fmt": self.fmt,
            "window_size": self.window_size,
            "_val": self._val,
            "_sum": self._sum,
            "_count": self._count,
            "_avg": self._avg,
            "_history": self._history,
        }

    def __setstate__(self, state):
        """Set state for unpickling."""
        self.__dict__.update(state)

    def reset(self) -> None:
        """Reset all metrics."""
        self._val = 0
        self._sum = 0
        self._count = 0
        self._avg = 0
        self._history.clear()

    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """Update with support for tensors and arrays."""
        if isinstance(val, (torch.Tensor, np.ndarray)):
            val = float(
                val.detach().cpu().item() if torch.is_tensor(val) else val.item()
            )

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
        """Get the most recent value added to the meter.

        Returns:
            float: The last value that was added
        """
        return self._val

    @property
    def avg(self) -> float:
        """Get the average of all values in the meter.

        Returns:
            float: The running average value
        """
        return self._avg

    @property
    def sum(self) -> float:
        """Get the sum of all values added to the meter.

        Returns:
            float: The total sum of values
        """
        return self._sum

    @property
    def count(self) -> int:
        """Get the count of values that have been added.

        Returns:
            int: The total number of updates
        """
        return self._count


class MetricsManager:
    """Fully process-safe metrics manager."""

    def __init__(self):
        self._manager = Manager()
        self._metrics = self._manager.dict()

    def get_metric(self, name: str) -> AverageMeter:
        """Retrieve or create a metric by name.

        Args:
            name (str): The name of the metric to retrieve

        Returns:
            AverageMeter: The metric object associated with the given name
        """
        if name not in self._metrics:
            self._metrics[name] = AverageMeter(name=name)
        return self._metrics[name]

    def update_metric(self, name: str, value: float, n: int = 1) -> None:
        """Update a metric with a new value.

        Args:
            name (str): The name of the metric to update
            value (float): The value to update the metric with
            n (int, optional): The weight of the update. Defaults to 1.
        """
        metric = self.get_metric(name)
        metric.update(value, n)
        self._metrics[name] = metric  # Update in shared dict

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary of averages.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their current average values
        """
        return {name: meter.avg for name, meter in dict(self._metrics).items()}


def initialize_training_components(args, models):
    """Initialize all training components with proper error handling"""
    components = {}

    try:
        # Setup optimizer with validation
        if not models.get("unet"):
            raise ValueError("UNet model not found in models dictionary")
        components["optimizer"] = setup_optimizer(args, models)

        # Setup data loader with validation
        required_models = [
            "vae",
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
        ]
        if not all(k in models for k in required_models):
            raise ValueError(
                f"Missing required models: {[k for k in required_models if k not in models]}"
            )

        # Create process-safe copies of models for the dataloader
        dataloader_models = {}
        for key in required_models:
            model = models[key]
            if key == "vae":
                # Special handling for VAE
                

                dataloader_models[key] = AutoencoderKL.from_config(model.config)
                dataloader_models[key].load_state_dict(model.state_dict())
            elif hasattr(model, "state_dict"):
                if hasattr(model, "config"):
                    # Handle transformers models
                    if "CLIPTextModel" in model.__class__.__name__:
                        dataloader_models[key] = type(model)(config=model.config)
                        dataloader_models[key].load_state_dict(model.state_dict())
                    else:
                        # Handle other models with configs
                        dataloader_models[key] = type(model)(**model.config)
                        dataloader_models[key].load_state_dict(model.state_dict())
                else:
                    # Handle models without config
                    dataloader_models[key] = type(model)()
                    dataloader_models[key].load_state_dict(model.state_dict())
            else:
                # For tokenizers and other objects that don't need special handling
                dataloader_models[key] = model

        # Initialize data loader
        components["train_dataloader"] = DataLoader(
            dataset=CustomDataset(
                data_dir=args.data.data_dir,
                tokenizer=dataloader_models["tokenizer"],
                tokenizer_2=dataloader_models["tokenizer_2"],
                text_encoder=dataloader_models["text_encoder"],
                text_encoder_2=dataloader_models["text_encoder_2"],
                vae=dataloader_models["vae"],
                cache_dir=args.data.cache_dir,
                no_caching_latents=args.data.no_caching,
                use_tag_weighting=args.tag_weighting.use_tag_weighting,
                num_workers=args.system.num_workers,
            ),
            batch_size=args.training.batch_size,
            shuffle=True,
            num_workers=args.system.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Setup scheduler
        num_update_steps_per_epoch = (
            len(components["train_dataloader"])
            // args.training.gradient_accumulation_steps
        )
        components["scheduler"] = get_cosine_schedule_with_warmup(
            optimizer=components["optimizer"],
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=num_update_steps_per_epoch * args.training.num_epochs,
        )

        # Setup optional components with parallel initialization
        optional_components = {
            "ema": (setup_ema, args.ema.use_ema, (args, models["unet"])),
            "tag_weighter": (setup_tag_weighter, args.tag_weighting.use_tag_weighting, (args,)),
            "vae_finetuner": (setup_vae_finetuner, args.vae.finetune_vae, (args, models)),
        }

        components.update(
            {
                name: setup_func(*setup_args) if use_flag else None
                for name, (setup_func, use_flag, setup_args) in optional_components.items()
            }
        )

        # Setup metrics tracking
        components["metrics"] = MetricsManager()

        # Validate components
        _validate_components(components)

        return components

    except (ValueError, RuntimeError, AttributeError) as e:
        _cleanup_failed_initialization(components)
        raise type(e)(f"Failed to initialize training components: {str(e)}") from e


def _validate_components(components: Dict[str, Any]) -> None:
    """Validate initialized components."""
    required = ["optimizer", "train_dataloader", "scheduler", "metrics"]
    if not all(k in components for k in required):
        raise ValueError(
            f"Missing required components: {[k for k in required if k not in components]}"
        )


def _cleanup_failed_initialization(components: Dict[str, Any]) -> None:
    """Clean up resources in case of failed initialization."""
    try:
        # Close data loader if it was created
        if "train_dataloader" in components:
            try:
                components["train_dataloader"].dataset.cleanup()
            except (AttributeError, RuntimeError, IOError, OSError) as e:
                raise type(e)(f"Failed to cleanup train dataloader: {str(e)}") from e
            except (KeyError, TypeError) as e:
                raise type(e)(f"Invalid train dataloader configuration: {str(e)}") from e

        # Clear CUDA cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except (RuntimeError, torch.cuda.CudaError) as e:
                raise type(e)(f"Failed to clear CUDA cache: {str(e)}") from e

    except (KeyError, TypeError, AttributeError) as e:
        raise type(e)(f"Failed to cleanup training resources: {str(e)}") from e


def train_epoch(
    epoch: int,
    args,
    models: Dict[str, Any],
    components: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    global_step: int = 0,
) -> Dict[str, float]:
    """Train for one epoch and return metrics.
    
    Args:
        epoch: Current epoch number
        args: Training configuration
        models: Dictionary of models (unet, vae, etc.)
        components: Dictionary of training components (optimizer, scheduler, etc.)
        device: Device to train on
        dtype: Data type for training
        global_step: Global training step counter
    """
    models["unet"].train()
    
    # Initialize progress bar
    progress = tqdm(
        total=len(components["train_dataloader"]),
        desc=f"Training epoch {epoch}",
        leave=False,
    )

    # Initialize metrics for this epoch
    epoch_metrics = defaultdict(float)
    samples_seen = 0
    current_step = global_step

    for batch_idx, batch in enumerate(components["train_dataloader"]):
        with components.get("metrics", nullcontext()) as metrics:
            # Forward pass
            with torch.cuda.amp.autocast(enabled=args.system.mixed_precision):
                loss = forward_pass(args, models, batch, device, dtype, components)
                loss = loss / args.training.gradient_accumulation_steps

            # Update metrics
            if metrics:
                metrics.update("loss", loss.item() * args.training.gradient_accumulation_steps)
                epoch_metrics["loss"] += loss.item() * args.training.gradient_accumulation_steps
                samples_seen += 1

            # Backward pass and optimization
            if args.system.mixed_precision:
                components["scaler"].scale(loss).backward()
                if (batch_idx + 1) % args.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        models["unet"].parameters(), args.training.max_grad_norm
                    )
                    components["scaler"].step(components["optimizer"])
                    components["scaler"].update()
                    components["scheduler"].step()
                    components["optimizer"].zero_grad()
                    current_step += 1
            else:
                loss.backward()
                if (batch_idx + 1) % args.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        models["unet"].parameters(), args.training.max_grad_norm
                    )
                    components["optimizer"].step()
                    components["scheduler"].step()
                    components["optimizer"].zero_grad()
                    current_step += 1

            # Update EMA model if enabled
            if (
                components.get("ema") is not None
                and (batch_idx + 1) % args.training.gradient_accumulation_steps == 0
            ):
                components["ema"].step(models["unet"])

            # Log progress
            if metrics and batch_idx % args.logging.logging_steps == 0:
                metrics.log_metrics(
                    step=current_step,
                    epoch=epoch,
                    learning_rate=components["scheduler"].get_last_lr()[0],
                )

            # Run validation if needed
            if (
                args.validation.validation_steps > 0
                and current_step > 0
                and current_step % args.validation.validation_steps == 0
            ):
                run_validation(args, models, components, device, dtype, global_step=current_step)

        # Update progress bar
        progress.update(1)
        if batch_idx % args.logging.logging_steps == 0:
            progress.set_postfix(
                {
                    "loss": epoch_metrics["loss"] / samples_seen if samples_seen > 0 else 0,
                    "lr": components["scheduler"].get_last_lr()[0],
                    "step": current_step,
                }
            )

    progress.close()

    # Log epoch metrics
    epoch_metrics = {k: v / samples_seen for k, v in epoch_metrics.items()} if samples_seen > 0 else {}
    if components.get("metrics"):
        components["metrics"].log_epoch_metrics(
            epoch=epoch,
            metrics=epoch_metrics,
            step=current_step,
        )

    # Save checkpoint if needed
    if epoch % args.logging.save_epochs == 0:
        training_history = {
            'loss_history': epoch_metrics,
            'total_steps': current_step
        }
        save_checkpoint(
            args.logging.checkpoint_dir,
            models,
            components,
            training_history
        )
        
    return epoch_metrics


def train(args, models, components, device, dtype) -> Dict[str, float]:
    """Execute training steps with proper error handling."""
    metrics_manager = components["metrics_manager"]

    # Set model to training mode
    models["unet"].train()
    if getattr(args, "train_text_encoder", False):
        models["text_encoder"].train()
        models["text_encoder_2"].train()

    start_time = time.time()

    progress_bar = tqdm(
        components["train_dataloader"],
        desc="Training",
        dynamic_ncols=True,
        leave=False,
    )

    try:
        for batch_idx, batch in enumerate(progress_bar):
            try:
                metrics_manager.update_metric("data_time", time.time() - start_time)

                # Execute training step
                batch_metrics = train_step(
                    args=args,
                    models=models,
                    components=components,
                    batch=batch,
                    batch_idx=batch_idx,
                    device=device,
                    dtype=dtype,
                )

                # Update metrics
                for k, v in batch_metrics.items():
                    metrics_manager.update_metric(k, v)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        k: f"{v:.6f}"
                        for k, v in metrics_manager.get_all_metrics().items()
                    }
                )

                metrics_manager.update_metric("batch_time", time.time() - start_time)
                start_time = time.time()

            except (ValueError, RuntimeError, AttributeError) as e:
                raise type(e)(f"Failed during training step: {str(e)}") from e

        return metrics_manager.get_all_metrics()

    finally:
        progress_bar.close()

        # Set models back to eval mode if needed
        models["unet"].eval()
        if getattr(args, "train_text_encoder", False):
            models["text_encoder"].eval()
            models["text_encoder_2"].eval()


def train_step(
    args, models, components, batch, batch_idx: int, device, dtype
) -> Dict[str, float]:
    """Execute single training step with proper error handling."""
    try:
        # Move batch to device
        batch = {
            k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Zero gradients
        components["optimizer"].zero_grad(set_to_none=True)

        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.system.mixed_precision):
            # Get loss from model
            loss = models["unet"](
                x_0=batch["latents"],
                sigma=batch["sigmas"],
                text_embeddings=batch["text_embeddings"],
                added_cond_kwargs=batch.get("added_cond_kwargs"),
                sigma_data=args.sigma_data,
                tag_weighter=components.get("tag_weighter"),
                batch_tags=batch.get("tags"),
                min_snr_gamma=args.min_snr_gamma,
                rescale_cfg=args.rescale_cfg,
                rescale_multiplier=args.rescale_multiplier,
                scale_method=args.scale_method,
                use_tag_weighting=args.use_tag_weighting,
                device=device,
                dtype=dtype,
            )

            # Scale loss for gradient accumulation
            if args.training.gradient_accumulation_steps > 1:
                loss = loss / args.training.gradient_accumulation_steps

        # Backward pass with gradient scaling
        if args.system.mixed_precision:
            components["scaler"].scale(loss).backward()
            if (batch_idx + 1) % args.training.gradient_accumulation_steps == 0:
                components["scaler"].unscale_(components["optimizer"])
                torch.nn.utils.clip_grad_norm_(
                    models["unet"].parameters(), args.training.max_grad_norm
                )
                components["scaler"].step(components["optimizer"])
                components["scaler"].update()
                components["scheduler"].step()
        else:
            loss.backward()
            if (batch_idx + 1) % args.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    models["unet"].parameters(), args.training.max_grad_norm
                )
                components["optimizer"].step()
                components["scheduler"].step()

        # Update EMA model if enabled
        if (
            components.get("ema") is not None
            and (batch_idx + 1) % args.training.gradient_accumulation_steps == 0
        ):
            components["ema"].step(models["unet"])

        # Update VAE if enabled
        if components.get("vae_finetuner") is not None:
            vae_metrics = components["vae_finetuner"].train_step(batch)
        else:
            vae_metrics = {}

        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "lr": components["optimizer"].param_groups[0]["lr"],
            "grad_norm": get_grad_norm(models["unet"]),
            **vae_metrics,
        }

        if components.get("ema"):
            metrics["ema_decay"] = components["ema"].cur_decay_value

        return metrics

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to execute training step: {str(e)}") from e


def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm with proper error handling."""
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to calculate gradient norm: {str(e)}") from e


def run_validation(
    args, models, components, device, dtype, global_step: int
) -> Dict[str, float]:
    """Run validation with proper error handling and metrics tracking."""
    try:
        # Initialize inference pipeline if not already in components
        if "inference" not in components:
            components["inference"] = SDXLInference(
                device=device,
                dtype=dtype,
                use_v_prediction=args.training_mode == "v_prediction",
            )

        # Set models to eval mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.eval()

        # Run validation
        with torch.no_grad():
            # Run validation using configured prompts
            validation_metrics = components["inference"].run_validation(
                prompts=args.validation_prompts,
                output_dir=os.path.join(
                    args.output_dir, f"validation_{global_step}"
                ),
                log_to_wandb=args.use_wandb,
                guidance_scale=args.validation_guidance_scale,
                num_inference_steps=args.validation_num_steps,
                height=args.validation_height,
                width=args.validation_width,
                num_images_per_prompt=args.validation_images_per_prompt,
                seed=args.validation_seed,
                rescale_cfg=args.rescale_cfg,
                scale_method=args.scale_method,
                rescale_multiplier=args.rescale_multiplier,
            )

            # Add EMA metrics if enabled
            if components.get("ema"):
                validation_metrics["validation/ema_decay"] = components[
                    "ema"
                ].cur_decay_value

            return validation_metrics

    finally:
        # Set models back to train mode
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.train()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def forward_pass(args, models, batch, device, dtype, components) -> torch.Tensor:
    """Execute forward pass with proper error handling."""
    try:
        # Move batch to device
        batch = {k: v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get model predictions
        latents = batch["latents"]
        timesteps = batch["timesteps"]
        
        # Get noise prediction using v-prediction loss
        loss = training_loss_v_prediction(
            model=models["unet"],
            x_0=latents,
            sigma=timesteps,
            text_embeddings=batch["prompt_embeds"],
            added_cond_kwargs={
                "text_embeds": batch["pooled_prompt_embeds"],
                "time_ids": batch["add_text_embeds"],
            },
            sigma_data=args.training.sigma_data,
            tag_weighter=components.get("tag_weighter"),
            batch_tags=batch.get("tags"),
            min_snr_gamma=args.training.min_snr_gamma,
            rescale_cfg=args.training.rescale_cfg,
            rescale_multiplier=args.training.rescale_multiplier,
            scale_method=args.training.scale_method,
            use_tag_weighting=args.tag_weighting.use_tag_weighting,
            device=device,
            dtype=dtype
        )

        # Apply VAE finetuning loss if enabled
        if components.get("vae_finetuner") is not None:
            vae_loss = components["vae_finetuner"].compute_loss(batch)
            loss = loss + vae_loss * args.vae_loss_weight

        return loss

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error("Forward pass failed: %s", str(e))
        raise type(e)(f"Failed during forward pass: {str(e)}") from e
