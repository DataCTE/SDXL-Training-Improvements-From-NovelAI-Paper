import logging
import math
import os
import time
import gc
from typing import Dict, Any, Union, Optional, List
from functools import lru_cache
from multiprocessing import Manager
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import Adafactor
from bitsandbytes.optim import AdamW8bit

from src.inference.text_to_image import SDXLInference
from src.training.ema import EMAModel
from src.data.tag_weighter import TagBasedLossWeighter
from src.training.vae_finetuner import VAEFineTuner

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
            getattr(args, "optimizer_type", "adamw"),
            args.learning_rate,
            args.weight_decay,
            args.adam_beta1,
            args.adam_beta2,
            args.adam_epsilon,
        )

        # Initialize optimizer based on type
        if args.use_8bit_adam:
            try:
                optimizer = AdamW8bit(params_to_optimize, **opt_config)
            except ImportError as e:
                logger.warning("Failed to import 8-bit AdamW: %s. Falling back to regular AdamW.", str(e))
                args.use_8bit_adam = False
                optimizer = torch.optim.AdamW(params_to_optimize, **opt_config)
        elif args.use_adafactor:
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
    """Initialize VAE finetuner with proper configuration."""
    try:
        if not getattr(args, "finetune_vae", False):
            return None

        vae_config = {
            "device": getattr(args, "device", "cuda"),
            "mixed_precision": getattr(args, "mixed_precision", "no"),
            "use_amp": getattr(args, "use_amp", False),
            "learning_rate": getattr(args, "vae_learning_rate", 1e-6),
            "adam_beta1": getattr(args, "adam_beta1", 0.9),
            "adam_beta2": getattr(args, "adam_beta2", 0.999),
            "adam_epsilon": getattr(args, "adam_epsilon", 1e-8),
            "weight_decay": getattr(args, "weight_decay", 1e-2),
            "max_grad_norm": getattr(args, "max_grad_norm", 1.0),
            "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
            "use_8bit_adam": getattr(args, "use_8bit_adam", False),
            "use_channel_scaling": getattr(args, "vae_use_channel_scaling", False),
            "adaptive_loss_scale": getattr(args, "vae_adaptive_loss_scale", False),
            "kl_weight": getattr(args, "vae_kl_weight", 0.0),
            "perceptual_weight": getattr(args, "vae_perceptual_weight", 0.0),
            "min_snr_gamma": getattr(args, "min_snr_gamma", 5.0),
            "initial_scale_factor": getattr(args, "vae_initial_scale_factor", 1.0),
            "decay": getattr(args, "vae_decay", 0.9999),
            "update_after_step": getattr(args, "vae_update_after_step", 100),
            "model_path": getattr(args, "vae_model_path", None),
        }
        vae_finetuner = VAEFineTuner(vae=models["vae"], **vae_config)

        return vae_finetuner

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup VAE finetuner: {str(e)}") from e


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

        decay = getattr(args, "ema_decay", 0.9999)

        # Basic validation
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay:.6f}")

        ema_config = _get_ema_config(
            decay=decay,
            update_every=getattr(args, "ema_update_every", 10),
            device=device,
            model_path=args.model_path,
        )

        if args.use_ema:
            ema = EMAModel(model, **ema_config)

            return ema
        return None

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to setup EMA model: {str(e)}") from e


def setup_validator(args, models, device, dtype) -> Optional[Any]:
    """Initialize validation components."""
    try:
        if args.skip_validation:
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
        if not getattr(args, "use_tag_weighting", False):
            return None

        weighter_config = _get_tag_weighter_config(
            base_weight=getattr(args, "tag_base_weight", 1.0),
            min_weight=getattr(args, "min_tag_weight", 0.1),
            max_weight=getattr(args, "max_tag_weight", 3.0),
            window_size=getattr(args, "tag_window_size", 100),
            no_cache=getattr(args, "no_caching", False),
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
        return self._val

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._count


class MetricsManager:
    """Fully process-safe metrics manager."""

    def __init__(self):
        self._manager = Manager()
        self._metrics = self._manager.dict()

    def get_metric(self, name: str) -> AverageMeter:
        if name not in self._metrics:
            self._metrics[name] = AverageMeter(name=name)
        return self._metrics[name]

    def update_metric(self, name: str, value: float, n: int = 1) -> None:
        metric = self.get_metric(name)
        metric.update(value, n)
        self._metrics[name] = metric  # Update in shared dict

    def get_all_metrics(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in dict(self._metrics).items()}


def initialize_training_components(args, device, dtype, models):
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
                from diffusers import AutoencoderKL

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

        components["train_dataloader"] = DataLoader(
            dataset=dataloader_models["vae"].dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        # Setup metrics manager
        components["metrics_manager"] = MetricsManager()

        # Setup learning rate scheduler
        num_training_steps = args.num_epochs * len(components["train_dataloader"])
        components["lr_scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=components["optimizer"],
            T_max=num_training_steps,
            eta_min=1e-6,
        )

        # Setup optional components with parallel initialization
        optional_components = {
            "ema": (setup_ema, args.use_ema, (args, models["unet"])),
            "tag_weighter": (setup_tag_weighter, args.use_tag_weighting, (args,)),
            "vae_finetuner": (setup_vae_finetuner, args.finetune_vae, (args, models)),
        }

        components.update(
            {
                name: setup_func(*setup_args) if use_flag else None
                for name, (
                    setup_func,
                    use_flag,
                    setup_args,
                ) in optional_components.items()
            }
        )

        # Cache and set training configuration
        components["training_config"] = {
            "mode": getattr(args, "training_mode", "v_prediction"),
            "min_snr_gamma": getattr(args, "min_snr_gamma", 5.0),
            "sigma_data": getattr(args, "sigma_data", 1.0),
            "sigma_min": getattr(args, "sigma_min", 0.002),
            "sigma_max": getattr(args, "sigma_max", 80.0),
            "scale_method": getattr(args, "scale_method", "v"),
            "scale_factor": getattr(args, "scale_factor", 1.0),
        }

        # Setup validator
        components["validator"] = setup_validator(args, models, device, dtype)

        # Validate components
        _validate_components(components)

        return components

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed to initialize training components: {str(e)}") from e


def _validate_components(components: Dict[str, Any]) -> None:
    """Validate initialized components."""
    required = ["optimizer", "train_dataloader", "lr_scheduler", "training_config"]
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


@torch.no_grad()
def train_epoch(
    epoch: int, args, models, components, device, dtype, global_step: int
) -> Dict[str, float]:
    """Execute single training epoch with proper logging and memory management."""
    try:
        # Create metrics manager here in the main process
        metrics_manager = MetricsManager()
        components["metrics_manager"] = metrics_manager

        # Train for one epoch
        epoch_metrics = train(args, models, components, device, dtype)

        # Run validation if configured
        if hasattr(args, "validation_epochs") and epoch % args.validation_epochs == 0:
            validation_metrics = run_validation(
                args, models, components, device, dtype, global_step
            )
            epoch_metrics.update(validation_metrics)

        return epoch_metrics

    except (ValueError, RuntimeError, AttributeError) as e:
        raise type(e)(f"Failed during training epoch: {str(e)}") from e


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
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
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
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling
        if args.mixed_precision:
            components["scaler"].scale(loss).backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                components["scaler"].unscale_(components["optimizer"])
                torch.nn.utils.clip_grad_norm_(
                    models["unet"].parameters(), args.max_grad_norm
                )
                components["scaler"].step(components["optimizer"])
                components["scaler"].update()
                components["lr_scheduler"].step()
        else:
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    models["unet"].parameters(), args.max_grad_norm
                )
                components["optimizer"].step()
                components["lr_scheduler"].step()

        # Update EMA model if enabled
        if (
            components.get("ema") is not None
            and (batch_idx + 1) % args.gradient_accumulation_steps == 0
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
