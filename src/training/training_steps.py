import torch
import logging
from typing import Dict, Any, Optional
from contextlib import nullcontext
from src.training.metrics import MetricsManager
from src.training.loss_functions import forward_pass

logger = logging.getLogger(__name__)

def train_step(
    args,
    models: Dict[str, Any],
    components: Dict[str, Any],
    batch: Dict[str, Any],
    batch_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    metrics_manager: Optional[MetricsManager] = None,
) -> Dict[str, float]:
    """Execute single training step with proper error handling."""
    try:
        # Set train mode
        models["unet"].train()
        if getattr(args, "train_text_encoder", False):
            models["text_encoder"].train()
            models["text_encoder_2"].train()

        # Get optimizer and scheduler
        optimizer = components["optimizer"]
        scheduler = components.get("scheduler")
        scaler = components.get("scaler")

        # Forward pass
        with torch.cuda.amp.autocast() if args.mixed_precision != "no" else nullcontext():
            loss = forward_pass(args, models, batch, device, dtype, components)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(models["unet"].parameters(), args.max_grad_norm)
            optimizer.step()

        # Update EMA model if enabled
        if components.get("ema_model") is not None:
            components["ema_model"].step(models["unet"])

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Update metrics
        metrics = {"loss": loss.item()}
        if metrics_manager is not None:
            for name, value in metrics.items():
                metrics_manager.update_metric(name, value, step=batch_idx)

        # Log progress if needed
        if batch_idx % getattr(args, "logging_steps", 50) == 0:
            logger.info(f"Step {batch_idx}: Loss = {loss.item():.4f}")

        return metrics

    except Exception as e:
        logger.error("Training step failed: %s", str(e))
        raise

def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm with proper error handling."""
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    except Exception as e:
        logger.error("Failed to calculate gradient norm: %s", str(e))
        return 0.0
