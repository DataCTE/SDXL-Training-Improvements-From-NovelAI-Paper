import torch
import logging
from typing import Dict, Any, Optional
from contextlib import nullcontext
from src.training.metrics import MetricsManager
from src.training.loss import forward_pass

logger = logging.getLogger(__name__)

def run_validation(
    args,
    models: Dict[str, Any],
    components: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    metrics_manager: Optional[MetricsManager] = None,
) -> Dict[str, float]:
    """Run validation with proper error handling and metrics tracking."""
    try:
        # Set eval mode
        models["unet"].eval()
        if getattr(args, "train_text_encoder", False):
            models["text_encoder"].eval()
            models["text_encoder_2"].eval()

        # Get validation dataloader
        val_dataloader = components.get("val_dataloader")
        if val_dataloader is None:
            logger.warning("No validation dataloader found, skipping validation")
            return {}

        total_val_loss = 0.0
        num_val_steps = 0

        # Run validation steps
        with torch.no_grad():
            for batch in val_dataloader:
                with torch.cuda.amp.autocast() if args.mixed_precision != "no" else nullcontext():
                    val_loss = forward_pass(args, models, batch, device, dtype, components)
                total_val_loss += val_loss.item()
                num_val_steps += 1

        # Compute average validation loss
        avg_val_loss = total_val_loss / num_val_steps if num_val_steps > 0 else 0.0

        # Update metrics
        metrics = {"val_loss": avg_val_loss}
        if metrics_manager is not None:
            for name, value in metrics.items():
                metrics_manager.update_metric(name, value)

        # Log validation results
        logger.info(
            "Validation [Step %d] Average Loss: %.6f",
            global_step,
            avg_val_loss,
        )

        return metrics

    except Exception as e:
        logger.error("Validation failed: %s", str(e))
        raise