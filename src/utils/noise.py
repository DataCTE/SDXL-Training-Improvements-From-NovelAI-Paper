import torch
import logging
from typing import Tuple
import traceback

logger = logging.getLogger(__name__)

def generate_noise(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype, 
                  noise_template: torch.Tensor) -> torch.Tensor:
    """Generate noise using pre-allocated template with error handling."""
    try:
        if not isinstance(shape, tuple):
            raise ValueError(f"Expected shape to be tuple, got {type(shape)}")
        if len(shape) != 4:
            raise ValueError(f"Expected 4D shape (B,C,H,W), got {len(shape)}D")

        return torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=None,
            layout=noise_template.layout
        )

    except Exception as e:
        logger.error(f"Error generating noise: {str(e)}")
        logger.error(f"Shape attempted: {shape}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 