import torch
from torchvision import transforms
import logging
import traceback

logger = logging.getLogger(__name__)

def ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has exactly three channels."""
    try:
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W), got {x.dim()}D")
        return x[:3]
    except Exception as e:
        logger.error(f"Error in ensure_three_channels: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def convert_to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to bfloat16 format."""
    try:
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x)}")
        return x.to(torch.bfloat16)
    except Exception as e:
        logger.error(f"Error in convert_to_bfloat16: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def get_transform() -> transforms.Compose:
    """Get composed transformation pipeline."""
    try:
        return transforms.Compose([
            transforms.ToTensor(),
            ensure_three_channels,
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            convert_to_bfloat16
        ])
    except Exception as e:
        logger.error(f"Error in get_transform: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 