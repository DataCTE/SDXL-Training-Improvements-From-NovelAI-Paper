import torch
import logging
from typing import List, Tuple
import traceback

logger = logging.getLogger(__name__)

def get_add_time_ids(
    original_sizes: List[Tuple[int, int]],
    crops_coords_top_lefts: List[Tuple[int, int]],
    target_sizes: List[Tuple[int, int]],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """Get time embeddings with error handling."""
    try:
        # Validate inputs
        if not all(isinstance(size, tuple) and len(size) == 2 for size in original_sizes):
            raise ValueError("Invalid original_sizes format")
        if not all(isinstance(coord, tuple) and len(coord) == 2 for coord in crops_coords_top_lefts):
            raise ValueError("Invalid crops_coords_top_lefts format")
        if not all(isinstance(size, tuple) and len(size) == 2 for size in target_sizes):
            raise ValueError("Invalid target_sizes format")
        
        add_time_ids = torch.zeros(
            (batch_size, 6),
            dtype=dtype,
            device=device
        )

        for i, (orig_size, crop_coords, target_size) in enumerate(
            zip(original_sizes, crops_coords_top_lefts, target_sizes)
        ):
            add_time_ids[i] = torch.tensor([
                orig_size[0],
                orig_size[1],
                crop_coords[0],
                crop_coords[1],
                target_size[0],
                target_size[1]
            ], dtype=dtype, device=device)

        return add_time_ids

    except Exception as e:
        logger.error(f"Error in get_add_time_ids: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 