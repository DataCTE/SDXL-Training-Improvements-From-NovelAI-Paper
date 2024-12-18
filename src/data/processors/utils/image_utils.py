from PIL import Image
import torch
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import gc
from src.config.config import VAEEncoderConfig, DEFAULT_MIN_IMAGE_SIZE, DEFAULT_MAX_IMAGE_SIZE
from src.utils.logging.metrics import log_error_with_context, log_metrics
import os

logger = logging.getLogger(__name__)

def load_and_validate_image(
    path: str,
    config: Any,
    min_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None,
    required_modes: Optional[Tuple[str, ...]] = None
) -> Optional[Image.Image]:
    """Load and validate image with size constraints, optionally enforcing specific modes."""
    try:
        min_image_size = min_size or getattr(config, 'min_image_size', (256, 256))
        max_image_size = max_size or getattr(config, 'max_image_size', (2048, 2048))

        image = Image.open(path)

        if required_modes and image.mode not in required_modes:
            # Convert if possible, or simply log a warning
            if 'RGB' in required_modes:
                image = image.convert('RGB')
            else:
                logger.warning(f"Image mode {image.mode} not in {required_modes}")
                return None

        width, height = image.size
        if width < min_image_size[0] or height < min_image_size[1]:
            logger.warning(
                f"Image too small: {path} ({width}x{height}), "
                f"minimum size: {min_image_size}"
            )
            return None

        if width > max_image_size[0] or height > max_image_size[1]:
            logger.warning(
                f"Image too large: {path} ({width}x{height}), "
                f"maximum size: {max_image_size}"
            )
            return None

        return image

    except Exception as e:
        log_error_with_context(e, f"Error loading image {path}")
        return None

def resize_image(
    img: Image.Image,
    target_size: Tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """Resize image if needed with metrics tracking."""
    if img.size != target_size:
        try:
            resize_stats = {
                'original_size': img.size,
                'target_size': target_size,
                'scale_factor': (
                    target_size[0] / img.size[0],
                    target_size[1] / img.size[1]
                )
            }
            
            resized = img.resize(target_size, resampling)
            
            # Update stats
            resize_stats['final_size'] = resized.size
            resize_stats['memory_change'] = (
                resized.size[0] * resized.size[1] - 
                img.size[0] * img.size[1]
            ) / (1024 * 1024)  # MB
            
            # Log metrics periodically
            if hasattr(resize_image, '_counter'):
                resize_image._counter += 1
            else:
                resize_image._counter = 1
                
            if resize_image._counter % 100 == 0:
                log_metrics(resize_stats, step=resize_image._counter, step_type="resize")
            
            # Close original if we created a new image
            if resized is not img:
                img.close()
                
            return resized
            
        except Exception as e:
            log_error_with_context(e, f"Error resizing image from {img.size} to {target_size}")
            return img
            
    return img

def get_image_stats(img: Image.Image) -> Dict:
    """Get image statistics."""
    return {
        'width': img.width,
        'height': img.height,
        'mode': img.mode,
        'aspect_ratio': img.width / img.height,
        'resolution': img.width * img.height,
        'memory_size': img.width * img.height * len(img.getbands())  # Approximate memory size in bytes
    }

def validate_image_text_pair(
    img_path: str,
    txt_path: Optional[str] = None,
    min_text_size: int = 1
) -> Tuple[bool, str]:
    """Validate image-text pair exists and is valid."""
    if not Path(img_path).exists():
        return False, "Image file not found"
        
    if txt_path is None:
        txt_path = str(Path(img_path).with_suffix('.txt'))
        
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return False, "Text file not found"
        
    if txt_path.stat().st_size < min_text_size:
        return False, "Text file empty"
        
    return True, "Valid pair"

def cleanup_image_resources(img: Optional[Image.Image] = None) -> None:
    """Clean up image resources and force garbage collection."""
    try:
        if img is not None:
            img.close()
    except Exception as e:
        logger.debug(f"Error closing image: {e}")
    finally:
        gc.collect()