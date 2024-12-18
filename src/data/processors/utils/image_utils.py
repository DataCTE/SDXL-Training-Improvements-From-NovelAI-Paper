from PIL import Image
import torch
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path
import gc
from src.config.config import VAEEncoderConfig
from src.utils.logging.metrics import log_error_with_context, log_metrics
import os

logger = logging.getLogger(__name__)

def load_and_validate_image(
    path: str,
    config: VAEEncoderConfig,
    required_modes: Tuple[str, ...] = ('RGB', 'RGBA')
) -> Optional[Image.Image]:
    """Load and validate an image file using config settings."""
    try:
        with Image.open(path) as temp_img:
            # Track image stats
            stats = {
                'original_size': temp_img.size,
                'original_mode': temp_img.mode,
                'file_size': os.path.getsize(path) / 1024  # KB
            }
            
            # Validate mode
            if temp_img.mode not in required_modes:
                stats['mode_conversion'] = f"{temp_img.mode} -> RGB"
                temp_img = temp_img.convert('RGB')
                
            # Validate dimensions
            width, height = temp_img.size
            if width < config.min_image_size[0] or height < config.min_image_size[1]:
                logger.debug(f"Image too small: {width}x{height} < {config.min_image_size}")
                stats['validation_error'] = 'size_too_small'
                return None
                
            if width > config.max_image_size[0] or height > config.max_image_size[1]:
                logger.debug(f"Image too large: {width}x{height} > {config.max_image_size}")
                stats['validation_error'] = 'size_too_large'
                return None
                
            # Make a copy and log stats
            img = temp_img.copy()
            stats.update({
                'final_size': img.size,
                'final_mode': img.mode,
                'aspect_ratio': width / height
            })
            
            # Log metrics periodically (every 100 images)
            if hasattr(load_and_validate_image, '_counter'):
                load_and_validate_image._counter += 1
            else:
                load_and_validate_image._counter = 1
                
            if load_and_validate_image._counter % 100 == 0:
                log_metrics(stats, step=load_and_validate_image._counter, step_type="image")
                
            return img
            
    except Exception as e:
        log_error_with_context(e, f"Error loading image {path}")
        return None
        
    finally:
        # Force garbage collection if we had any failed operations
        if 'img' not in locals():
            gc.collect()

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