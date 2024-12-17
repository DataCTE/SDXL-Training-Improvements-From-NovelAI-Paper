from PIL import Image
import torch
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path
import gc

logger = logging.getLogger(__name__)

def load_and_validate_image(
    path: str,
    min_size: Tuple[int, int] = (32, 32),
    max_size: Tuple[int, int] = (2048, 2048),
    required_modes: Tuple[str, ...] = ('RGB', 'RGBA')
) -> Optional[Image.Image]:
    """Load and validate an image file."""
    img = None
    try:
        with Image.open(path) as temp_img:
            # Validate mode
            if temp_img.mode not in required_modes:
                temp_img = temp_img.convert('RGB')
                
            # Validate dimensions
            width, height = temp_img.size
            if width < min_size[0] or height < min_size[1]:
                logger.debug(f"Image too small: {width}x{height} < {min_size}")
                return None
                
            if width > max_size[0] or height > max_size[1]:
                logger.debug(f"Image too large: {width}x{height} > {max_size}")
                return None
                
            # Make a copy to ensure file is closed
            img = temp_img.copy()
            
    except Exception as e:
        logger.debug(f"Error loading image {path}: {e}")
        return None
        
    finally:
        # Force garbage collection if we had any failed operations
        if img is None:
            gc.collect()
            
    return img

def resize_image(
    img: Image.Image,
    target_size: Tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """Resize image if needed."""
    if img.size != target_size:
        try:
            resized = img.resize(target_size, resampling)
            # Close original if we created a new image
            if resized is not img:
                img.close()
            return resized
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
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