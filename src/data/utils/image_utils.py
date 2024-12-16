from PIL import Image
import torch
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_and_validate_image(
    path: str,
    min_size: Tuple[int, int] = (32, 32),
    max_size: Tuple[int, int] = (8192, 8192),
    required_modes: Tuple[str, ...] = ('RGB', 'RGBA')
) -> Optional[Image.Image]:
    """Load and validate an image file."""
    try:
        with Image.open(path) as img:
            # Validate mode
            if img.mode not in required_modes:
                img = img.convert('RGB')
                
            # Validate dimensions
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                logger.debug(f"Image too small: {width}x{height} < {min_size}")
                return None
                
            if width > max_size[0] or height > max_size[1]:
                logger.debug(f"Image too large: {width}x{height} > {max_size}")
                return None
                
            return img.copy()
            
    except Exception as e:
        logger.debug(f"Error loading image {path}: {e}")
        return None

def resize_image(
    img: Image.Image,
    target_size: Tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """Resize image if needed."""
    if img.size != target_size:
        return img.resize(target_size, resampling)
    return img

def get_image_stats(img: Image.Image) -> Dict:
    """Get image statistics."""
    return {
        'width': img.width,
        'height': img.height,
        'mode': img.mode,
        'aspect_ratio': img.width / img.height,
        'resolution': img.width * img.height
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