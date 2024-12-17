import os
import logging
from typing import Tuple, List
from src.config.config import Config

logger = logging.getLogger(__name__)

def validate_directories(config: Config) -> Tuple[List[str], int]:
    """Validate image directories and count total images."""
    try:
        if not config.data.image_dirs:
            raise ValueError(
                "No image directories specified in config. Please add valid image directories to config.data.image_dirs"
            )
            
        total_images = 0
        valid_dirs = []
        
        for img_dir in config.data.image_dirs:
            if not os.path.exists(img_dir):
                logger.warning(f"Directory not found: {img_dir}")
                continue
                
            num_images = len([f for f in os.listdir(img_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            
            if num_images == 0:
                logger.warning(f"No valid images found in directory: {img_dir}")
                continue
                
            total_images += num_images
            valid_dirs.append(img_dir)
            
        if total_images == 0:
            raise ValueError(
                "No valid images found in any of the specified directories. "
                "Please ensure the directories contain supported image files "
                "(.png, .jpg, .jpeg, .webp)"
            )
            
        return valid_dirs, total_images

    except Exception as e:
        logger.error(f"Failed to validate directories: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise 