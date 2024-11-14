import logging
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def validate_dataset(data_dir):
    """Basic dataset validation - only checks if directory exists and contains images"""
    try:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return False, "Dataset directory does not exist"
            
        image_files = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))
        if not image_files:
            return False, "No image files found in dataset directory"
            
        return True, {"num_images": len(image_files)}
        
    except Exception as e:
        return False, str(e)

def validate_image_dimensions(width, height):
    """Basic image dimension validation"""
    try:
        # only to small images
        if width < 256 or height < 256:
            return False, "Image is too small"
            
        return True, {"width": width, "height": height}
    except Exception as e:
        return False, str(e)
