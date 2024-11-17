"""
Image loading functionality for SDXL training pipeline.

This module provides utilities for loading and verifying images from files,
with support for various formats and error handling.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image

from .validation import ValidationError, validate_image

logger = logging.getLogger(__name__)


def load_and_verify_image(image_path: str) -> Optional[Image.Image]:
    """Load and verify an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Optional[Image.Image]: Loaded PIL Image if successful, None otherwise
        
    Note:
        Images are converted to RGB mode if necessary.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
            # Reopen because verify() invalidates the image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            validate_image(img)
            return img
    except Exception as error:
        logger.debug(
            "Failed to load image %s: %s",
            image_path, str(error)
        )
        return None


def get_image_dimensions(img: Image.Image) -> Tuple[int, int]:
    """Get image dimensions (height, width).
    
    Args:
        img: PIL Image
        
    Returns:
        Tuple[int, int]: Image dimensions (height, width)
    """
    return img.size[1], img.size[0]


def load_image_batch(
    image_paths: list[str],
    convert_to_rgb: bool = True
) -> list[Optional[Image.Image]]:
    """Load a batch of images.
    
    Args:
        image_paths: List of paths to image files
        convert_to_rgb: Whether to convert images to RGB mode
        
    Returns:
        List of loaded PIL Images, None for failed loads
    """
    images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
                img = Image.open(path)
                if convert_to_rgb and img.mode != 'RGB':
                    img = img.convert('RGB')
                validate_image(img)
                images.append(img)
        except Exception as error:
            logger.debug(
                "Failed to load image %s: %s",
                path, str(error)
            )
            images.append(None)
    return images


def is_valid_image_file(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid image.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file is a valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except:
        return False