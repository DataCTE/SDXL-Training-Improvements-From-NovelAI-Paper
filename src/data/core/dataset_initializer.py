"""Dataset initialization and validation module.

This module handles the initialization and validation of image-caption pairs
for the SDXL training pipeline. It includes functionality for finding valid
pairs, loading captions, and ensuring data integrity.

Classes:
    DatasetInitializer: Main class for dataset initialization operations
    DatasetError: Base exception for dataset-related errors
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

from PIL import Image
from PIL.Image import DecompressionBombError

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Base exception for dataset-related errors."""


class ImageLoadError(DatasetError):
    """Exception raised when an image cannot be loaded or verified."""


class CaptionLoadError(DatasetError):
    """Exception raised when a caption cannot be loaded."""


class DatasetInitializer:
    """Handles dataset initialization and validation.
    
    This class provides methods for finding and validating image-caption pairs,
    loading captions, and ensuring dataset integrity. It supports multiple image
    formats and handles various error conditions gracefully.
    
    Attributes:
        SUPPORTED_FORMATS: Set of supported image file extensions
    """
    
    SUPPORTED_FORMATS: Set[str] = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp",
        ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"
    }
    
    @classmethod
    def validate_image_caption_pair(cls, image_path: str) -> Tuple[bool, Optional[str]]:
        """Validate that both image and caption exist and are valid.
        
        Args:
            image_path: Path to the image file to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if the pair is valid
                - Optional error message if validation failed
                
        Note:
            An image-caption pair is considered valid if:
            1. The image file exists and can be opened
            2. The corresponding .txt caption file exists
            3. The caption file is not empty
        """
        try:
            # Validate image path
            if Path(image_path).suffix not in cls.SUPPORTED_FORMATS:
                return False, f"Unsupported image format: {Path(image_path).suffix}"
            
            # Check image
            with Image.open(image_path) as img:
                img.verify()
            
            # Check caption
            caption_path = Path(image_path).with_suffix('.txt')
            if not caption_path.exists():
                return False, "Missing caption file"
                
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
                if not caption:
                    return False, "Empty caption file"
                
            return True, None
            
        except (OSError, DecompressionBombError) as error:
            return False, f"Failed to open image: {str(error)}"
        except UnicodeDecodeError as error:
            return False, f"Failed to read caption (invalid encoding): {str(error)}"
        except Exception as error:
            return False, f"Unexpected error: {str(error)}"
    
    @classmethod
    def find_valid_pairs(cls, data_dir: str) -> List[str]:
        """Find all valid image-caption pairs in the directory.
        
        Args:
            data_dir: Directory to search for image-caption pairs
            
        Returns:
            List of paths to valid image files
            
        Raises:
            RuntimeError: If no valid image-caption pairs are found
            
        Note:
            A pair is considered valid if it passes validate_image_caption_pair()
        """
        all_image_paths: List[str] = []
        valid_paths: List[str] = []
        skipped_count = 0
        data_path = Path(data_dir)
        
        # Find all image files
        for ext in cls.SUPPORTED_FORMATS:
            pattern = f"*{ext}"
            all_image_paths.extend([str(p) for p in data_path.glob(pattern)])
            
        if not all_image_paths:
            raise RuntimeError(
                f"No images found in {data_dir}. "
                f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}"
            )
            
        # Validate each pair
        for img_path in all_image_paths:
            is_valid, error = cls.validate_image_caption_pair(img_path)
            if is_valid:
                valid_paths.append(img_path)
            else:
                logger.debug("Skipping %s: %s", img_path, error)
                skipped_count += 1
                
        # Log results
        if skipped_count > 0:
            logger.warning(
                "Skipped %d images due to missing or invalid caption files",
                skipped_count
            )
            
        if not valid_paths:
            raise RuntimeError(
                f"No valid image-caption pairs found in {data_dir}. "
                f"Found {len(all_image_paths)} images but none had valid caption files."
            )
            
        logger.info(
            "Found %d valid image-caption pairs out of %d total images",
            len(valid_paths), len(all_image_paths)
        )
        
        return valid_paths

    @classmethod
    def load_captions(cls, image_paths: List[str]) -> Dict[str, str]:
        """Load captions for the given image paths.
        
        Args:
            image_paths: List of paths to images to load captions for
            
        Returns:
            Dictionary mapping image paths to their captions
            
        Note:
            If a caption file cannot be loaded, an empty string is used
            as a fallback and an error is logged.
        """
        captions: Dict[str, str] = {}
        
        for img_path in image_paths:
            caption_path = Path(img_path).with_suffix('.txt')
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                captions[img_path] = caption
            except (OSError, UnicodeDecodeError) as error:
                logger.error(
                    "Failed to load caption for %s: %s",
                    img_path, str(error)
                )
                captions[img_path] = ""  # Use empty string as fallback
                
        return captions
