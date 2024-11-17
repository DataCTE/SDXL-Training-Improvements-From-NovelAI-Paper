import logging
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class DatasetInitializer:
    """Handles dataset initialization and validation."""
    
    @staticmethod
    def validate_image_caption_pair(image_path: str) -> Tuple[bool, Optional[str]]:
        """Validate that both image and caption exist and are valid."""
        try:
            # Check image
            with Image.open(image_path) as img:
                img.verify()
            
            # Check caption
            caption_path = Path(image_path).with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                if caption:
                    return True, None
                return False, "Empty caption file"
            return False, "Missing caption file"
            
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def find_valid_pairs(data_dir: str) -> List[str]:
        """Find all valid image-caption pairs in the directory."""
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
        all_image_paths = []
        valid_paths = []
        skipped_count = 0
        
        # Find all image files
        data_path = Path(data_dir)
        for ext in image_extensions:
            all_image_paths.extend([str(p) for p in data_path.glob(ext)])
            all_image_paths.extend([str(p) for p in data_path.glob(ext.upper())])
            
        if not all_image_paths:
            raise RuntimeError(f"No images found in {data_dir}. Supported formats: {', '.join(image_extensions)}")
            
        # Validate each pair
        for img_path in all_image_paths:
            is_valid, error = DatasetInitializer.validate_image_caption_pair(img_path)
            if is_valid:
                valid_paths.append(img_path)
            else:
                logger.debug(f"Skipping {img_path}: {error}")
                skipped_count += 1
                
        # Log results
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} images due to missing or invalid caption files")
            
        if not valid_paths:
            raise RuntimeError(f"No valid image-caption pairs found in {data_dir}. Found {len(all_image_paths)} images but none had valid caption files.")
            
        logger.info(f"Found {len(valid_paths)} valid image-caption pairs out of {len(all_image_paths)} total images")
        
        return valid_paths
