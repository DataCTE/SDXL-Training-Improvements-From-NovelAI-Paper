import logging
import os
import glob
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


def verify_training_components(train_components):
    """
    Verify that all required training components are present and properly configured
    
    Args:
        train_components (dict): Dictionary of training components
        
    Returns:
        bool: True if verification passes
    """
    required_components = [
        "dataset",
        "train_dataloader",
        "optimizer",
        "lr_scheduler",
        "tag_weighter",
        "num_update_steps_per_epoch",
        "num_training_steps"
    ]
    
    try:
        # Check for required components
        for component_name in required_components:
            if component_name not in train_components:
                raise ValueError(f"Missing required component: {component_name}")
            
        # Verify dataloader
        if len(train_components["train_dataloader"]) == 0:
            raise ValueError("Empty training dataloader")
            
        # Verify optimizer
        if len(list(train_components["optimizer"].param_groups)) == 0:
            raise ValueError("Optimizer has no parameter groups")
            
        return True
        
    except Exception as e:
        logger.error(f"Training component verification failed: {str(e)}")
        return False


def get_sdxl_bucket_resolutions():
    """
    Generate SDXL resolution buckets dynamically based on common multipliers.
    Valid if either dimension is >= 1024px.
    
    Returns:
        list: List of (width, height) tuples representing valid SDXL resolutions
    """
    buckets = set()
    
    # Base sizes to scale from
    base_sizes = [1024, 1280, 1536, 1792, 2048]
    
    # Aspect ratio multipliers
    ar_multipliers = [
        1.0,    # 1:1
        1.25,   # 5:4
        1.33,   # 4:3
        1.5,    # 3:2
        1.77,   # 16:9
        2.0     # 2:1
    ]
    
    for base in base_sizes:
        for multiplier in ar_multipliers:
            # Calculate dimensions for both landscape and portrait
            width = int(base * multiplier)
            height = base
            
            # Add landscape variant if valid
            if width <= 2048 and (width >= 1024 or height >= 1024):
                buckets.add((width, height))
            
            # Add portrait variant if valid and not square
            if multiplier != 1.0 and height <= 2048 and (width >= 1024 or height >= 1024):
                buckets.add((height, width))
    
    return sorted(buckets)

def validate_image_dimensions(width, height):
    """
    Check if image dimensions are valid for SDXL.
    Only intervenes for extreme aspect ratios or very small/large dimensions.
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        tuple: (bool, closest_bucket) - Valid flag and closest matching resolution
    """
    try:
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Only invalid in these cases:
        # 1. If BOTH dimensions are very small (< 512px)
        if width < 512 and height < 512:
            return False, None
            
        # 2. If ANY dimension is extremely large (> 2560px)
        if width > 2560 or height > 2560:
            return False, None
            
        # 3. If aspect ratio is extremely skewed (> 4:1 or < 1:4)
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            return False, None
            
        # 4. If smallest dimension is tiny (< 384px) while other is normal/large
        min_dim = min(width, height)
        max_dim = max(width, height)
        if min_dim < 384 and max_dim > 768:
            return False, None
            
        # Otherwise, the image is valid - keep original dimensions
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating dimensions: {str(e)}")
        return True, None  # Default to keeping original dimensions

def validate_dataset(data_dir):
    """Validate dataset structure and contents"""
    data_dir = Path(data_dir)
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'resize_needed': 0,
        'missing_captions': 0,
        'buckets': defaultdict(int)
    }
    
    # Get all image files with supported extensions
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        image_files.extend(data_dir.glob(ext))
    
    stats['total_images'] = len(image_files)
    
    # Process each image
    for img_path in tqdm(image_files, desc="Validating dataset"):
        caption_path = img_path.with_suffix('.txt')
        
        try:
            # Check if caption exists
            if not caption_path.exists():
                stats['missing_captions'] += 1
                logger.warning(f"Missing caption file for {img_path}")
                continue
                
            # Validate image
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
                img = Image.open(img_path)  # Reopen after verify
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Check dimensions
                width, height = img.size
                is_valid, needs_resize = validate_image_dimensions(width, height)
                
                if needs_resize:
                    stats['resize_needed'] += 1
                
                if is_valid:
                    stats['valid_images'] += 1
                    # Record bucket information
                    bucket = f"{width}x{height}"
                    stats['buckets'][bucket] += 1
                    
        except Exception as e:
            logger.error(f"Error validating {img_path}: {str(e)}")
            continue
    
    # Log statistics
    logger.info("Dataset validation complete:")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Valid images: {stats['valid_images']}")
    logger.info(f"Images requiring resize: {stats['resize_needed']}")
    logger.info(f"Missing captions: {stats['missing_captions']}")
    logger.info("\nBucket distribution:")
    
    # Sort buckets by count
    sorted_buckets = sorted(stats['buckets'].items(), key=lambda x: x[1], reverse=True)
    for bucket, count in sorted_buckets[:10]:  # Show top 10 buckets
        logger.info(f"{bucket}: {count} images")
    
    # Validation passes if we have valid images
    return stats['valid_images'] > 0, stats
    

