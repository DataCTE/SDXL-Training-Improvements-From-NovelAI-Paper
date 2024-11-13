import logging
import os
import glob
from PIL import Image

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
    """
    Pre-process validation of all images in dataset, logging improper sizes but allowing them
    
    Args:
        data_dir (str): Directory containing the dataset
        
    Returns:
        tuple: (is_valid, stats)
    """
    try:
        logger.info(f"Validating dataset in {data_dir}")
        stats = {
            "total_images": 0,
            "valid_images": 0,
            "resized_images": [],
            "missing_captions": 0,
            "bucket_distribution": {}
        }
        
        # Get all image files
        image_files = glob.glob(os.path.join(data_dir, "*.png"))
        stats["total_images"] = len(image_files)
        
        if stats["total_images"] == 0:
            logger.error("No images found in dataset directory")
            return False, stats
            
        # Validate each image
        for img_path in image_files:
            try:
                # Check for caption file
                caption_path = os.path.splitext(img_path)[0] + ".txt"
                if not os.path.exists(caption_path):
                    logger.warning(f"Missing caption file for {img_path}")
                    stats["missing_captions"] += 1
                    continue
                
                # Validate image
                with Image.open(img_path) as img:
                    width, height = img.size
                    is_valid, closest_bucket = validate_image_dimensions(width, height)
                    
                    if closest_bucket:
                        bucket_key = f"{closest_bucket[0]}x{closest_bucket[1]}"
                        stats["bucket_distribution"][bucket_key] = stats["bucket_distribution"].get(bucket_key, 0) + 1
                        
                        if not is_valid:
                            stats["resized_images"].append({
                                "path": img_path,
                                "original": (width, height),
                                "target": closest_bucket
                            })
                    
                    # Check image mode
                    if img.mode != "RGB":
                        logger.warning(f"Converting non-RGB image to RGB: {img_path}")
                    
                stats["valid_images"] += 1
                
            except Exception as e:
                logger.error(f"Error validating {img_path}: {str(e)}")
                
        # Log validation results
        logger.info(f"Dataset validation complete:")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Valid images: {stats['valid_images']}")
        logger.info(f"Images requiring resize: {len(stats['resized_images'])}")
        logger.info(f"Missing captions: {stats['missing_captions']}")
        logger.info("\nBucket distribution:")
        for bucket, count in sorted(stats["bucket_distribution"].items()):
            logger.info(f"  {bucket}: {count} images")
        
        # Dataset is valid if we have at least one valid image
        is_valid = stats["valid_images"] > 0
        return is_valid, stats
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False, {"error": str(e)}
    

