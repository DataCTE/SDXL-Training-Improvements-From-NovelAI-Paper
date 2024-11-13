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
    Image is valid if at least one dimension is >= 1024px.
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        tuple: (bool, closest_bucket) - Valid flag and closest matching resolution
    """
    try:
        # Invalid if BOTH dimensions are below 1024
        if width < 1024 and height < 1024:
            return False, None
            
        # Check maximum dimension
        if width > 2048 or height > 2048:
            return False, None
            
        # Calculate aspect ratio of input image
        input_ar = width / height
        input_pixels = width * height
        
        # Get all valid SDXL buckets
        buckets = get_sdxl_bucket_resolutions()
        
        # Find closest matching bucket based on aspect ratio and total pixels
        min_diff_score = float('inf')
        closest_bucket = None
        
        for bucket_w, bucket_h in buckets:
            bucket_ar = bucket_w / bucket_h
            bucket_pixels = bucket_w * bucket_h
            
            # Calculate difference score based on both AR and total pixels
            ar_diff = abs(input_ar - bucket_ar)
            pixel_diff = abs(input_pixels - bucket_pixels) / max(input_pixels, bucket_pixels)
            diff_score = ar_diff * 2 + pixel_diff  # Weigh AR difference more heavily
            
            if diff_score < min_diff_score:
                min_diff_score = diff_score
                closest_bucket = (bucket_w, bucket_h)
        
        # Allow 30% tolerance in combined difference score
        max_diff = 0.3
        is_valid = min_diff_score <= max_diff
        
        return is_valid, closest_bucket
        
    except Exception as e:
        logger.error(f"Error validating dimensions: {str(e)}")
        return False, None

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
    

