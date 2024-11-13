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
    Generate SDXL resolution buckets with dynamic aspect ratios
    
    Returns:
        list: List of (width, height) tuples representing valid SDXL resolutions
    """
    base_resolutions = [
        (1024, 1024),  # 1:1
        (1152, 896),   # 1.29:1
        (896, 1152),   # 1:1.29
        (1216, 832),   # 1.46:1
        (832, 1216),   # 1:1.46
        (1344, 768),   # 1.75:1
        (768, 1344),   # 1:1.75
        (1536, 640),   # 2.4:1
        (640, 1536),   # 1:2.4
    ]
    
    # Generate scaled versions (0.5x to 1.5x)
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    buckets = []
    
    for width, height in base_resolutions:
        for scale in scale_factors:
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)
            
            # Ensure minimum dimension of 512
            if scaled_w >= 512 and scaled_h >= 512:
                # Ensure maximum dimension of 2048
                if scaled_w <= 2048 and scaled_h <= 2048:
                    buckets.append((scaled_w, scaled_h))
    
    return sorted(set(buckets))  # Remove duplicates and sort

def validate_image_dimensions(width, height):
    """
    Check if image dimensions are close enough to any SDXL bucket resolution
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        tuple: (bool, closest_bucket) - Valid flag and closest matching resolution
    """
    try:
        # Basic boundary checks
        if width < 512 or height < 512:
            return False, None
        if width > 2048 or height > 2048:
            return False, None
            
        # Get all valid SDXL buckets
        buckets = get_sdxl_bucket_resolutions()
        
        # Calculate aspect ratio of input image
        input_ar = width / height
        
        # Find closest matching bucket
        min_ar_diff = float('inf')
        closest_bucket = None
        
        for bucket_w, bucket_h in buckets:
            bucket_ar = bucket_w / bucket_h
            ar_diff = abs(input_ar - bucket_ar)
            
            if ar_diff < min_ar_diff:
                min_ar_diff = ar_diff
                closest_bucket = (bucket_w, bucket_h)
        
        # Allow 20% tolerance in aspect ratio difference
        max_ar_diff = 0.2
        is_valid = min_ar_diff <= max_ar_diff
        
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
    

