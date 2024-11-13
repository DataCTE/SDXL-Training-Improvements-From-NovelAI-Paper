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
            "improper_sized_images": [],
            "missing_captions": 0
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
                    if not validate_image_dimensions(width, height):
                        logger.info(f"Note: Image with non-standard dimensions: {img_path} ({width}x{height})")
                        stats["improper_sized_images"].append((img_path, width, height))
                        
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
        logger.info(f"Images with non-standard dimensions: {len(stats['improper_sized_images'])}")
        logger.info(f"Missing captions: {stats['missing_captions']}")
        
        # Dataset is valid if we have at least one valid image
        is_valid = stats["valid_images"] > 0
        return is_valid, stats
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False, {"error": str(e)}

def validate_image_dimensions(width, height):
    """
    Check if image dimensions are within SDXL preferred ranges
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        bool: True if dimensions are within preferred range
    """
    try:
        # Preferred dimensions
        min_size = 256
        max_size = 2048
        
        # Check dimensions
        if width < min_size or height < min_size:
            return False
            
        if width > max_size or height > max_size:
            return False
            
        # Check aspect ratio (maximum 2:1 or 1:2)
        aspect_ratio = width / height
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating dimensions: {str(e)}")
        return False
    

