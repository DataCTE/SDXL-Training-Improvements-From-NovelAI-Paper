import logging
import os
import glob
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

logger = logging.getLogger(__name__)


def verify_training_components(train_components):
    """Verify that all training components are properly configured"""
    try:
        # Check if required components exist
        if not train_components:
            return False, "No training components provided"
            
        # Validate text encoders if present
        if 'text_encoder' in train_components and 'text_encoder_2' in train_components:
            is_valid, error_msg = validate_text_encoders(
                train_components['text_encoder'],
                train_components['text_encoder_2']
            )
            if not is_valid:
                return False, f"Text encoder validation failed: {error_msg}"
                
        # Validate dataset configuration
        if 'dataset' in train_components:
            dataset = train_components['dataset']
            if not hasattr(dataset, 'data_dir') or not dataset.data_dir.exists():
                return False, "Dataset directory does not exist"
            if not hasattr(dataset, 'cache_dir'):
                return False, "Dataset missing cache directory configuration"
            if not (dataset.min_size <= dataset.max_size):
                return False, f"Invalid dataset size range: min_size ({dataset.min_size}) > max_size ({dataset.max_size})"
                
        # Validate tag weighter configuration
        if 'tag_weighter' in train_components:
            weighter = train_components['tag_weighter']
            if not hasattr(weighter, 'tag_classes'):
                return False, "Tag weighter missing tag classes configuration"
            if not (0 < weighter.min_weight <= weighter.max_weight):
                return False, f"Invalid tag weight range: min_weight ({weighter.min_weight}) > max_weight ({weighter.max_weight})"
                
        # Validate optimizer and scheduler
        if 'optimizer' in train_components and 'lr_scheduler' in train_components:
            is_valid, error_msg = validate_optimizer_config(
                train_components['optimizer'],
                train_components['lr_scheduler']
            )
            if not is_valid:
                return False, f"Optimizer validation failed: {error_msg}"
                
        # Validate training configuration
        if 'training_config' in train_components:
            is_valid, error_msg = validate_training_config(train_components['training_config'])
            if not is_valid:
                return False, f"Training configuration validation failed: {error_msg}"
                
        # Validate core model components
        required_components = ['vae', 'unet', 'noise_scheduler']
        for component in required_components:
            if component not in train_components:
                return False, f"Missing required component: {component}"
                
        return True, None
        
    except Exception as e:
        logger.error(f"Error verifying training components: {str(e)}")
        return False, f"Component verification failed: {str(e)}"


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

def validate_text_encoders(text_encoder, text_encoder_2):
    """
    Validate SDXL text encoder architectures and configurations.
    
    Args:
        text_encoder: First CLIP text encoder (base)
        text_encoder_2: Second CLIP text encoder (large)
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        # Check types
        if not isinstance(text_encoder, CLIPTextModel):
            return False, "First text encoder must be CLIPTextModel"
        if not isinstance(text_encoder_2, CLIPTextModel):
            return False, "Second text encoder must be CLIPTextModel"
            
        # Check hidden sizes
        if text_encoder.config.hidden_size != 768:
            return False, f"First text encoder (base) hidden size must be 768, got {text_encoder.config.hidden_size}"
        if text_encoder_2.config.hidden_size != 1280:
            return False, f"Second text encoder (large) hidden size must be 1280, got {text_encoder_2.config.hidden_size}"
            
        # Check model states
        if text_encoder.training:
            return False, "First text encoder must be in eval mode"
        if text_encoder_2.training:
            return False, "Second text encoder must be in eval mode"
            
        # Check gradient states
        if any(p.requires_grad for p in text_encoder.parameters()):
            return False, "First text encoder parameters should not require gradients"
        if any(p.requires_grad for p in text_encoder_2.parameters()):
            return False, "Second text encoder parameters should not require gradients"
            
        return True, None
        
    except Exception as e:
        logger.error(f"Text encoder validation failed: {str(e)}")
        return False, f"Text encoder validation failed: {str(e)}"

def validate_optimizer_config(optimizer, lr_scheduler):
    """Validate optimizer and learning rate scheduler configuration"""
    try:
        if not isinstance(optimizer, torch.optim.Optimizer):
            return False, "Invalid optimizer type"
            
        if len(list(optimizer.param_groups)) == 0:
            return False, "Optimizer has no parameter groups"
            
        if not hasattr(lr_scheduler, "get_last_lr"):
            return False, "Invalid learning rate scheduler"
            
        return True, None
    except Exception as e:
        return False, f"Optimizer validation failed: {str(e)}"

def validate_training_config(config):
    """Validate training hyperparameters and configuration"""
    try:
        required_params = {
            'gradient_accumulation_steps': (int, lambda x: x > 0),
            'max_grad_norm': (float, lambda x: x > 0),
            'mixed_precision': (bool, None),
            'min_snr_gamma': (float, lambda x: x >= 0),
            'sigma_data': (float, lambda x: x > 0)
        }
        
        for param, (param_type, validator) in required_params.items():
            if param not in config:
                return False, f"Missing required parameter: {param}"
                
            if not isinstance(config[param], param_type):
                return False, f"Invalid type for {param}: expected {param_type.__name__}"
                
            if validator and not validator(config[param]):
                return False, f"Invalid value for {param}: {config[param]}"
                
        return True, None
    except Exception as e:
        return False, f"Training config validation failed: {str(e)}"

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

def validate_model_components(models):
    """
    Validate model components for SDXL training
    
    Args:
        models (dict): Dictionary of model components
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        required_models = [
            "unet",
            "vae",
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2"
        ]
        
        # Check for required models
        for model_name in required_models:
            if model_name not in models:
                return False, f"Missing required model: {model_name}"
                
        # Validate text encoders
        is_valid, error_msg = validate_text_encoders(
            models["text_encoder"],
            models["text_encoder_2"]
        )
        if not is_valid:
            return False, error_msg
            
        # Validate VAE
        if not isinstance(models["vae"], AutoencoderKL):
            return False, "VAE must be an instance of AutoencoderKL"
        if models["vae"].training:
            return False, "VAE should be in eval mode"
        if any(p.requires_grad for p in models["vae"].parameters()):
            return False, "VAE parameters should not require gradients"
            
        # Validate UNet
        if not isinstance(models["unet"], UNet2DConditionModel):
            return False, "UNet must be an instance of UNet2DConditionModel"
            
        # Validate tokenizers
        if not isinstance(models["tokenizer"], CLIPTokenizer):
            return False, "Tokenizer must be an instance of CLIPTokenizer"
        if not isinstance(models["tokenizer_2"], CLIPTokenizer):
            return False, "Tokenizer 2 must be an instance of CLIPTokenizer"
            
        return True, None
        
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"

def validate_training_args(args):
    """
    Validate training arguments and configuration
    
    Args:
        args: Training arguments
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        required_args = {
            'data_dir': (str, lambda x: Path(x).exists(), "Data directory must exist"),
            'cache_dir': (str, None, None),
            'batch_size': (int, lambda x: x > 0, "Batch size must be positive"),
            'gradient_accumulation_steps': (int, lambda x: x > 0, "Gradient accumulation steps must be positive"),
            'max_grad_norm': (float, lambda x: x > 0, "Max gradient norm must be positive"),
            'mixed_precision': (bool, None, None),
            'num_workers': (int, lambda x: x >= 0, "Number of workers must be non-negative")
        }
        
        for arg_name, (arg_type, validator, error_msg) in required_args.items():
            if not hasattr(args, arg_name):
                return False, f"Missing required argument: {arg_name}"
                
            value = getattr(args, arg_name)
            if not isinstance(value, arg_type):
                return False, f"Invalid type for {arg_name}: expected {arg_type.__name__}"
                
            if validator and not validator(value):
                return False, error_msg
                
        return True, None
        
    except Exception as e:
        return False, f"Training arguments validation failed: {str(e)}"

def validate_ema_config(ema_config):
    """
    Validate EMA configuration
    
    Args:
        ema_config (dict): EMA configuration parameters
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        required_params = {
            'decay': (float, lambda x: 0 <= x <= 1, "Decay must be between 0 and 1"),
            'update_after_step': (int, lambda x: x >= 0, "Update after step must be non-negative"),
            'inv_gamma': (float, lambda x: x > 0, "Inverse gamma must be positive"),
            'power': (float, lambda x: x > 0, "Power must be positive"),
            'min_decay': (float, lambda x: 0 <= x <= 1, "Min decay must be between 0 and 1"),
            'max_decay': (float, lambda x: 0 <= x <= 1, "Max decay must be between 0 and 1"),
            'update_every': (int, lambda x: x > 0, "Update every must be positive"),
            'use_ema_warmup': (bool, None, None),
            'grad_scale_factor': (float, lambda x: x >= 0, "Gradient scale factor must be non-negative")
        }
        
        for param, (param_type, validator, error_msg) in required_params.items():
            if param not in ema_config:
                return False, f"Missing required EMA parameter: {param}"
                
            value = ema_config[param]
            if not isinstance(value, param_type):
                return False, f"Invalid type for {param}: expected {param_type.__name__}"
                
            if validator and not validator(value):
                return False, error_msg
                
        # Additional validation
        if ema_config['min_decay'] > ema_config['max_decay']:
            return False, f"Min decay ({ema_config['min_decay']}) cannot be greater than max decay ({ema_config['max_decay']})"
            
        return True, None
        
    except Exception as e:
        return False, f"EMA config validation failed: {str(e)}"

def validate_vae_finetuner_config(config):
    """
    Validate VAE finetuner configuration
    
    Args:
        config (dict): VAE finetuner configuration parameters
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        required_params = {
            'learning_rate': (float, lambda x: x > 0, "Learning rate must be positive"),
            'min_snr_gamma': (float, lambda x: x >= 0, "Min SNR gamma must be non-negative"),
            'adaptive_loss_scale': (bool, None, None),
            'kl_weight': (float, lambda x: x >= 0, "KL weight must be non-negative"),
            'perceptual_weight': (float, lambda x: x >= 0, "Perceptual weight must be non-negative"),
            'use_8bit_adam': (bool, None, None),
            'gradient_checkpointing': (bool, None, None),
            'mixed_precision': (str, lambda x: x in ["no", "fp16", "bf16"], "Mixed precision must be 'no', 'fp16', or 'bf16'"),
            'use_channel_scaling': (bool, None, None)
        }
        
        for param, (param_type, validator, error_msg) in required_params.items():
            if param not in config:
                return False, f"Missing required VAE finetuner parameter: {param}"
                
            value = config[param]
            if not isinstance(value, param_type):
                return False, f"Invalid type for {param}: expected {param_type.__name__}"
                
            if validator and not validator(value):
                return False, error_msg
                
        return True, None
        
    except Exception as e:
        return False, f"VAE finetuner config validation failed: {str(e)}"
