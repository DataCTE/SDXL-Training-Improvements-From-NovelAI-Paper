import torch
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from src.data.dataset import CustomDataset
from src.data.core.dataset_initializer import DatasetInitializer
from src.data.multiaspect.bucket_manager import BucketManager
from src.data.cacheing.latent_cache import LatentCacheManager
from src.data.prompt.caption_processor import CaptionProcessor
from src.data.multiaspect.image_grouper import ImageGrouper
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def create_train_dataloader(
    data_dir: Union[str, Path],
    vae: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_2: Optional[Any] = None,
    text_encoder: Optional[Any] = None,
    text_encoder_2: Optional[Any] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    no_caching_latents: bool = False,
    all_ar: bool = True,
    min_size: int = 512,
    max_size: int = 1024,
    bucket_step_size: int = 64,
    max_bucket_area: int = 1024 * 1024,
    token_dropout_rate: float = 0.1,
    caption_dropout_rate: float = 0.1,
    min_tag_weight: float = 0.5,
    max_tag_weight: float = 2.0,
    use_tag_weighting: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create training dataloader with all SDXL training features.
    
    Args:
        data_dir: Directory containing training data
        vae: Optional VAE model for latent caching
        tokenizer: Primary tokenizer for text processing
        tokenizer_2: Secondary tokenizer for SDXL
        text_encoder: Primary text encoder
        text_encoder_2: Secondary text encoder for SDXL
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching latents
        no_caching_latents: Whether to disable latent caching
        all_ar: Whether to use aspect ratio bucketing
        min_size: Minimum image size
        max_size: Maximum image size
        bucket_step_size: Resolution step size for buckets
        max_bucket_area: Maximum area for buckets
        token_dropout_rate: Token dropout probability
        caption_dropout_rate: Caption dropout probability
        min_tag_weight: Minimum weight for tags
        max_tag_weight: Maximum weight for tags
        use_tag_weighting: Whether to enable tag-based loss weighting
        **kwargs: Additional dataset configuration
        
    Returns:
        Configured DataLoader for training
    """
    try:
        # Initialize dataset components
        dataset = CustomDataset(
            data_dir=data_dir,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            cache_dir=cache_dir,
            no_caching_latents=no_caching_latents,
            all_ar=all_ar,
            min_size=min_size,
            max_size=max_size,
            bucket_step_size=bucket_step_size,
            max_bucket_area=max_bucket_area,
            token_dropout_rate=token_dropout_rate,
            caption_dropout_rate=caption_dropout_rate,
            min_tag_weight=min_tag_weight,
            max_tag_weight=max_tag_weight,
            use_tag_weighting=use_tag_weighting,
            **kwargs
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )
        
        logger.info(f"Created training dataloader with {len(dataset)} samples")
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create training dataloader: {str(e)}")
        raise

def create_validation_dataloader(
    data_dir: Union[str, Path],
    vae: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_2: Optional[Any] = None,
    text_encoder: Optional[Any] = None,
    text_encoder_2: Optional[Any] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs
) -> Optional[DataLoader]:
    """
    Create validation dataloader.
    
    Args:
        data_dir: Directory containing validation data
        vae: Optional VAE model for latent caching
        tokenizer: Primary tokenizer for text processing
        tokenizer_2: Secondary tokenizer for SDXL
        text_encoder: Primary text encoder
        text_encoder_2: Secondary text encoder for SDXL
        batch_size: Batch size for validation
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching latents
        min_size: Minimum image size
        max_size: Maximum image size
        **kwargs: Additional dataset configuration
        
    Returns:
        Configured DataLoader for validation or None if validation data not found
    """
    try:
        # Initialize validation dataset with no augmentation
        dataset = CustomDataset(
            data_dir=data_dir,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            cache_dir=cache_dir,
            no_caching_latents=True,  # Don't cache validation latents
            all_ar=False,  # No aspect ratio bucketing for validation
            min_size=min_size,
            max_size=max_size,
            token_dropout_rate=0.0,  # No dropout for validation
            caption_dropout_rate=0.0,
            use_tag_weighting=False,  # No tag weighting for validation
            **kwargs
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )
        
        logger.info(f"Created validation dataloader with {len(dataset)} samples")
        return dataloader
        
    except FileNotFoundError:
        logger.warning("No validation data found")
        return None
        
    except Exception as e:
        logger.error(f"Failed to create validation dataloader: {str(e)}")
        raise

def get_dataset_statistics(
    data_dir: Union[str, Path],
    vae: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get dataset statistics for monitoring.
    
    Args:
        data_dir: Dataset directory
        vae: Optional VAE model for latent statistics
        **kwargs: Additional dataset configuration
        
    Returns:
        Dictionary containing dataset statistics
    """
    try:
        dataset = CustomDataset(data_dir=data_dir, vae=vae, **kwargs)
        
        stats = {
            "num_samples": len(dataset),
            "bucket_statistics": dataset.get_bucket_statistics(),
            "cache_statistics": dataset.get_cache_statistics() if hasattr(dataset, 'get_cache_statistics') else {},
            "tag_statistics": dataset.get_tag_statistics() if hasattr(dataset, 'get_tag_statistics') else {}
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {str(e)}")
        return {}