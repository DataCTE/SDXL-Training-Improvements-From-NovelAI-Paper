"""Multi-aspect dataset implementation for SDXL training.

This module provides a PyTorch Dataset implementation that supports training
with images of different aspect ratios using bucketing and efficient caching.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm.auto import tqdm

from src.data.image_processing.loading import load_and_verify_image
from src.data.image_processing.transforms import (
    resize_image, random_flip, random_jitter, random_rotate, gaussian_blur
)
from src.data.image_processing.manipluations import converter
from src.data.image_processing.validation import (
    validate_image_comprehensive, validate_tensor_comprehensive,
    validate_dimensions, check_image_corruption
)
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.multiaspect.bucket_manager import BucketManager
from src.data.multiaspect.image_grouper import ImageGrouper
from src.data.prompt.caption_processor import CaptionProcessor

logger = logging.getLogger(__name__)


class MultiAspectDataset(Dataset):
    """Dataset that handles images with different aspect ratios.
    
    This dataset implementation integrates:
    - Dynamic bucketing based on aspect ratios
    - Efficient caching of VAE latents and text embeddings
    - Multi-threaded image loading and processing
    - Comprehensive image validation and error handling
    - Configurable data augmentation pipeline
    """
    
    def __init__(
        self,
        image_paths: List[str],
        prompts: List[str],
        bucket_manager: BucketManager,
        vae_cache: VAECache,
        text_embedding_cache: TextEmbeddingCache,
        num_workers: int = 4,
        enable_transforms: bool = True,
        transform_params: Optional[Dict[str, Any]] = None,
        min_size: int = 512,
        max_size: int = 4096
    ):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to training images
            prompts: List of corresponding prompt texts
            bucket_manager: Manager for aspect ratio buckets
            vae_cache: Cache for VAE latents
            text_embedding_cache: Cache for text embeddings
            num_workers: Number of worker threads
            enable_transforms: Whether to use data augmentation
            transform_params: Parameters for transforms
            min_size: Minimum image dimension
            max_size: Maximum image dimension
        """
        self.image_paths = image_paths
        self.prompts = prompts
        self.bucket_manager = bucket_manager
        self.vae_cache = vae_cache
        self.text_embedding_cache = text_embedding_cache
        self.num_workers = num_workers
        self.enable_transforms = enable_transforms
        self.transform_params = transform_params or {}
        self.min_size = min_size
        self.max_size = max_size

        # Initialize components
        self.image_grouper = ImageGrouper(bucket_manager)
        self.caption_processor = CaptionProcessor()
        
        # Process and validate dataset
        self._process_dataset()

    def _process_dataset(self):
        """Process and validate the entire dataset.
        
        This method:
        1. Validates all images
        2. Groups images into buckets
        3. Preprocesses text prompts
        4. Initializes caches
        """
        logger.info("Processing dataset with %d images", len(self.image_paths))
        
        # Validate images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for path in self.image_paths:
                futures.append(executor.submit(self._validate_image, path))
            
            # Collect results and filter invalid images
            valid_indices = []
            for idx, future in enumerate(futures):
                try:
                    future.result()
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning("Skipping invalid image %s: %s", self.image_paths[idx], str(e))
            
            # Filter dataset
            self.image_paths = [self.image_paths[i] for i in valid_indices]
            self.prompts = [self.prompts[i] for i in valid_indices]

        # Group images into buckets
        self.image_buckets = self.image_grouper.group_images(self.image_paths)
        
        # Process prompts
        self.processed_prompts = [
            self.caption_processor.process_caption(prompt, training=True) for prompt in self.prompts
        ]
        
        # Initialize caches
        self._init_caches()
        
        logger.info("Dataset processing complete. %d valid images in %d buckets",
                   len(self.image_paths), len(self.image_buckets))

    def _validate_image(self, image_path: str) -> None:
        """Validate a single image file.
        
        Args:
            image_path: Path to image file
            
        Raises:
            Various validation errors if image is invalid
        """
        # Check for corruption
        if error := check_image_corruption(image_path):
            raise ValueError(f"Corrupted image: {error}")
            
        # Load and validate image
        image = load_and_verify_image(image_path)
        validate_image_comprehensive(
            image,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        # Validate dimensions
        width, height = image.size
        validate_dimensions(
            width, height,
            min_size=self.min_size,
            max_size=self.max_size
        )

    def _init_caches(self):
        """Initialize VAE and text embedding caches."""
        logger.info("Initializing caches...")
        
        # Process images in batches
        batch_size = 32
        total_batches = (len(self.image_paths) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Caching VAE latents") as pbar_vae:
            for i in range(0, len(self.image_paths), batch_size):
                batch_paths = self.image_paths[i:i + batch_size]
                self.vae_cache.cache_batch(batch_paths)
                pbar_vae.update(1)
        
        with tqdm(total=total_batches, desc="Caching text embeddings") as pbar_text:
            for i in range(0, len(self.processed_prompts), batch_size):
                batch_prompts = self.processed_prompts[i:i + batch_size]
                self.text_embedding_cache.cache_batch(batch_prompts)
                pbar_text.update(1)

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.
        
        Args:
            idx: Index of example
            
        Returns:
            Dict containing:
            - image_latents: VAE latents tensor
            - prompt_embeds: Text embedding tensor
            - bucket_size: (height, width) of bucket
        """
        image_path = self.image_paths[idx]
        prompt = self.processed_prompts[idx]
        
        # Get bucket size
        bucket_size = self.image_buckets[image_path]
        
        # Get cached tensors
        image_latents = self.vae_cache.get(image_path)
        prompt_embeds = self.text_embedding_cache.get(prompt)
        
        # Validate tensors
        validate_tensor_comprehensive(image_latents)
        validate_tensor_comprehensive(prompt_embeds)
        
        # Apply transforms if enabled
        if self.enable_transforms:
            image_latents = self._apply_transforms(image_latents)
        
        return {
            'image_latents': image_latents,
            'prompt_embeds': prompt_embeds,
            'bucket_size': bucket_size
        }

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation transforms.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Transformed tensor
        """
        if random.random() < self.transform_params.get('flip_prob', 0.5):
            tensor = random_flip(tensor)
            
        if random.random() < self.transform_params.get('rotate_prob', 0.3):
            angle_range = self.transform_params.get('angle_range', (-10, 10))
            tensor = random_rotate(tensor, angle_range)
            
        if random.random() < self.transform_params.get('jitter_prob', 0.5):
            tensor = random_jitter(
                tensor,
                brightness=self.transform_params.get('brightness', 0.2),
                contrast=self.transform_params.get('contrast', 0.2),
                saturation=self.transform_params.get('saturation', 0.2),
                hue=self.transform_params.get('hue', 0.1)
            )
            
        if random.random() < self.transform_params.get('blur_prob', 0.1):
            tensor = gaussian_blur(
                tensor,
                kernel_size=self.transform_params.get('blur_kernel', 3),
                sigma=self.transform_params.get('blur_sigma', 1.0)
            )
            
        return tensor


def create_train_dataloader(
    train_data_dir: Union[str, Path],
    vae_cache: VAECache,
    text_embedding_cache: TextEmbeddingCache,
    batch_size: int,
    num_workers: int = 4,
    enable_transforms: bool = True,
    transform_params: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Create training dataloader.
    
    Args:
        train_data_dir: Directory with training data
        vae_cache: VAE latents cache
        text_embedding_cache: Text embeddings cache
        batch_size: Batch size
        num_workers: Number of workers
        enable_transforms: Whether to use augmentation
        transform_params: Transform parameters
        
    Returns:
        DataLoader for training
    """
    # Create bucket manager
    bucket_manager = BucketManager()
    
    # Get image paths and prompts
    train_data_dir = Path(train_data_dir)
    image_paths = []
    prompts = []
    
    # Collect only valid image-caption pairs
    for img_path in train_data_dir.glob('*.jpg'):
        txt_path = img_path.with_suffix('.txt')
        try:
            if txt_path.exists():
                prompt = txt_path.read_text().strip()
                image_paths.append(img_path)
                prompts.append(prompt)
            else:
                logger.warning(f"Skipping {img_path}: No matching caption file found")
        except Exception as e:
            logger.warning(f"Error reading caption for {img_path}: {str(e)}")
            continue
            
    # Also check png files
    for img_path in train_data_dir.glob('*.png'):
        txt_path = img_path.with_suffix('.txt')
        try:
            if txt_path.exists():
                prompt = txt_path.read_text().strip()
                image_paths.append(img_path)
                prompts.append(prompt)
            else:
                logger.warning(f"Skipping {img_path}: No matching caption file found")
        except Exception as e:
            logger.warning(f"Error reading caption for {img_path}: {str(e)}")
            continue
    
    if not image_paths:
        raise ValueError(f"No valid image-caption pairs found in {train_data_dir}")
        
    logger.info(f"Found {len(image_paths)} valid image-caption pairs")
    
    # Create dataset
    dataset = MultiAspectDataset(
        image_paths=[str(p) for p in image_paths],
        prompts=prompts,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_embedding_cache=text_embedding_cache,
        num_workers=num_workers,
        enable_transforms=enable_transforms,
        transform_params=transform_params
    )
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def create_validation_dataloader(
    val_data_dir: Union[str, Path],
    vae_cache: VAECache,
    text_embedding_cache: TextEmbeddingCache,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create validation dataloader.
    
    Args:
        val_data_dir: Directory with validation data
        vae_cache: VAE latents cache
        text_embedding_cache: Text embeddings cache
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        DataLoader for validation
    """
    # Create bucket manager
    bucket_manager = BucketManager()
    
    # Get image paths and prompts
    val_data_dir = Path(val_data_dir)
    image_paths = []
    prompts = []
    
    # Collect only valid image-caption pairs
    for img_path in val_data_dir.glob('*.jpg'):
        txt_path = img_path.with_suffix('.txt')
        try:
            if txt_path.exists():
                prompt = txt_path.read_text().strip()
                image_paths.append(img_path)
                prompts.append(prompt)
            else:
                logger.warning(f"Skipping {img_path}: No matching caption file found")
        except Exception as e:
            logger.warning(f"Error reading caption for {img_path}: {str(e)}")
            continue
            
    # Also check png files
    for img_path in val_data_dir.glob('*.png'):
        txt_path = img_path.with_suffix('.txt')
        try:
            if txt_path.exists():
                prompt = txt_path.read_text().strip()
                image_paths.append(img_path)
                prompts.append(prompt)
            else:
                logger.warning(f"Skipping {img_path}: No matching caption file found")
        except Exception as e:
            logger.warning(f"Error reading caption for {img_path}: {str(e)}")
            continue
    
    if not image_paths:
        raise ValueError(f"No valid image-caption pairs found in {val_data_dir}")
        
    logger.info(f"Found {len(image_paths)} valid image-caption pairs")
    
    # Create dataset
    dataset = MultiAspectDataset(
        image_paths=[str(p) for p in image_paths],
        prompts=prompts,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_embedding_cache=text_embedding_cache,
        num_workers=num_workers,
        enable_transforms=False  # No augmentation for validation
    )
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
