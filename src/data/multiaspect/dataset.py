"""
Multi-aspect dataset implementation for SDXL training.

This module provides a PyTorch Dataset implementation that supports training
with images of different aspect ratios using bucketing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.data.image_processing.loading import load_and_verify_image
from src.data.image_processing.transforms import (
    resize_image, random_flip, random_jitter
)
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.multiaspect.bucket_manager import BucketManager
from src.data.prompt.caption_processor import CaptionProcessor
logger = logging.getLogger(__name__)


class MultiAspectDataset(Dataset):
    """Dataset that handles images with different aspect ratios.
    
    This dataset implementation supports:
    - Dynamic bucketing based on aspect ratios
    - Efficient caching of VAE latents and text embeddings
    - Multi-threaded image loading and processing
    - Data augmentation with configurable transforms
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
        transform_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to training images
            prompts: List of corresponding prompts
            bucket_manager: BucketManager instance for aspect ratio handling
            vae_cache: VAECache instance for caching latents
            text_embedding_cache: TextEmbeddingCache for prompt embeddings
            num_workers: Number of worker threads for loading
            enable_transforms: Whether to apply data augmentation
            transform_params: Parameters for data augmentation
        """
        super().__init__()
        
        if len(image_paths) != len(prompts):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of prompts ({len(prompts)})"
            )
            
        self.image_paths = image_paths
        self.prompts = prompts
        self.bucket_manager = bucket_manager
        self.vae_cache = vae_cache
        self.text_embedding_cache = text_embedding_cache
        
        self.enable_transforms = enable_transforms
        self.transform_params = transform_params or {}
        
        # Initialize workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prompt_processor = CaptionProcessor()
        
        # Process and bucket all images
        self._process_dataset()
        
    def _process_dataset(self) -> None:
        """Process all images and assign to buckets."""
        logger.info("Processing dataset...")
        
        # Submit all image loading tasks
        futures = []
        for path in self.image_paths:
            futures.append(
                self.executor.submit(load_and_verify_image, path)
            )
            
        # Process results
        for idx, future in enumerate(futures):
            try:
                image = future.result()
                if image is None:
                    logger.warning("Failed to load image: %s", self.image_paths[idx])
                    continue
                    
                # Get target size from bucket manager
                try:
                    target_width, target_height = self.bucket_manager.get_target_size(self.image_paths[idx])
                    
                    # Resize image to target size
                    image = resize_image(image, (target_width, target_height))
                    
                    # Get VAE latents
                    latents = self.vae_cache.get_latents(image)
                    if latents is None:
                        logger.warning("Failed to get VAE latents for: %s", self.image_paths[idx])
                        continue
                        
                    # Get text embeddings
                    text_embeddings = self.text_embedding_cache.get_embeddings(self.prompts[idx])
                    if text_embeddings is None:
                        logger.warning("Failed to get text embeddings for: %s", self.image_paths[idx])
                        continue
                        
                except Exception as e:
                    logger.error("Error processing image %s: %s", self.image_paths[idx], str(e))
                    continue
                    
            except Exception as e:
                logger.error("Failed to process image %s: %s", self.image_paths[idx], str(e))
                continue
        
    def _apply_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation transforms.
        
        Args:
            image: Input tensor image
            
        Returns:
            Transformed image
        """
        if not self.enable_transforms:
            return image
            
        # Apply random flipping
        if self.transform_params.get('flip', True):
            image = random_flip(image, p=0.5)
            
        # Apply color jittering
        if self.transform_params.get('jitter', True):
            image = random_jitter(
                image,
                brightness=self.transform_params.get('brightness', 0.2),
                contrast=self.transform_params.get('contrast', 0.2),
                saturation=self.transform_params.get('saturation', 0.2),
                hue=self.transform_params.get('hue', 0.1),
                p=0.5
            )
            
        return image
        
    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing:
                - 'latents': VAE latents
                - 'text_embeds': Text embeddings
                - 'text_masks': Attention masks
        """
        # Load and process image
        image_path = self.image_paths[idx]
        image = load_and_verify_image(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
            
        # Get target size and resize
        target_width, target_height = self.bucket_manager.get_target_size(image_path)
        image = resize_image(image, (target_width, target_height))
        
        # Apply transforms
        image = self._apply_transforms(image)
        
        # Get VAE latents
        latents = self.vae_cache.get_latents(image)
        
        # Process prompt and get embeddings
        prompt = self.prompts[idx]
        text_embeds, text_masks = self.text_embedding_cache.get_text_embeddings(
            self.prompt_processor.format_caption(prompt)
        )
        
        return {
            'latents': latents,
            'text_embeds': text_embeds,
            'text_masks': text_masks
        }

def create_train_dataloader(
    train_data_dir: Union[str, Path],
    vae_cache: VAECache,
    text_embedding_cache: TextEmbeddingCache,
    batch_size: int,
    num_workers: int = 4,
    enable_transforms: bool = True,
    transform_params: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Create training dataloader with multi-aspect dataset.
    
    Args:
        train_data_dir: Directory containing training images and captions
        vae_cache: Cache for VAE latents
        text_embedding_cache: Cache for text embeddings
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        enable_transforms: Whether to use data augmentation
        transform_params: Parameters for data augmentation
        
    Returns:
        DataLoader for training
    """
    # Get image paths and prompts from data directory
    image_paths = []
    prompts = []
    data_dir = Path(train_data_dir)
    for img_path in data_dir.glob("*.jpg"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_paths.append(str(img_path))
            prompts.append(prompt)
    
    # Create bucket manager for aspect ratio handling
    bucket_manager = BucketManager(
        min_resolution=512,  # SDXL default minimum resolution
        max_resolution=2048,  # SDXL default maximum resolution
        resolution_step=64,   # Standard step size
        tolerance=0.033      # 3.3% aspect ratio tolerance
    )
    
    # Add images to bucket manager
    for img_path in image_paths:
        img = Image.open(img_path)
        width, height = img.size
        bucket_manager.add_image(img_path, width, height)
    
    # Create dataset
    dataset = MultiAspectDataset(
        image_paths=image_paths,
        prompts=prompts,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_embedding_cache=text_embedding_cache,
        num_workers=num_workers,
        enable_transforms=enable_transforms,
        transform_params=transform_params
    )
    
    # Create and return dataloader
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
    """Create validation dataloader with multi-aspect dataset.
    
    Args:
        val_data_dir: Directory containing validation images and captions
        vae_cache: Cache for VAE latents
        text_embedding_cache: Cache for text embeddings
        batch_size: Batch size for validation
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for validation
    """
    # Get image paths and prompts from validation directory
    image_paths = []
    prompts = []
    data_dir = Path(val_data_dir)
    for img_path in data_dir.glob("*.jpg"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_paths.append(str(img_path))
            prompts.append(prompt)
    
    # Create bucket manager for aspect ratio handling
    bucket_manager = BucketManager(
        min_resolution=512,  # SDXL default minimum resolution
        max_resolution=2048,  # SDXL default maximum resolution
        resolution_step=64,   # Standard step size
        tolerance=0.033      # 3.3% aspect ratio tolerance
    )
    
    # Add images to bucket manager
    for img_path in image_paths:
        img = Image.open(img_path)
        width, height = img.size
        bucket_manager.add_image(img_path, width, height)
    
    # Create dataset - no transforms for validation
    dataset = MultiAspectDataset(
        image_paths=image_paths,
        prompts=prompts,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_embedding_cache=text_embedding_cache,
        num_workers=num_workers,
        enable_transforms=False
    )
    
    # Create and return dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
