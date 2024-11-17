"""
Dataset module for SDXL training pipeline.

This module implements a custom dataset for training SDXL models, handling
image-caption pairs with dynamic bucketing, caching, and preprocessing.

Classes:
    CustomDataset: Main dataset class for SDXL training
"""

import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import random
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.data.image_processor import ImageProcessor
from src.data.dataset.bucket_manager import BucketManager
from src.data.latent_cache import LatentCacheManager
from src.data.caption_processor import CaptionProcessor
from src.data.dataset.image_grouper import ImageGrouper
from src.data.dataset.dataset_initializer import DatasetInitializer

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """Dataset class for SDXL training.
    
    This class manages the training dataset, including image loading,
    preprocessing, bucketing, and caching. It supports dynamic aspect
    ratio bucketing and efficient latent caching.
    
    Attributes:
        data_dir: Directory containing image-caption pairs
        vae: Optional VAE model for latent caching
        tokenizer: Optional tokenizer for text processing
        tokenizer_2: Optional secondary tokenizer for text processing
        text_encoder: Optional text encoder for text processing
        text_encoder_2: Optional secondary text encoder for text processing
        cache_dir: Directory for caching latents
        no_caching_latents: Whether to disable latent caching
        all_ar: Whether to use aspect ratio for bucketing
        min_size: Minimum image size
        max_size: Maximum image size
        bucket_step_size: Resolution step size for buckets
        max_bucket_area: Maximum area for buckets
        token_dropout_rate: Token dropout probability
        caption_dropout_rate: Caption dropout probability
        min_tag_weight: Minimum weight for tags
        max_tag_weight: Maximum weight for tags
        use_tag_weighting: Whether to enable tag-based loss weighting
        device: Torch device for processing
    """
    
    def __init__(
        self,
        data_dir: str,
        vae: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        cache_dir: str = "latents_cache",
        no_caching_latents: bool = False,
        all_ar: bool = False,
        min_size: int = 512,
        max_size: int = 4096,
        bucket_step_size: int = 64,
        max_bucket_area: int = 1024*1024,
        token_dropout_rate: float = 0.1,
        caption_dropout_rate: float = 0.1,
        min_tag_weight: float = 0.1,
        max_tag_weight: float = 3.0,
        use_tag_weighting: bool = True,
        device: str = "cuda"
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing image-caption pairs
            vae: Optional VAE model for latent caching
            tokenizer: Optional tokenizer for text processing
            tokenizer_2: Optional secondary tokenizer for text processing
            text_encoder: Optional text encoder for text processing
            text_encoder_2: Optional secondary text encoder for text processing
            cache_dir: Directory for caching latents
            no_caching_latents: Whether to disable latent caching
            all_ar: Whether to use aspect ratio for bucketing
            min_size: Minimum image size
            max_size: Maximum image size
            bucket_step_size: Resolution step size for buckets
            max_bucket_area: Maximum area for buckets
            token_dropout_rate: Token dropout probability
            caption_dropout_rate: Caption dropout probability
            min_tag_weight: Minimum weight for tags
            max_tag_weight: Maximum weight for tags
            use_tag_weighting: Whether to enable tag-based loss weighting
            device: Device to use for processing
        """
        super().__init__()
        self.data_dir = data_dir
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.cache_dir = cache_dir
        self.no_caching_latents = no_caching_latents
        self.all_ar = all_ar
        self.min_size = min_size
        self.max_size = max_size
        self.bucket_step_size = bucket_step_size
        self.max_bucket_area = max_bucket_area
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        self.min_tag_weight = min_tag_weight
        self.max_tag_weight = max_tag_weight
        self.use_tag_weighting = use_tag_weighting
        self.device = device
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.caption_processor = CaptionProcessor(
            token_dropout_rate=token_dropout_rate,
            caption_dropout_rate=caption_dropout_rate
        )
        self.bucket_manager = BucketManager(
            min_resolution=min_size,
            max_resolution=max_size,
            resolution_step=bucket_step_size,
            tolerance=0.033
        )
        
        # Initialize dataset components
        self.dataset_initializer = DatasetInitializer()
        self.image_grouper = ImageGrouper(
            bucket_manager=self.bucket_manager
        )
        
        # Initialize latent cache if VAE provided
        self.latents_cache = (
            LatentCacheManager(vae=vae, cache_dir=cache_dir)
            if vae is not None else None
        )
        
        # Process images and create buckets
        self._initialize_dataset()
        
    def _initialize_dataset(self) -> None:
        """Initialize dataset structure and caches."""
        try:
            # Find valid image-caption pairs
            self.image_paths = self.dataset_initializer.find_valid_pairs(self.data_dir)
            if not self.image_paths:
                raise ValueError(f"No valid image-caption pairs found in {self.data_dir}")
            
            # Load captions for all images
            self.captions = self.dataset_initializer.load_captions(self.image_paths)
            
            # Group images into buckets
            self.bucket_data = self.image_grouper.group_images(
                image_paths=self.image_paths,
                use_ar=self.all_ar
            )
            
            # Initialize latent caching if enabled
            if not self.no_caching_latents:
                self._initialize_latent_cache()
            
            # Precompute tag weights if enabled
            if self.use_tag_weighting:
                self._precompute_tag_weights()
            
            logger.info(f"Dataset initialized with {len(self.image_paths)} images in {len(self.bucket_data)} buckets")
            
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {str(e)}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all dataset components with proper error handling."""
        try:
            # Initialize processors
            self.image_processor = ImageProcessor()
            self.caption_processor = CaptionProcessor(
                token_dropout_rate=self.token_dropout_rate,
                caption_dropout_rate=self.caption_dropout_rate
            )
            
            # Initialize cache managers
            self.latents_cache = LatentCacheManager(
                cache_dir=self.cache_dir,
                vae=self.vae
            )
            
            # Initialize bucket management
            self.bucket_manager = BucketManager(
                min_resolution=self.min_size,
                max_resolution=self.max_size,
                resolution_step=self.bucket_step_size,
                tolerance=0.033
            )
            
            # Initialize image grouper
            self.image_grouper = ImageGrouper(self.bucket_manager)
            
            # Initialize dataset initializer
            self.dataset_initializer = DatasetInitializer()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _create_buckets(self) -> None:
        """Create aspect ratio buckets for the dataset images."""
        try:
            for image_path in self.image_paths:
                image = self.image_processor.load_and_verify_image(image_path)
                if image is None:
                    continue
                    
                # Get image dimensions and create bucket
                width, height = image.size
                self.bucket_manager.add_image(
                    image_path,
                    width,
                    height,
                    no_upscale=True
                )
                
            self.bucket_manager.finalize_buckets()
            logger.info(
                "Created %d buckets for %d images",
                len(self.bucket_manager.buckets),
                len(self.image_paths)
            )
            
        except Exception as error:
            logger.error("Failed to create buckets: %s", str(error))
            raise
            
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset item.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dict containing the processed image/latent and caption
            
        Raises:
            RuntimeError: If image processing fails
        """
        try:
            image_path = self.image_paths[idx]
            caption = self.captions.get(image_path, "")
            
            # Get target size from bucket if enabled
            target_size = None
            if self.all_ar:
                target_size = self.bucket_manager.get_bucket_size(image_path)
            
            # Try to get cached latent first
            if self.latents_cache is not None:
                latent = self.latents_cache.get_latents(image_path)
                if latent is not None:
                    return {
                        "latent": latent,
                        "caption": caption,
                        "image_path": image_path
                    }
            
            # Process image if no cached latent
            image = self.image_processor.process_image(
                image_path,
                target_size=target_size
            )
            if image is None:
                raise RuntimeError(f"Failed to process image: {image_path}")
                
            return {
                "image": image,
                "caption": caption,
                "image_path": image_path
            }
            
        except Exception as error:
            logger.error(
                "Error processing item %d (%s): %s",
                idx, image_path, str(error)
            )
            raise

    def _initialize_latent_cache(self) -> None:
        """Initialize and process latent caching."""
        if self.vae is None:
            logger.warning("VAE not provided, skipping latent caching")
            return
            
        try:
            # Process images by bucket to ensure consistent tensor sizes
            batch_size = 4  # Adjust based on GPU memory
            
            # Initialize latent caching with all image paths
            all_image_paths = []
            for image_paths in self.bucket_data.values():
                all_image_paths.extend(image_paths)
                
            # Check if we need to process any new images
            if not self.latents_cache.initialize_latent_caching(all_image_paths):
                return
                
            # Initialize GPU workers
            self.latents_cache.initialize_workers()
            
            try:
                # Iterate through each bucket group
                for bucket_dims, image_paths in self.bucket_data.items():
                    # Filter for uncached images
                    uncached_paths = [path for path in image_paths if not self.latents_cache.is_cached(path)]
                    if not uncached_paths:
                        continue
                        
                    for i in range(0, len(uncached_paths), batch_size):
                        batch_paths = uncached_paths[i:i + batch_size]
                        
                        # Load and transform images using bucket dimensions
                        batch_tensors = []
                        valid_paths = []
                        for path in batch_paths:
                            image = self.image_processor.load_and_transform(
                                path,
                                target_height=bucket_dims[0],
                                target_width=bucket_dims[1]
                            )
                            if image is not None:
                                batch_tensors.append(image)
                                valid_paths.append(path)
                        
                        if not batch_tensors:
                            continue
                            
                        # Stack tensors and queue for processing
                        try:
                            batch_tensor = torch.stack(batch_tensors)
                            self.latents_cache.process_latents_batch(batch_tensor, valid_paths)
                        except Exception as e:
                            logger.error(f"Failed to process latent batch: {str(e)}")
                            continue
                        
                        # Process any completed results
                        self.latents_cache.process_results()
                
                # Process any remaining results
                while self.latents_cache.processed_images < self.latents_cache.total_images:
                    self.latents_cache.process_results()
                    
            finally:
                # Always ensure workers are closed
                self.latents_cache.close_workers()
                
            logger.info("Latent cache initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize latent cache: {str(e)}")
            raise

    def _precompute_tag_weights(self) -> None:
        """Pre-compute tag weights for all captions using thread pool."""
        try:
            # Process captions in parallel
            with ThreadPoolExecutor(max_workers=min(32, len(self.image_paths))) as executor:
                futures = []
                for path in self.image_paths:
                    futures.append(executor.submit(self.caption_processor.load_caption, path))
                
                # Process results
                for path, future in zip(self.image_paths, futures):
                    try:
                        caption = future.result()
                        if caption:
                            tags, weights = self.caption_processor.process_tags_with_weights(caption)
                            self.caption_processor.weight_cache[path] = (tags, weights)
                    except Exception as e:
                        logger.debug(f"Failed to process caption for {path}: {str(e)}")
                        continue
            
            logger.info(f"Precomputed weights for {len(self.caption_processor.weight_cache)} captions")
            
        except Exception as e:
            logger.error(f"Failed to precompute tag weights: {str(e)}")

    def shuffle_dataset(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset while maintaining bucket grouping.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Shuffle within each bucket
        for bucket_dims in self.bucket_data:
            paths = self.bucket_data[bucket_dims]
            indices = torch.randperm(len(paths)).tolist()
            self.bucket_data[bucket_dims] = [paths[i] for i in indices]
        
        logger.info("Dataset shuffled within buckets")

    def get_bucket_stats(self) -> Dict[Tuple[int, int], int]:
        """Get statistics about bucket usage.
        
        Returns:
            Dictionary mapping bucket dimensions to number of images
        """
        return {dims: len(paths) for dims, paths in self.bucket_data.items()}