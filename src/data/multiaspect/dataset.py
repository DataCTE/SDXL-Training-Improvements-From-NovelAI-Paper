"""Ultra-optimized multi-aspect ratio dataset implementation."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from torch.cuda import amp
import os
from functools import lru_cache

from src.data.image_processing.validation import validate_image
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.multiaspect.bucket_manager import Bucket, BucketManager
from src.data.prompt.caption_processor import CaptionProcessor, load_captions

logger = logging.getLogger(__name__)

class MultiAspectDataset(Dataset):
    """Ultra-optimized dataset for multi-aspect ratio training."""
    
    __slots__ = ('image_paths', 'captions', 'bucket_manager', 'vae_cache',
                 'text_cache', '_lock', '_executor', '_stats', '_image_cache',
                 '_transform_cache', '_batch_cache', 'num_workers', 'caption_processor')
    
    def __init__(
        self,
        image_paths: List[str],
        captions: Dict[str, str],
        bucket_manager: BucketManager,
        vae_cache: Optional[VAECache] = None,
        text_cache: Optional[TextEmbeddingCache] = None,
        num_workers: int = 4,
        token_dropout: float = 0.1,
        caption_dropout: float = 0.1
    ):
        """Initialize with optimized caching and parallel processing."""
        self.image_paths = image_paths
        self.captions = captions
        self.bucket_manager = bucket_manager
        self.vae_cache = vae_cache
        self.text_cache = text_cache
        
        # Initialize caption processor
        self.caption_processor = CaptionProcessor(
            token_dropout_rate=token_dropout,
            caption_dropout_rate=caption_dropout
        )
        
        # Rest of initialization remains the same
        self.num_workers = num_workers
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._stats = defaultdict(int)
        
        self._image_cache = {}
        self._transform_cache = {}
        self._batch_cache = {}
        
        self._parallel_preprocess_images()
    
    def _parallel_preprocess_images(self) -> None:
        """Pre-process images in parallel for faster access."""
        chunk_size = max(1, len(self.image_paths) // (self.num_workers * 4))
        chunks = [self.image_paths[i:i + chunk_size] 
                 for i in range(0, len(self.image_paths), chunk_size)]
        
        futures = [
            self._executor.submit(self._preprocess_chunk, chunk)
            for chunk in chunks
        ]
        
        for future in futures:
            future.result()
    
    def _preprocess_chunk(self, paths: List[str]) -> None:
        """Process a chunk of images."""
        for path in paths:
            try:
                # Validate image
                if not validate_image(path):
                    logger.warning(f"Invalid image: {path}")
                    continue
                
                # Get image dimensions
                with Image.open(path) as img:
                    width, height = img.size
                
                # Add to bucket manager
                self.bucket_manager.add_image(path, width, height)
                self._stats['processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                self._stats['errors'] += 1
    
    @lru_cache(maxsize=1024)
    def _load_image(self, path: str) -> Image.Image:
        """Load and cache image with memory efficiency."""
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise
    
    def _transform_image(
        self,
        image: Image.Image,
        bucket: Bucket
    ) -> torch.Tensor:
        """Transform image with caching and mixed precision."""
        # Resize to bucket dimensions
        if image.size != (bucket.width, bucket.height):
            image = image.resize((bucket.width, bucket.height),
                               Image.Resampling.LANCZOS)
        
        # Convert to tensor with mixed precision
        with torch.amp.autocast('cuda'):
            tensor = torch.from_numpy(np.array(image))
            tensor = tensor.permute(2, 0, 1).float()
            tensor = tensor / 127.5 - 1.0
        
        return tensor
    
    def _prepare_text(self, path: str) -> Tuple[torch.Tensor, ...]:
        """Prepare text embeddings with caption processing and caching."""
        if self.text_cache is None:
            raise RuntimeError("Text cache not initialized")
            
        caption = self.captions.get(path, "")
        # Process caption to get tags and weights
        tags, weights = self.caption_processor.process_caption(caption)
        
        if not tags:
            # Return empty/default embeddings if no valid tags
            return self.text_cache.encode_text("")
            
        # Join tags with weights into a weighted prompt
        weighted_caption = ", ".join(
            f"{{{tag}}}" if weight > 1.5 else 
            f"{{{{{tag}}}}}" if weight > 2.0 else 
            tag 
            for tag, weight in zip(tags, weights)
        )
        
        return self.text_cache.encode_text(weighted_caption)
    
    def _prepare_batch(
        self,
        paths: List[str],
        bucket: Bucket
    ) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data with parallel processing."""
        # Load and transform images in parallel
        image_futures = [
            self._executor.submit(self._load_image, path)
            for path in paths
        ]
        
        transform_futures = [
            self._executor.submit(self._transform_image, future.result(), bucket)
            for future in image_futures
        ]
        
        # Prepare text embeddings in parallel
        text_futures = [
            self._executor.submit(self._prepare_text, path)
            for path in paths
        ]
        
        # Gather results
        pixel_values = torch.stack([future.result() for future in transform_futures])
        text_embeddings = [future.result() for future in text_futures]
        
        # Stack text embeddings
        encoder_hidden_states = torch.stack([e[0] for e in text_embeddings])
        pooled_outputs = torch.stack([e[1] for e in text_embeddings])
        
        return {
            'pixel_values': pixel_values,
            'encoder_hidden_states': encoder_hidden_states,
            'pooled_outputs': pooled_outputs
        }
    
    def get_batch(self, bucket: Bucket) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch for a specific bucket with caching."""
        # Get images for this bucket
        images = self.bucket_manager.group_by_bucket([self.image_paths])[bucket]
        
        if not images or len(images) < bucket.batch_size:
            return None
        
        # Select random batch
        batch_indices = torch.randperm(len(images))[:bucket.batch_size]
        batch_paths = [images[i] for i in batch_indices]
        
        # Check cache
        cache_key = tuple(batch_paths)
        cached = self._batch_cache.get(cache_key)
        if cached is not None:
            self._stats['cache_hits'] += 1
            return cached
        
        # Prepare batch
        batch = self._prepare_batch(batch_paths, bucket)
        self._batch_cache[cache_key] = batch
        self._stats['cache_misses'] += 1
        
        return batch
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item with caching."""
        path = self.image_paths[idx]
        bucket = self.bucket_manager.get_bucket(path)
        
        if bucket is None:
            raise ValueError(f"No bucket found for {path}")
        
        # Load and transform image
        image = self._load_image(path)
        pixel_values = self._transform_image(image, bucket)
        
        # Prepare text embeddings
        text_embeds = self._prepare_text(path)
        
        return {
            'pixel_values': pixel_values,
            'encoder_hidden_states': text_embeds[0],
            'pooled_outputs': text_embeds[1]
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        with self._lock:
            return dict(self._stats)
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._image_cache.clear()
            self._transform_cache.clear()
            self._batch_cache.clear()
            self._stats.clear()


def create_train_dataloader(
    image_paths: List[str],
    captions: Dict[str, str],
    bucket_manager: BucketManager,
    batch_size: int,
    num_workers: int = 4,
    vae_cache: Optional[VAECache] = None,
    text_cache: Optional[TextEmbeddingCache] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create an optimized DataLoader for training.
    
    Args:
        image_paths: List of paths to training images
        captions: Dictionary mapping image paths to their captions
        bucket_manager: BucketManager instance for aspect ratio bucketing
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        vae_cache: Optional VAE cache for faster encoding
        text_cache: Optional text embedding cache
        shuffle: Whether to shuffle the dataset
        pin_memory: If True, pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader configured for training
    """
    dataset = MultiAspectDataset(
        image_paths=image_paths,
        captions=captions,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_cache=text_cache,
        num_workers=num_workers
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_validation_dataloader(
    image_paths: List[str],
    captions: Dict[str, str],
    bucket_manager: BucketManager,
    batch_size: int,
    num_workers: int = 4,
    vae_cache: Optional[VAECache] = None,
    text_cache: Optional[TextEmbeddingCache] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create an optimized DataLoader for validation.
    
    Args:
        image_paths: List of paths to validation images
        captions: Dictionary mapping image paths to their captions
        bucket_manager: BucketManager instance for aspect ratio bucketing
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        vae_cache: Optional VAE cache for faster encoding
        text_cache: Optional text embedding cache
        pin_memory: If True, pin memory for faster GPU transfer
        
    Returns:
        DataLoader configured for validation
    """
    dataset = MultiAspectDataset(
        image_paths=image_paths,
        captions=captions,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_cache=text_cache,
        num_workers=num_workers
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples for validation
    )
