"""Ultra-optimized multi-aspect ratio dataset implementation."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from multiprocessing import Manager
from functools import lru_cache
from src.data.cacheing.vae import VAECache
from src.data.cacheing.text_embeds import TextEmbeddingCache
from src.data.multiaspect.bucket_manager import Bucket, BucketManager, _process_chunk
from src.data.prompt.caption_processor import CaptionProcessor
from PIL import Image
from multiprocessing import Pool

logger = logging.getLogger(__name__)



class MultiAspectDataset(Dataset):
    """Ultra-optimized dataset for multi-aspect ratio training."""
    
    __slots__ = ('image_paths', 'captions', 'bucket_manager', 'vae_cache',
                 'text_cache', '_stats', '_image_cache',
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
        caption_dropout: float = 0.1,
        rarity_factor: float = 0.9,
        emphasis_factor: float = 1.2,
    ):
        """Initialize with optimized caching and parallel processing."""
        self.image_paths = image_paths
        self.captions = captions
        self.bucket_manager = bucket_manager
        self.vae_cache = vae_cache
        self.text_cache = text_cache
        self.num_workers = num_workers
        
        # Initialize caption processor
        self.caption_processor = CaptionProcessor(
            token_dropout_rate=token_dropout,
            caption_dropout_rate=caption_dropout,
            rarity_factor=rarity_factor,
            emphasis_factor=emphasis_factor,
            num_workers=num_workers
        )
        
        # Use multiprocessing Manager for shared state
        manager = Manager()
        self._stats = manager.dict()
        self._image_cache = manager.dict()
        self._transform_cache = manager.dict()
        self._batch_cache = manager.dict()
        
        # Process images
        self._preprocess_images()
    
    def _preprocess_images(self) -> None:
        """Pre-process images using multiprocessing Pool."""
        # Initialize stats
        self._stats.update({
            'processed': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        chunk_size = max(1, len(self.image_paths) // (self.num_workers * 4))
        chunks = [self.image_paths[i:i + chunk_size] 
                 for i in range(0, len(self.image_paths), chunk_size)]
        
        # Process chunks in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(
                _process_chunk,
                [(chunk, self.bucket_manager) for chunk in chunks]
            )
            
        # Aggregate results
        total_processed = sum(r['processed'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        
        # Update stats atomically
        self._stats['processed'] = total_processed
        self._stats['errors'] = total_errors
    
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
    
    def _prepare_text(self, path: str, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare text embeddings with caption processing and caching."""
        if self.text_cache is None:
            raise RuntimeError("Text cache not initialized")
            
        caption = self.captions.get(path, "")
        # Process caption to get tags and weights
        tags, weights = self.caption_processor.process_caption(caption, training=training)
        
        if not tags:
            # Return empty/default embeddings if no valid tags
            return self.text_cache.encode("")
            
        # Join tags with weights into a weighted prompt
        weighted_caption = ", ".join(
            f"{{{tag}}}" if weight > 1.5 else 
            f"{{{{{tag}}}}}" if weight > 2.0 else 
            tag 
            for tag, weight in zip(tags, weights)
        )
        
        return self.text_cache.encode(weighted_caption)
    
    def _prepare_batch(
        self,
        paths: List[str],
        bucket: Bucket
    ) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data."""
        # Process images
        pixel_values = []
        for path in paths:
            image = self._load_image(path)
            tensor = self._transform_image(image, bucket)
            pixel_values.append(tensor)
            
        # Process text embeddings
        text_embeddings = [self._prepare_text(path) for path in paths]
        
        # Stack results
        pixel_values = torch.stack(pixel_values)
        encoder_hidden_states = torch.stack([e[0] for e in text_embeddings])
        pooled_outputs = torch.stack([e[1] for e in text_embeddings])
        
        return {
            'pixel_values': pixel_values,
            'encoder_hidden_states': encoder_hidden_states,
            'pooled_outputs': pooled_outputs
        }
    
    def get_batch(self, bucket: Bucket) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch for a specific bucket with caching."""
        images = self.bucket_manager.group_by_bucket([self.image_paths])[bucket]
        
        if not images or len(images) < bucket.batch_size:
            return None
        
        batch_indices = torch.randperm(len(images))[:bucket.batch_size]
        batch_paths = [images[i] for i in batch_indices]
        
        # Use string key for cache
        cache_key = "|".join(batch_paths)
        cached = self._batch_cache.get(cache_key)
        if cached is not None:
            self._stats['cache_hits'] = self._stats.get('cache_hits', 0) + 1
            return cached
        
        batch = self._prepare_batch(batch_paths, bucket)
        self._batch_cache[cache_key] = batch
        self._stats['cache_misses'] = self._stats.get('cache_misses', 0) + 1
        
        return batch
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int, training: bool = True) -> Dict[str, torch.Tensor]:
        """Get a single item with caching."""
        path = self.image_paths[idx]
        bucket = self.bucket_manager.get_bucket(path)
        
        if bucket is None:
            raise ValueError(f"No bucket found for {path}")
        
        # Load and transform image
        image = self._load_image(path)
        pixel_values = self._transform_image(image, bucket)
        
        # Prepare text embeddings
        text_embeds = self._prepare_text(path, training=training)
        
        return {
            'pixel_values': pixel_values,
            'encoder_hidden_states': text_embeds[0],
            'pooled_outputs': text_embeds[1]
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        return dict(self._stats)
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self._image_cache.clear()
        self._transform_cache.clear()
        self._batch_cache.clear()
        self._stats.clear()
    
    def _initialize_resources(self) -> None:
        """Initialize/reinitialize resources after unpickling."""
        # Create new Manager instance
        manager = Manager()
        
        # Reinitialize shared dictionaries
        self._stats = manager.dict()
        self._image_cache = manager.dict()
        self._transform_cache = manager.dict()
        self._batch_cache = manager.dict()
        
        # Initialize stats
        self._stats.update({
            'processed': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        # Reinitialize caption processor if needed
        if not hasattr(self, 'caption_processor'):
            self.caption_processor = CaptionProcessor(
                token_dropout_rate=0.1,  # Default values
                caption_dropout_rate=0.1
            )
    
    def __getstate__(self):
        """Control pickling behavior."""
        state = {
            'image_paths': self.image_paths,
            'captions': self.captions,
            'bucket_manager': self.bucket_manager,
            'vae_cache': self.vae_cache,
            'text_cache': self.text_cache,
            'num_workers': self.num_workers,
            'caption_processor': self.caption_processor
        }
        return state
    
    def __setstate__(self, state):
        """Control unpickling behavior."""
        # Restore basic attributes
        self.image_paths = state['image_paths']
        self.captions = state['captions']
        self.bucket_manager = state['bucket_manager']
        self.vae_cache = state['vae_cache']
        self.text_cache = state['text_cache']
        self.num_workers = state['num_workers']
        self.caption_processor = state['caption_processor']
        
        # Initialize shared resources
        self._initialize_resources()


def create_train_dataloader(
    image_paths: List[str],
    captions: Dict[str, str],
    bucket_manager: BucketManager,
    batch_size: int,
    num_workers: int = 4,
    vae_cache: Optional[VAECache] = None,
    text_cache: Optional[TextEmbeddingCache] = None,
    token_dropout: float = 0.1,
    caption_dropout: float = 0.1,
    rarity_factor: float = 0.9,
    emphasis_factor: float = 1.2,
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
        token_dropout: Rate at which individual tokens are dropped during training
        caption_dropout: Rate at which entire captions are dropped during training
        rarity_factor: Factor for weighting rare tags
        emphasis_factor: Factor for emphasis tag weighting
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
        num_workers=num_workers,
        token_dropout=token_dropout,
        caption_dropout=caption_dropout,
        rarity_factor=rarity_factor,
        emphasis_factor=emphasis_factor
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
        num_workers=num_workers,
        token_dropout=0.0,  # No dropout for validation
        caption_dropout=0.0  # No dropout for validation
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples for validation
    )
