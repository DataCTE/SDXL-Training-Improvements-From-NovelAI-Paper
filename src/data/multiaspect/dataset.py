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
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from src.config.args import TrainingConfig

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
        num_workers: int = 0,
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
        
        # Use regular dictionaries instead of Manager
        self._stats = {}
        self._image_cache = {}
        self._transform_cache = {}
        self._batch_cache = {}
        
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
        
        # Handle case when num_workers is 0
        if self.num_workers == 0:
            # Process images sequentially
            results = [_process_chunk(([path], self.bucket_manager)) 
                      for path in self.image_paths]
        else:
            # Process in parallel with chunks
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
    
    def _transform_image(self, image: Image.Image, bucket: Bucket) -> torch.Tensor:
        """Transform image with caching and mixed precision."""
        # Always resize to bucket dimensions, regardless of current size
        image = image.resize((bucket.width, bucket.height), Image.Resampling.LANCZOS)
        
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        path = self.image_paths[idx]
        
        # Load and process image
        pixel_values = self._prepare_image(path)
        
        # Get text embeddings
        text_embeds = self._prepare_text(path, training=True)
        
        # Ensure all tensors are on CPU for pinning
        batch = {
            "pixel_values": pixel_values.cpu(),
            "text_embeds": tuple(t.cpu() for t in text_embeds),
            "path": path
        }
        
        return batch
    
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
    dataset: MultiAspectDataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create training dataloader with proper memory handling."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu",
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn
    )

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles text embedding tuples."""
    elem = batch[0]
    collated = {}
    
    for key in elem:
        if key == "text_embeds":
            # Handle text embedding tuples
            text_embeds_list = [b[key] for b in batch]
            collated[key] = tuple(
                torch.stack([emb[i] for emb in text_embeds_list])
                for i in range(len(text_embeds_list[0]))
            )
        elif key == "path":
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = torch.stack([b[key] for b in batch])
            
    return collated

def create_validation_dataloader(
    image_paths: List[str],
    captions: Dict[str, str],
    config: TrainingConfig,
    bucket_manager: Optional[BucketManager] = None,
    vae_cache: Optional[VAECache] = None,
    text_cache: Optional[TextEmbeddingCache] = None,
) -> DataLoader:
    """Create an optimized DataLoader for validation.
    
    Args:
        image_paths: List of paths to validation images
        captions: Dictionary mapping image paths to their captions
        config: Training configuration
        bucket_manager: Optional BucketManager instance
        vae_cache: Optional VAE cache for faster encoding
        text_cache: Optional text embedding cache
        
    Returns:
        DataLoader configured for validation
    """
    # Create bucket manager if not provided
    if bucket_manager is None:
        bucket_manager = BucketManager(
            max_resolution=config.max_resolution,
            min_batch_size=1,
            max_batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Add images to bucket manager
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                bucket_manager.add_image(image_path, width, height)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
    
    # Create dataset with no dropout for validation
    dataset = MultiAspectDataset(
        image_paths=image_paths,
        captions=captions,
        bucket_manager=bucket_manager,
        vae_cache=vae_cache,
        text_cache=text_cache,
        num_workers=config.num_workers,
        token_dropout=0.0,
        caption_dropout=0.0,
        rarity_factor=config.tag_weighting.rarity_factor,
        emphasis_factor=config.tag_weighting.emphasis_factor
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
