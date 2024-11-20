"""
Ultra-optimized text embedding cache system.

This module provides an ultra-optimized text embedding cache with parallel processing, 
mixed precision, and efficient memory management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import logging
from torch.cuda import amp
import hashlib
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import os
from .memory import MemoryCache
import random
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool



logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Set start method to 'spawn' for CUDA multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

class TextEmbeddingCache:
    """Ultra-optimized text embedding cache with parallel processing."""
    
    __slots__ = ('tokenizer1', 'tokenizer2', 'text_encoder1', 'text_encoder2',
                 '_memory_cache', '_batch_size', '_max_cache_size', '_stats',
                 '_scaler', '_cache_dir', 'dropout_rate', '_manager',
                 '_num_workers', '_pool')
    
    def __init__(
        self,
        text_encoder1: CLIPTextModel,
        text_encoder2: CLIPTextModel,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        max_cache_size: int = 10000,
        batch_size: int = 16,
        cache_dir: Optional[Union[str, Path]] = None,
        dropout_rate: float = 0.0,
        num_workers: int = 4
    ) -> None:
        """Initialize text embedding cache with dual encoders."""
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.tokenizer1 = tokenizer1 
        self.tokenizer2 = tokenizer2
        
        # Initialize cache and stats
        self._memory_cache = MemoryCache(
            max_memory_gb=32.0,
            max_cache_size=max_cache_size,
            cache_dir=cache_dir
        )
        self._batch_size = batch_size
        self._max_cache_size = max_cache_size
        self._stats = {'hits': 0, 'misses': 0}
        
        # Setup multiprocessing
        self._manager = Manager()
        self._num_workers = num_workers
        self._pool = Pool(num_workers) if num_workers > 0 else None
        
        # Mixed precision
        self._scaler = amp.GradScaler() if torch.cuda.is_available() else None
        
        # Cache directory
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.dropout_rate = dropout_rate
        
    def _get_cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()
        
    def _should_evict(self) -> bool:
        """Check if cache needs eviction."""
        return len(self._memory_cache) >= self._max_cache_size
        
    @torch.no_grad()
    def _encode_text(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text with both encoders."""
        if isinstance(text, str):
            text = [text]
            
        # Tokenize
        tokens1 = self.tokenizer1(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        tokens2 = self.tokenizer2(
            text, 
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            tokens1 = {k: v.cuda() for k,v in tokens1.items()}
            tokens2 = {k: v.cuda() for k,v in tokens2.items()}
            
        # Get embeddings from both encoders
        with torch.cuda.amp.autocast():
            embed1 = self.text_encoder1(**tokens1)[0]
            embed2 = self.text_encoder2(**tokens2)[0]
            
            # Concatenate along channel dimension
            embed = torch.cat([embed1, embed2], dim=-1)
            
            # Get pooled output from second encoder
            pooled = self.text_encoder2(**tokens2)[1]
            
        # Move back to CPU for caching
        embed = embed.cpu()
        pooled = pooled.cpu()
        
        return embed, pooled
        
    def _parallel_encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel batch processing with optimal batch size."""
        batch_size = self._batch_size
        num_texts = len(texts)
        
        if num_texts <= batch_size or not self._pool:
            return self._encode_text(texts)
            
        # Split into optimal batches
        batches = [texts[i:i + batch_size] 
                  for i in range(0, num_texts, batch_size)]
        
        # Process batches in parallel
        results = self._pool.map(self._encode_text, batches)
        
        # Gather results efficiently
        embed1 = torch.cat([r[0] for r in results], dim=0)
        embed2 = torch.cat([r[1] for r in results], dim=0)
        
        return embed1, embed2
        
    def encode(self, texts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ultra-fast text encoding with caching."""
        if isinstance(texts, str):
            texts = [texts]
            
        # Check cache for each text
        embed1_list = []
        embed2_list = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            cached = self._memory_cache.get(key)
            
            if cached is not None:
                embed1, embed2 = cached
                # Ensure tensors are on CPU for pinning
                if torch.cuda.is_available():
                    embed1 = embed1.cpu()
                    embed2 = embed2.cpu()
                embed1_list.append(embed1)
                embed2_list.append(embed2)
                self._stats['hits'] += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._stats['misses'] += 1
                
        # Process uncached texts in parallel
        if uncached_texts:
            embed1_batch, embed2_batch = self._parallel_encode(uncached_texts)
            
            # Cache new embeddings
            for i, (embed1, embed2) in enumerate(zip(embed1_batch, embed2_batch)):
                key = self._get_cache_key(texts[uncached_indices[i]])
                # Store on CPU in cache
                self._memory_cache.put(key, (embed1.cpu(), embed2.cpu()))
                embed1_list.append(embed1.cpu())
                embed2_list.append(embed2.cpu())
                
        # Stack results and pin memory
        result1 = torch.stack(embed1_list)
        result2 = torch.stack(embed2_list)
        
        if torch.cuda.is_available():
            result1 = result1.pin_memory()
            result2 = result2.pin_memory()
            
        return result1, result2
        
    def clear(self) -> None:
        """Efficient cache clearing."""
        self._memory_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {**self._stats, **self._memory_cache.get_stats()}
            
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._memory_cache)
        
    def __del__(self) -> None:
        """Clean shutdown."""
        self.clear()
        if self._pool:
            self._pool.close()
            self._pool.join()
        
    def process_text(self, text: str, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text and return embeddings with optional dropout during training."""
        if not text or (training and random.random() < self.dropout_rate):
            return None, None
        
        return self.encode(text)
        
    def process_batch(self, texts: List[str], training: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process a batch of texts in parallel."""
        if not texts:
            return []
        
        # Filter out empty texts and apply dropout
        if training:
            texts = [t for t in texts if t and random.random() > self.dropout_rate]
        
        if not texts:
            return []
        
        # Process all texts at once using existing encode method
        embed1_batch, embed2_batch = self.encode(texts)
        return list(zip(embed1_batch, embed2_batch))