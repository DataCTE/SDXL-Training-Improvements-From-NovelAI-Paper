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
        num_workers: int = 4,
        max_memory_gb: float = 32.0
    ) -> None:
        """Initialize text embedding cache with dual encoders."""
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.tokenizer1 = tokenizer1 
        self.tokenizer2 = tokenizer2
        
        # Initialize cache and stats
        self._memory_cache = MemoryCache(
            max_memory_gb=max_memory_gb,
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
        """Encode text following SDXL's dual encoder architecture exactly.
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - hidden_states: Combined text embeddings [batch_size, seq_len, 768+1280]
                - pooled: Second encoder pooled output [batch_size, 1280]
        """
        if isinstance(text, str):
            text = [text]
            
        # Tokenize with both encoders
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
            
        with torch.amp.autocast('cuda'):
            # First encoder (CLIP-L/14)
            outputs1 = self.text_encoder1(**tokens1)
            hidden_states1 = outputs1.last_hidden_state  # [batch, 77, 768]
            
            # Second encoder (CLIP-G/14)
            outputs2 = self.text_encoder2(**tokens2)
            hidden_states2 = outputs2.last_hidden_state  # [batch, 77, 1280]
            
            # Get pooled output from second encoder's last hidden state
            # Take the first token's embedding (CLS token) as pooled representation
            pooled_output = hidden_states2[:, 0]  # [batch, 1280]
            
            # Verify shapes match SDXL exactly
            assert hidden_states1.shape[-1] == 768, f"CLIP-L hidden size must be 768, got {hidden_states1.shape[-1]}"
            assert hidden_states2.shape[-1] == 1280, f"CLIP-G hidden size must be 1280, got {hidden_states2.shape[-1]}"
            assert hidden_states1.shape[1] == hidden_states2.shape[1] == 77, f"Sequence length must be 77"
            assert pooled_output.shape[-1] == 1280, f"Pooled output size must be 1280, got {pooled_output.shape[-1]}"
                    
            # Concatenate along hidden dimension as per SDXL
            combined_hidden = torch.cat([hidden_states1, hidden_states2], dim=-1)
            
        # Move to CPU for caching
        combined_hidden = combined_hidden.cpu()
        pooled_output = pooled_output.cpu()
        
        return combined_hidden, pooled_output
        
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

    def _evict_items(self) -> None:
        """Evict items when cache is full."""
        if len(self._memory_cache) > self._max_cache_size:
            # Calculate number of items to evict (20% of cache)
            evict_count = max(1, int(self._max_cache_size * 0.2))
            self._memory_cache.evict(evict_count)

    def __delitem__(self, key: str) -> None:
        """Support item deletion."""
        if key in self._memory_cache:
            del self._memory_cache[key]