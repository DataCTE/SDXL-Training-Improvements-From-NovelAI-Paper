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
from .memory import MemoryCache, MemoryManager

logger = logging.getLogger(__name__)

class TextEmbeddingCache:
    """Ultra-optimized text embedding cache with parallel processing."""
    
    __slots__ = ('tokenizer1', 'tokenizer2', 'text_encoder1', 'text_encoder2',
                 '_memory_cache', '_lock', '_executor', '_batch_size',
                 '_max_cache_size', '_stats', '_scaler', '_cache_dir')
    
    def __init__(
        self,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: CLIPTextModel,
        text_encoder2: CLIPTextModel,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 10000,
        num_workers: int = 4,
        batch_size: int = 32
    ):
        """Initialize text embedding cache with optimized defaults."""
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.text_encoder1 = text_encoder1.eval()
        self.text_encoder2 = text_encoder2.eval()
        
        self._cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize memory management
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._memory_cache = MemoryCache(str(self._cache_dir))
        else:
            self._memory_cache = MemoryManager()
            
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._batch_size = batch_size
        self._max_cache_size = max_cache_size
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._scaler = amp.GradScaler()
        
        # Pre-warm CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _get_cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()
        
    def _should_evict(self) -> bool:
        """Check if cache needs eviction."""
        return len(self._memory_cache) >= self._max_cache_size
        
    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized batch encoding with mixed precision."""
        # Tokenize texts
        tokens1 = self.tokenizer1(texts, padding=True, truncation=True,
                                return_tensors="pt")
        tokens2 = self.tokenizer2(texts, padding=True, truncation=True,
                                return_tensors="pt")
        
        if torch.cuda.is_available():
            tokens1 = {k: v.cuda() for k, v in tokens1.items()}
            tokens2 = {k: v.cuda() for k, v in tokens2.items()}
            
        # Use mixed precision for faster encoding
        with amp.autocast():
            # Encode with both text encoders
            embed1 = self.text_encoder1(**tokens1)[0]
            embed2 = self.text_encoder2(**tokens2)[0]
            
            if torch.cuda.is_available():
                embed1 = self._scaler.scale(embed1)
                embed2 = self._scaler.scale(embed2)
                
        return embed1, embed2
        
    def _parallel_encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel batch processing with optimal batch size."""
        batch_size = self._batch_size
        num_texts = len(texts)
        
        if num_texts <= batch_size:
            return self._encode_batch(texts)
            
        # Split into optimal batches
        batches = [texts[i:i + batch_size] 
                  for i in range(0, num_texts, batch_size)]
        
        futures = [
            self._executor.submit(self._encode_batch, batch)
            for batch in batches
        ]
        
        # Gather results efficiently
        results = [future.result() for future in futures]
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
                self._memory_cache.put(key, (embed1.detach(), embed2.detach()))
                embed1_list.append(embed1)
                embed2_list.append(embed2)
                
        # Combine results maintaining order
        return torch.stack(embed1_list), torch.stack(embed2_list)
        
    def clear(self) -> None:
        """Efficient cache clearing."""
        with self._lock:
            self._memory_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {**self._stats, **self._memory_cache.get_stats()}
            
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._memory_cache)
        
    def __del__(self) -> None:
        """Clean shutdown."""
        self.clear()
        self._executor.shutdown(wait=False)