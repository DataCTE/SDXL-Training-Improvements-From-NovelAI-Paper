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

logger = logging.getLogger(__name__)

class TextEmbeddingCache:
    """Ultra-optimized text embedding cache with parallel processing."""
    
    __slots__ = ('tokenizer1', 'tokenizer2', 'text_encoder1', 'text_encoder2',
                 '_cache', '_lock', '_executor', '_batch_size', '_max_cache_size',
                 '_stats', '_scaler')
    
    def __init__(
        self,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: CLIPTextModel,
        text_encoder2: CLIPTextModel,
        max_cache_size: int = 10000,
        num_workers: int = 4,
        batch_size: int = 32
    ):
        """Initialize text embedding cache with optimized defaults."""
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.text_encoder1 = text_encoder1.eval()
        self.text_encoder2 = text_encoder2.eval()
        
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._batch_size = batch_size
        self._max_cache_size = max_cache_size
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._scaler = amp.GradScaler()
        
        # Pre-warm CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    def _get_cache_key(self, text: str) -> str:
        """Ultra-fast cache key generation using SHA-256."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _should_evict(self) -> bool:
        """Check if cache needs eviction."""
        return len(self._cache) >= self._max_cache_size
    
    def _evict_items(self) -> None:
        """Efficient batch eviction."""
        with self._lock:
            while self._should_evict():
                _, tensors = self._cache.popitem(last=False)
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor):
                        tensor.detach_()
                        del tensor
                self._stats['evictions'] += 1
    
    @torch.no_grad()
    def _encode_text_batch(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized batch text encoding with mixed precision."""
        # Tokenize
        tokens1 = self.tokenizer1(
            texts,
            padding="max_length",
            max_length=self.tokenizer1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens2 = self.tokenizer2(
            texts,
            padding="max_length",
            max_length=self.tokenizer2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            tokens1 = tokens1.to('cuda')
            tokens2 = tokens2.to('cuda')
            
        # Encode with mixed precision
        with amp.autocast():
            encoder1_hidden = self.text_encoder1(
                tokens1.input_ids,
                attention_mask=tokens1.attention_mask
            )[0]
            
            encoder2_hidden = self.text_encoder2(
                tokens2.input_ids,
                attention_mask=tokens2.attention_mask
            )[0]
            
            # Scale outputs
            encoder1_hidden = self._scaler.scale(encoder1_hidden)
            encoder2_hidden = self._scaler.scale(encoder2_hidden)
            
            # Move to CPU if needed
            if torch.cuda.is_available():
                encoder1_hidden = encoder1_hidden.cpu()
                encoder2_hidden = encoder2_hidden.cpu()
                tokens1 = tokens1.cpu()
                tokens2 = tokens2.cpu()
            
        return encoder1_hidden, encoder2_hidden, tokens1.attention_mask, tokens2.attention_mask
    
    def _parallel_encode(
        self,
        texts: List[str]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Parallel batch processing with optimal batch size."""
        batch_size = self._batch_size
        num_texts = len(texts)
        
        if num_texts <= batch_size:
            return [self._encode_text_batch(texts)]
            
        # Split into optimal batches
        batches = [
            texts[i:i + batch_size]
            for i in range(0, num_texts, batch_size)
        ]
        
        # Process in parallel
        futures = [
            self._executor.submit(self._encode_text_batch, batch)
            for batch in batches
        ]
        
        return [future.result() for future in futures]
    
    def encode_text(
        self,
        text: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ultra-fast text encoding with caching."""
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False
            
        # Check cache for each text
        encoded_list = []
        uncached_indices = []
        uncached_texts = []
        
        for i, t in enumerate(texts):
            key = self._get_cache_key(t)
            cached = self._cache.get(key)
            
            if cached is not None:
                encoded_list.append(cached)
                self._stats['hits'] += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)
                self._stats['misses'] += 1
        
        # Process uncached texts in parallel
        if uncached_texts:
            batch_results = self._parallel_encode(uncached_texts)
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                for i in range(len(batch[0])):
                    result = tuple(x[i] for x in batch)
                    all_results.append(result)
            
            # Cache new encodings
            for i, result in zip(uncached_indices, all_results):
                key = self._get_cache_key(texts[i])
                if self._should_evict():
                    self._evict_items()
                self._cache[key] = result
                encoded_list.append(result)
        
        # Combine results maintaining order
        if single:
            return encoded_list[0]
            
        # Stack tensors from tuples
        return tuple(
            torch.stack([x[i] for x in encoded_list])
            for i in range(4)
        )
    
    def clear(self) -> None:
        """Efficient cache clearing."""
        with self._lock:
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return dict(self._stats)
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._cache)
    
    def __del__(self) -> None:
        """Clean shutdown."""
        self.clear()
        self._executor.shutdown(wait=False)