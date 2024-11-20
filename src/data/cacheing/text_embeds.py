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
    """Ultra-optimized text embedding cache system."""
    
    def __init__(
        self,
        text_encoder1,
        text_encoder2,
        tokenizer1,
        tokenizer2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_memory_gb: float = 32.0,
        max_cache_size: int = 100000,
        batch_size: int = 16,
        num_workers: int = 4,
        cache_dir: str = None
    ):
        """Initialize cache with encoders and settings."""
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.device = device
        self._batch_size = batch_size
        
        # Initialize memory cache
        self._memory_cache = MemoryCache(
            max_memory_gb=max_memory_gb,
            max_cache_size=max_cache_size,
            cache_dir=cache_dir
        )
        
        # Setup parallel processing
        self._pool = Pool(num_workers) if num_workers > 0 else None
        
        # Initialize stats
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def _process_embeddings(self, text_encoder, tokenizer, text_input, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text through encoder and handle dimensionality."""
        # Tokenize and encode text
        tokens = tokenizer(
            text_input,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get embeddings from text encoder
        with torch.no_grad():
            encoder_output = text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get pooled output (2D) and last hidden state (3D)
            pooled = encoder_output.pooled_output  # [batch, hidden_dim]
            hidden = encoder_output.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # Add sequence dimension to pooled output to match hidden states
            pooled = pooled.unsqueeze(1)  # [batch, 1, hidden_dim]
            
        return pooled, hidden

    def encode_text(self, text_input: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text embeddings with matching dimensions.
        
        Returns:
            Tuple of (pooled_embeddings, hidden_state_embeddings)
            Both with shape (batch, sequence_length, hidden_dim)
        """
        # Convert single string to list
        if isinstance(text_input, str):
            text_input = [text_input]
            
        # Check cache for each text
        embed1_list = []
        embed2_list = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(text_input):
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
                
        # Process uncached texts
        if uncached_texts:
            for text in uncached_texts:
                # Generate embeddings
                pooled1, hidden1 = self._process_embeddings(
                    self.text_encoder1, 
                    self.tokenizer1,
                    text,
                    self.device
                )
                
                pooled2, hidden2 = self._process_embeddings(
                    self.text_encoder2,
                    self.tokenizer2, 
                    text,
                    self.device
                )

                # Concatenate embeddings
                result = (
                    torch.cat([pooled1, pooled2], dim=-1),  # Concatenate pooled embeddings
                    torch.cat([hidden1, hidden2], dim=-1)   # Concatenate hidden states
                )
                
                # Store in cache
                key = self._get_cache_key(text)
                self._memory_cache.put(key, result)
                
                # Add to results
                embed1_list.append(result[0])
                embed2_list.append(result[1])
        
        # Stack results
        final_pooled = torch.cat(embed1_list, dim=0)
        final_hidden = torch.cat(embed2_list, dim=0)
        
        return final_pooled, final_hidden

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text input."""
        return hashlib.md5(text.encode()).hexdigest()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear the cache."""
        self._memory_cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._memory_cache)

    def __del__(self):
        """Clean up resources."""
        if self._pool:
            self._pool.close()
            self._pool.join()