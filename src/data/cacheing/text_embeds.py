"""
Ultra-optimized text embedding cache system.

This module provides an ultra-optimized text embedding cache with parallel processing, 
mixed precision, and efficient memory management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
from .memory import MemoryCache
import multiprocessing
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

    def encode(self, text_input: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text input to embeddings.
        
        Args:
            text_input: Text to encode
            
        Returns:
            Tuple of (pooled_embeddings, hidden_state_embeddings)
            Both with shape (batch, sequence_length, hidden_dim)
        """
        # Check cache first
        key = self._get_cache_key(text_input)
        cached = self._memory_cache.get(key)
        
        if cached is not None:
            self._stats['hits'] += 1
            return cached
        
        self._stats['misses'] += 1
        
        # Generate embeddings for uncached text
        pooled1, hidden1 = self._process_embeddings(
            self.text_encoder1,
            self.tokenizer1,
            text_input, 
            self.device
        )
        
        pooled2, hidden2 = self._process_embeddings(
            self.text_encoder2,
            self.tokenizer2,
            text_input,
            self.device
        )

        # Concatenate embeddings
        result = (
            torch.cat([pooled1, pooled2], dim=-1),  # Concatenate pooled embeddings
            torch.cat([hidden1, hidden2], dim=-1)   # Concatenate hidden states
        )
        
        # Store in cache
        self._memory_cache.put(key, result)
        
        return result

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