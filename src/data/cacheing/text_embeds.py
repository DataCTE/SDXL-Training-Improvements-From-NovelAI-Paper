"""Ultra-optimized text embedding cache system for SDXL."""

import torch
from typing import Dict, Tuple
import logging
from .memory import MemoryCache
import hashlib

logger = logging.getLogger(__name__)

class TextEmbeddingCache:
    """SDXL text embedding cache with mixed precision and memory management."""
    
    def __init__(
        self,
        text_encoder_1: torch.nn.Module,  # CLIP ViT-L/14
        text_encoder_2: torch.nn.Module,  # CLIP ViT-G/14
        tokenizer_1: object,  # CLIPTokenizer
        tokenizer_2: object,  # CLIPTokenizer
        device: torch.device = torch.device("cuda"),
        max_memory_gb: float = 32.0,
        max_cache_size: int = 100000,
        cache_dir: str = None
    ):
        """
        Initialize SDXL text embedding cache.
        Args:
            text_encoder_1: First CLIP text encoder (ViT-L/14)
            text_encoder_2: Second CLIP text encoder (ViT-G/14)
            tokenizer_1: First CLIP tokenizer
            tokenizer_2: Second CLIP tokenizer
            device: Compute device
            max_memory_gb: Maximum cache memory in GB
            max_cache_size: Maximum number of cached items
            cache_dir: Optional path to persistent cache
        """
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.device = device
        
        # Initialize memory cache
        self._memory_cache = MemoryCache(
            max_memory_gb=max_memory_gb,
            max_cache_size=max_cache_size,
            cache_dir=cache_dir
        )
        
        # Initialize stats
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def _process_embeddings(
        self,
        text: str,
        encoder: torch.nn.Module,
        tokenizer: object,
        max_length: int = 77
    ) -> torch.Tensor:
        """Process text through CLIP encoder with proper output handling."""
        
        # Tokenize text
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get encoder output
        with torch.no_grad():
            encoder_output = encoder(**tokens)
        
        # Get hidden states and mean pool
        hidden_states = encoder_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Mean pool over sequence length (excluding padding)
        attention_mask = tokens.attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        # Normalize embeddings
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        
        return pooled

    def encode(self, text_input: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text input to SDXL embeddings.
        Args:
            text_input: Input text
        Returns:
            text_embeddings: First encoder hidden states [1, 77, D]
            pooled_text_embeddings: Second encoder pooled output [1, D]
        """
        # Check cache
        key = self._get_cache_key(text_input)
        cached = self._memory_cache.get(key)
        
        if cached is not None:
            self._stats['hits'] += 1
            return cached
        
        self._stats['misses'] += 1
        
        # Generate embeddings
        with torch.no_grad():
            # First encoder: get hidden states
            tokens_1 = self.tokenizer_1(
                text_input,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            hidden_1 = self.text_encoder_1(**tokens_1).last_hidden_state
            
            # Second encoder: get pooled output
            pooled_2 = self._process_embeddings(
                text_input,
                self.text_encoder_2,
                self.tokenizer_2,
                77
            )
        
        # Format outputs
        text_embeddings = hidden_1  # [1, 77, D]
        pooled_text_embeddings = pooled_2  # [1, D]
        
        # Cache results
        result = (text_embeddings, pooled_text_embeddings)
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
        """Get number of cached items."""
        return len(self._memory_cache)