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
        text_encoder: torch.nn.Module,
        tokenizer: object,
        text_input: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process text through CLIP encoder.
        Args:
            text_encoder: CLIP text encoder
            tokenizer: CLIP tokenizer
            text_input: Input text
        Returns:
            pooled: Pooled text embeddings [1, D]
            hidden: Hidden state embeddings [1, 77, D]
        """
        # Tokenize text (on CPU)
        tokens = tokenizer(
            text_input,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move tokens to device
        tokens = {
            'input_ids': tokens['input_ids'].to(self.device),
            'attention_mask': tokens['attention_mask'].to(self.device)
        }
        
        # Get embeddings
        with torch.no_grad():
            encoder_output = text_encoder(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get penultimate hidden state [1, 77, D]
            hidden = encoder_output.hidden_states[-2]
            
            # Get pooled output [1, D]
            pooled = encoder_output.text_embeds
            
            # Move to CPU for caching
            hidden = hidden.cpu()
            pooled = pooled.cpu()
            
        return pooled, hidden

    def encode(self, text_input: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text input to SDXL embeddings.
        Args:
            text_input: Input text
        Returns:
            text_embeddings: Concatenated hidden states [1, 77, D]
            pooled_text_embeddings: Second encoder pooled output [1, D]
            time_ids: SDXL time embeddings [1, 6]
        """
        # Check cache
        key = self._get_cache_key(text_input)
        cached = self._memory_cache.get(key)
        
        if cached is not None:
            self._stats['hits'] += 1
            return cached
        
        self._stats['misses'] += 1
        
        # Generate embeddings
        hidden_1 = self._process_embeddings(
            self.text_encoder_1,
            self.tokenizer_1,
            text_input
        )
        
        pooled_2, hidden_2 = self._process_embeddings(
            self.text_encoder_2,
            self.tokenizer_2,
            text_input
        )
        
        # Format outputs
        text_embeddings = torch.cat([hidden_1, hidden_2], dim=-1)  # [1, 77, D]
        pooled_text_embeddings = pooled_2  # [1, D] (only use second encoder pooled)
        
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