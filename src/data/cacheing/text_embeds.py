"""
Text embeddings caching functionality for SDXL training.

This module provides specialized caching for text embeddings with efficient
memory management and disk persistence.
"""

import logging
import torch
from typing import Optional, List, Any, Dict, Tuple
from tqdm import tqdm

from .memory import MemoryCache

logger = logging.getLogger(__name__)

class TextEmbeddingCache:
    """Manages caching of text embeddings.
    
    This class provides efficient caching of text embeddings with memory
    management and disk persistence capabilities.
    
    Attributes:
        cache_dir: Directory for storing cached embeddings
        text_encoder: Text encoder model for generating embeddings
        memory_cache: Memory cache instance for storage
    """
    
    def __init__(
        self,
        cache_dir: str = "text_embeds_cache",
        text_encoder: Optional[Any] = None
    ) -> None:
        """Initialize the text embeddings cache.
        
        Args:
            cache_dir: Directory for storing cached embeddings
            text_encoder: Text encoder model for generating embeddings
        """
        self.memory_cache = MemoryCache(cache_dir)
        self.text_encoder = text_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if text_encoder is not None:
            self.text_encoder.to(self.device)
            
    def get_embeddings(
        self,
        text: str,
        return_pooled: bool = True
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Get text embeddings from cache or compute them.
        
        Args:
            text: Input text to get embeddings for
            return_pooled: Whether to return pooled embeddings
            
        Returns:
            Tuple of (text_embeddings, pooled_embeddings) if available
        """
        cache_key = f"{text}_{return_pooled}"
        cached = self.memory_cache.get(cache_key)
        
        if cached is not None:
            return cached
            
        if self.text_encoder is not None:
            with torch.amp.autocast('cuda', enabled=True):
                with torch.no_grad():
                    inputs = self.text_encoder.tokenizer(
                        text,
                        padding="max_length",
                        max_length=self.text_encoder.max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.text_encoder(**inputs)
                    text_embeddings = outputs.last_hidden_state
                    pooled_embeddings = outputs.pooler_output if return_pooled else None
                    
                    # Move to CPU to save GPU memory
                    embeddings = (
                        text_embeddings.cpu(),
                        pooled_embeddings.cpu() if pooled_embeddings is not None else None
                    )
                    
            self.memory_cache.put(cache_key, embeddings)
            return embeddings
            
        return None
        
    def process_batch(
        self,
        texts: List[str],
        return_pooled: bool = True,
        pbar: Optional[tqdm] = None
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Process a batch of texts to get their embeddings.
        
        Args:
            texts: List of input texts
            return_pooled: Whether to return pooled embeddings
            pbar: Optional progress bar
            
        Returns:
            Dictionary mapping texts to their embeddings
        """
        results = {}
        batch_size = 64  # Process in batches for memory efficiency
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                embeddings = self.get_embeddings(text, return_pooled)
                if embeddings is not None:
                    results[text] = embeddings
                    
            if pbar is not None:
                pbar.update(len(batch))
                
        return results
        
    def offload_to_disk(self) -> None:
        """Offload cache to disk."""
        self.memory_cache.offload_to_disk()