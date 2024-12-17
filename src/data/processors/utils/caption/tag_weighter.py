import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from src.config.config import TagWeighterConfig

logger = logging.getLogger(__name__)

@dataclass

    
class TagWeighter:
    def __init__(self, config: Optional[TagWeighterConfig] = None):
        """Initialize tag weighter with configuration."""
        self.config = config or TagWeighterConfig()
        
        # Track tag frequencies per class
        self.tag_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.class_totals: Dict[str, int] = defaultdict(int)
        
        # Cache for computed weights
        self._weight_cache: Dict[Tuple[str, str], float] = {}
        
    def update_frequencies(self, tag_class: str, tag: str):
        """Update frequency counters for a tag in its class."""
        self.tag_frequencies[tag_class][tag] += 1
        self.class_totals[tag_class] += 1
        
        # Clear cache when frequencies update
        self._weight_cache.clear()
        
    def get_tag_weight(self, tag_class: str, tag: str) -> float:
        """Calculate weight for a tag based on its frequency in its class."""
        cache_key = (tag_class, tag)
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]
            
        try:
            class_total = self.class_totals[tag_class]
            if class_total == 0:
                return self.config.default_weight
                
            tag_freq = self.tag_frequencies[tag_class][tag]
            if tag_freq == 0:
                return self.config.max_weight
                
            # Calculate relative frequency with smoothing
            smoothed_freq = (tag_freq + self.config.smoothing_factor) / (class_total + self.config.smoothing_factor)
            
            # Inverse frequency weighting with bounds
            weight = 1.0 / (smoothed_freq + self.config.smoothing_factor)
            weight = max(self.config.min_weight, min(self.config.max_weight, weight))
            
            self._weight_cache[cache_key] = weight
            return weight
            
        except Exception as e:
            logger.warning(f"Error calculating tag weight: {e}")
            return self.config.default_weight
            
    def calculate_similarity_factor(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate similarity factors between embeddings with proper masking."""
        try:
            batch_size, seq_len, hidden_dim = embeddings.shape
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), 
                                         device=embeddings.device,
                                         dtype=embeddings.dtype)
                                         
            # Expand mask for attention
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            
            # Calculate cosine similarity with proper masking
            norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
            similarity = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))
            
            # Apply attention mask
            masked_similarity = similarity * attention_mask
            
            # Calculate mean similarity per sequence
            seq_lengths = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_similarity = masked_similarity.sum(dim=-1) / seq_lengths
            
            return mean_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating similarity factor: {e}")
            # Return ones as fallback
            return torch.ones((batch_size, seq_len), 
                            device=embeddings.device,
                            dtype=embeddings.dtype)
                            
    def weight_loss(
        self,
        loss: torch.Tensor,
        tags: List[Dict[str, List[str]]],
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weight loss based on tags and optional embedding similarity."""
        try:
            batch_size = len(tags)
            weights = torch.ones(batch_size, device=loss.device, dtype=self.config.dtype)
            
            # Calculate tag-based weights
            for i, sample_tags in enumerate(tags):
                sample_weight = 1.0
                for tag_class, tag_list in sample_tags.items():
                    for tag in tag_list:
                        tag_weight = self.get_tag_weight(tag_class, tag)
                        sample_weight *= tag_weight
                weights[i] = sample_weight
                
            # Apply similarity factor if embeddings provided
            if embeddings is not None:
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    similarity_factors = self.calculate_similarity_factor(embeddings, attention_mask)
                    # Use mean similarity as a modulation factor
                    similarity_weights = 1.0 / (similarity_factors.mean(dim=-1) + self.config.smoothing_factor)
                    weights = weights * similarity_weights
                    
            # Normalize weights
            weights = weights / weights.mean()
            weights = weights.clamp(self.config.min_weight, self.config.max_weight)
            
            # Apply weights to loss
            weighted_loss = loss * weights
            return weighted_loss.mean()
            
        except Exception as e:
            logger.error(f"Error weighting loss: {e}")
            return loss.mean()
            
    def reset_statistics(self):
        """Reset all frequency counters and cache."""
        self.tag_frequencies.clear()
        self.class_totals.clear()
        self._weight_cache.clear()


