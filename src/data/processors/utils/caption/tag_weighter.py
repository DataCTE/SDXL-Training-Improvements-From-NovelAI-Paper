import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import re
import gc
from src.config.config import TagWeighterConfig
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

def parse_tags(caption: str) -> Dict[str, List[str]]:
    """Parse a caption string into categorized tags."""
    tags: Dict[str, List[str]] = defaultdict(list)
    
    try:
        # Find all bracketed sections
        brackets = re.findall(r'\[(.*?)\]', caption)
        
        for bracket in brackets:
            if '::' in bracket:
                # Handle class-specific tags
                tag_class, tag_list = bracket.split('::', 1)
                tag_class = tag_class.strip()
                for tag in tag_list.split(','):
                    tag = tag.strip()
                    if tag:
                        tags[tag_class].append(tag)
            else:
                # Handle general tags without class
                for tag in bracket.split(','):
                    tag = tag.strip()
                    if tag:
                        tags['general'].append(tag)
                        
        return dict(tags)
        
    except Exception as e:
        logger.error(f"Error parsing tags: {e}")
        return {}
    
class TagWeighter:
    def __init__(self, config: Optional[TagWeighterConfig] = None):
        """Initialize tag weighter with configuration."""
        self.config = config or TagWeighterConfig()
        
        # Track tag frequencies per class using defaultdict for automatic cleanup
        self.tag_frequencies = defaultdict(lambda: defaultdict(int))
        self.class_totals = defaultdict(int)
        
        # Cache for computed weights using weak references
        self._weight_cache = WeakValueDictionary()
        
        # Tensor cache for reuse
        self._tensor_cache = WeakValueDictionary()
        
    def __del__(self):
        """Cleanup when weighter is deleted."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            if hasattr(self, '_weight_cache'):
                self._weight_cache.clear()
            if hasattr(self, '_tensor_cache'):
                self._tensor_cache.clear()
            if hasattr(self, 'tag_frequencies'):
                self.tag_frequencies.clear()
            if hasattr(self, 'class_totals'):
                self.class_totals.clear()
                
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
    def update_frequencies(self, tag_class: str, tag: str):
        """Update frequency counters for a tag in its class."""
        self.tag_frequencies[tag_class][tag] += 1
        self.class_totals[tag_class] += 1
        
        # Clear cache when frequencies update
        self._weight_cache.clear()
        gc.collect()  # Help clean up old cached values
        
    def get_tag_weight(self, tag_class: str, tag: str) -> float:
        """Calculate weight for a tag based on its frequency in its class."""
        cache_key = (tag_class, tag)
        
        try:
            # Try to get from cache first
            cached_weight = self._weight_cache.get(cache_key)
            if cached_weight is not None:
                return cached_weight
                
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
            
            # Cache the computed weight
            self._weight_cache[cache_key] = weight
            return weight
            
        except Exception as e:
            logger.warning(f"Error calculating tag weight: {e}")
            return self.config.default_weight
            
    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self._tensor_cache[key] = tensor
        return tensor
            
    def calculate_similarity_factor(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate similarity factors between embeddings with proper masking."""
        try:
            batch_size, seq_len, hidden_dim = embeddings.shape
            device = embeddings.device
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = self._get_cached_tensor(
                    'attention_mask',
                    (batch_size, seq_len),
                    embeddings.dtype,
                    device
                ).fill_(1)
                
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
            
            # Clean up intermediate tensors
            del norm_embeddings
            del similarity
            del masked_similarity
            del seq_lengths
            
            return mean_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating similarity factor: {e}")
            # Return ones as fallback
            return torch.ones((batch_size, seq_len), 
                            device=embeddings.device,
                            dtype=embeddings.dtype)
        finally:
            # Clear any remaining intermediate tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                            
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
            device = loss.device
            weights = self._get_cached_tensor(
                'weights',
                (batch_size,),
                self.config.dtype,
                device
            ).fill_(1)
            
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
                    
                    # Clean up similarity tensors
                    del similarity_factors
                    del similarity_weights
                    
            # Normalize weights
            weights = weights / weights.mean()
            weights = weights.clamp(self.config.min_weight, self.config.max_weight)
            
            # Apply weights to loss
            weighted_loss = loss * weights
            result = weighted_loss.mean()
            
            # Clean up intermediate tensors
            del weights
            del weighted_loss
            
            return result
            
        except Exception as e:
            logger.error(f"Error weighting loss: {e}")
            return loss.mean()
            
        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def reset_statistics(self):
        """Reset all frequency counters and cache."""
        self.tag_frequencies.clear()
        self.class_totals.clear()
        self._weight_cache.clear()
        gc.collect()  # Force cleanup of cleared data


