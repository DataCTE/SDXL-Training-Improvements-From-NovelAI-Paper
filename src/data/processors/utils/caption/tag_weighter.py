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
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
import time
from src.data.processors.utils.batch_utils import get_gpu_memory_usage

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
    def __init__(self, config: TagWeighterConfig):
        """Initialize tag weighter with configuration."""
        try:
            self.config = config
            self.tag_frequencies = defaultdict(lambda: defaultdict(int))
            self.class_totals = defaultdict(int)
            self._weight_cache = WeakValueDictionary() if config.use_cache else None
            
            # Log initialization
            logger.info(
                f"Initialized TagWeighter:\n"
                f"- Min weight: {config.min_weight}\n"
                f"- Max weight: {config.max_weight}\n"
                f"- Smoothing factor: {config.smoothing_factor}\n"
                f"- Cache enabled: {config.use_cache}"
            )
            log_system_metrics(prefix="TagWeighter initialization: ")
            
        except Exception as e:
            log_error_with_context(e, "Error initializing tag weighter")
            raise

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
        
    def update_frequencies(self, tags: Dict[str, List[str]]) -> None:
        """Update tag frequency counters with metrics tracking."""
        try:
            update_stats = {
                'total_tags': 0,
                'unique_tags': 0,
                'tag_classes': 0
            }
            
            # Track new tags
            new_tags = set()
            
            for tag_class, tag_list in tags.items():
                self.class_totals[tag_class] += len(tag_list)
                update_stats['tag_classes'] += 1
                
                for tag in tag_list:
                    self.tag_frequencies[tag_class][tag] += 1
                    update_stats['total_tags'] += 1
                    if tag not in new_tags:
                        new_tags.add(tag)
                        update_stats['unique_tags'] += 1
            
            # Log update metrics periodically
            if hasattr(self, '_update_counter'):
                self._update_counter += 1
            else:
                self._update_counter = 1
                
            if self._update_counter % 1000 == 0:  # Log every 1000 updates
                update_stats.update({
                    'total_unique_tags': len(self._weight_cache),
                    'memory_usage_mb': get_gpu_memory_usage() * 1024
                })
                log_metrics(update_stats, step=self._update_counter, step_type="tag_update")
                
        except Exception as e:
            log_error_with_context(e, "Error updating tag frequencies")

    def calculate_weights(
        self,
        tags: Dict[str, List[str]],
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate tag weights with detailed metrics."""
        try:
            weight_stats = {
                'start_time': time.time(),
                'total_tags': sum(len(tags_list) for tags_list in tags.values()),
                'tag_classes': len(tags),
                'cache_hits': 0,
                'cache_misses': 0
            }
            
            # Calculate base weights
            weights = []
            for tag_class, tag_list in tags.items():
                class_total = self.class_totals[tag_class]
                for tag in tag_list:
                    # Try cache first if enabled
                    if self.config.use_cache and self._weight_cache is not None:
                        cache_key = f"{tag_class}:{tag}"
                        cached_weight = self._weight_cache.get(cache_key)
                        
                        if cached_weight is not None:
                            weights.append(cached_weight)
                            weight_stats['cache_hits'] += 1
                            continue
                    
                    # Calculate weight
                    freq = self.tag_frequencies[tag_class][tag]
                    weight = 1.0 / (freq + self.config.smoothing_factor)
                    weights.append(weight)
                    
                    # Cache result if enabled
                    if self.config.use_cache and self._weight_cache is not None:
                        self._weight_cache[cache_key] = weight
                    weight_stats['cache_misses'] += 1
            
            weights = torch.tensor(weights, device=self.config.device)
            
            # Apply similarity factor if embeddings provided
            if embeddings is not None:
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    similarity_factors = self.calculate_similarity_factor(embeddings, attention_mask)
                    weight_stats['similarity_min'] = similarity_factors.min().item()
                    weight_stats['similarity_max'] = similarity_factors.max().item()
                    weight_stats['similarity_mean'] = similarity_factors.mean().item()
                    
                    # Use mean similarity as a modulation factor
                    similarity_weights = 1.0 / (similarity_factors.mean(dim=-1) + self.config.smoothing_factor)
                    weights = weights * similarity_weights
                    
                    # Clean up similarity tensors
                    del similarity_factors
                    del similarity_weights
                    
            # Normalize weights
            weights = weights / weights.mean()
            weights = weights.clamp(self.config.min_weight, self.config.max_weight)
            
            # Log final stats
            weight_stats.update({
                'duration': time.time() - weight_stats['start_time'],
                'min_weight': weights.min().item(),
                'max_weight': weights.max().item(),
                'mean_weight': weights.mean().item(),
                'std_weight': weights.std().item()
            })
            
            # Log metrics periodically
            if hasattr(self, '_weight_counter'):
                self._weight_counter += 1
            else:
                self._weight_counter = 1
                
            if self._weight_counter % 100 == 0:  # Log every 100 weight calculations
                log_metrics(weight_stats, step=self._weight_counter, step_type="tag_weight")
            
            return weights
            
        except Exception as e:
            log_error_with_context(e, "Error calculating tag weights")
            return torch.ones(len(tags), device=self.config.device)

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


