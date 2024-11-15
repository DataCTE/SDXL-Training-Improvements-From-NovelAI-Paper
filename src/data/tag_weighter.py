import torch
import logging
import traceback
from typing import Dict, Set, Optional, List, Tuple, Any, FrozenSet
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor
import os
from threading import Lock
import re

logger = logging.getLogger(__name__)


@dataclass
class TagStats:
    """Thread-safe container for tag statistics."""
    frequencies: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    class_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cooccurrence: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _lock: Lock = field(default_factory=Lock)

    def update(self, class_name: str, tag: str) -> None:
        """Thread-safe update of tag frequencies."""
        with self._lock:
            self.frequencies[class_name][tag] += 1
            self.class_counts[class_name] += 1

    def update_cooccurrence(self, tag1: str, tag2: str) -> None:
        """Thread-safe update of tag co-occurrence."""
        with self._lock:
            self.cooccurrence[tag1][tag2] += 1
            self.cooccurrence[tag2][tag1] += 1

@dataclass
class TagCache:
    """Cache container for tag computations."""
    rarity_scores: Dict[str, float] = field(default_factory=dict)
    importance_scores: Dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def update_rarity(self, tag: str, score: float) -> None:
        with self._lock:
            self.rarity_scores[tag] = score

    def get_rarity(self, tag: str, default: float = 1.0) -> float:
        with self._lock:
            return self.rarity_scores.get(tag, default)

class TagBasedLossWeighter:
    """Improved tag-based loss weighting system with thread safety and optimized performance."""
    
    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the tag-based loss weighting system with improved configuration.
        
        Args:
            tag_classes: Dictionary mapping tag class names to sets of tags
            config: Configuration dictionary for weighter parameters
        """
        try:
            self._init_config(config or {})
            self._init_tag_classes(tag_classes)
            self._init_statistics()
            self._init_executor()
            
            logger.info("TagBasedLossWeighter initialized successfully")
            self._log_config()
            
        except Exception as e:
            logger.error(f"TagBasedLossWeighter initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _init_config(self, config: Dict[str, Any]) -> None:
        """Initialize configuration parameters."""
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 3.0)
        self.emphasis_factor = config.get('emphasis_factor', 1.1)
        self.rarity_factor = config.get('rarity_factor', 0.9)
        self.quality_bonus = config.get('quality_bonus', 0.2)
        self.character_emphasis = config.get('character_emphasis', 1.2)
        self.cache_size = config.get('cache_size', 1024)
        self.no_cache = config.get('no_cache', False)

    def _init_tag_classes(self, tag_classes: Optional[Dict[str, Set[str]]]) -> None:
        """Initialize tag classification system."""
        self.tag_classes = tag_classes or {
            'character': set(),
            'style': set(),
            'setting': set(),
            'action': set(),
            'object': set(),
            'quality': set(),
            'emphasis': set(),
            'meta': set()
        }
        
        # Initialize class weights
        self.class_base_weights = {
            'character': 1.2,
            'style': 1.1,
            'setting': 0.9,
            'action': 1.0,
            'object': 0.8,
            'quality': 1.3,
            'emphasis': 1.4,
            'meta': 0.7
        }
        
        # Create tag to class mapping
        self.tag_to_class = {
            tag: class_name 
            for class_name, tags in self.tag_classes.items() 
            for tag in tags
        }

    def _init_statistics(self) -> None:
        """Initialize statistical tracking components."""
        self.stats = TagStats()
        self.cache = TagCache()
        
        # Configure caching behavior
        if not self.no_cache:
            self.calculate_tag_weights = lru_cache(maxsize=self.cache_size)(self._calculate_tag_weights)
        else:
            self.calculate_tag_weights = self._calculate_tag_weights

    def _init_executor(self) -> None:
        """Initialize thread pool executor."""
        self.num_workers = min(8, os.cpu_count() or 1)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def _log_config(self) -> None:
        """Log configuration settings."""
        logger.info("TagBasedLossWeighter configuration:")
        logger.info(f"- Min weight: {self.min_weight}")
        logger.info(f"- Max weight: {self.max_weight}")
        logger.info(f"- Emphasis factor: {self.emphasis_factor}")
        logger.info(f"- Rarity factor: {self.rarity_factor}")
        logger.info(f"- Cache size: {self.cache_size}")
        logger.info(f"- Workers: {self.num_workers}")

    @staticmethod
    def parse_tags(caption: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse caption into tags and special tags with improved error handling.
        """
        try:
            if not caption:
                return [], {}

            tags = []
            special_tags = {}
            
            # Process tags in a single pass
            raw_tags = [t.strip().lower() for t in caption.split(',')]
            
            # Early processing for MJ tags
            has_mj_tags = any('niji' in t or t in ['4', '5', '6'] for t in raw_tags)
            
            for i, tag in enumerate(raw_tags):
                # Skip empty tags
                if not tag:
                    continue
                    
                # Process special tag formats
                if '::' in tag:
                    tag, weight = TagBasedLossWeighter._process_weighted_tag(tag)
                    if weight is not None:
                        special_tags[f'{tag}_weight'] = weight
                
                elif has_mj_tags:
                    tag = TagBasedLossWeighter._process_mj_tag(tag, i, len(raw_tags), special_tags)
                
                # Clean and add tag
                if tag := TagBasedLossWeighter._clean_tag(tag):
                    tags.append(tag)
            
            return tags, special_tags
            
        except Exception as e:
            logger.error(f"Tag parsing failed: {str(e)}")
            return [], {}

    @staticmethod
    def _process_weighted_tag(tag: str) -> Tuple[str, Optional[float]]:
        """Process a weighted tag format (tag::weight)."""
        parts = tag.split('::')
        try:
            return parts[0].strip(), float(parts[1])
        except (IndexError, ValueError):
            return parts[0].strip(), None

    @staticmethod
    def _process_mj_tag(tag: str, index: int, total_tags: int, special_tags: Dict[str, Any]) -> Optional[str]:
        """Process Midjourney-specific tags."""
        # Handle style/version tags
        if index == 0 and ('anime style' in tag or 'niji' in tag):
            special_tags['niji'] = True
            return None
        if index == total_tags - 1 and tag in ['4', '5', '6']:
            return 'masterpiece'
            
        # Handle parameters
        for param in ['stylize', 'chaos', 'sw', 'sv']:
            if param in tag:
                try:
                    value = float(re.search(r'[\d.]+', tag).group())
                    special_tags[param] = value
                    return None
                except:
                    pass
        
        return tag

    @staticmethod
    def _clean_tag(tag: str) -> Optional[str]:
        """Clean and normalize a tag."""
        if tag.startswith(('a ', 'an ', 'the ')):
            tag = ' '.join(tag.split()[1:])
        return tag.strip() if tag.strip() else None

    @staticmethod
    def calculate_static_weights(tags: List[str], special_tags: Dict[str, any] = None) -> float:
        """
        Static method to calculate basic weights without instance-specific data.
        
        Args:
            tags (List[str]): List of tags
            special_tags (Dict): Special tag parameters
            
        Returns:
            float: Basic weight value
        """
        if special_tags is None:
            special_tags = {}
            
        base_weight = 1.0
        
        # Apply basic modifiers
        if 'masterpiece' in tags:
            base_weight *= 1.3
        if special_tags.get('niji', False):
            base_weight *= 1.2
        if 'stylize' in special_tags:
            stylize_value = special_tags['stylize']
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if 'chaos' in special_tags:
            chaos_value = special_tags['chaos']
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor
            
        # Clamp between min and max
        return max(TagBasedLossWeighter.min_weight, 
                  min(TagBasedLossWeighter.max_weight, base_weight))

    @staticmethod
    def format_caption(caption: str) -> str:
        """
        Static method to format caption text with standardized formatting.
        
        Args:
            caption (str): Raw caption text
            
        Returns:
            str: Formatted caption text
        """
        if not caption:
            return ""
            
        try:
            # Split into tags
            tags = [t.strip() for t in caption.split(',')]
            
            # Remove empty tags
            tags = [t for t in tags if t]
            
            # Basic cleanup for each tag
            formatted_tags = []
            for tag in tags:
                # Convert to lowercase
                tag = tag.lower()
                
                # Remove extra spaces
                tag = ' '.join(tag.split())
                
                # Remove articles from start
                if tag.startswith(('a ', 'an ', 'the ')):
                    tag = ' '.join(tag.split()[1:])
                
                # Handle special formatting for quality tags
                if any(q in tag for q in ['masterpiece', 'best quality', 'high quality']):
                    formatted_tags.insert(0, tag)  # Move to front
                    continue
                    
                # Handle special formatting for negative tags
                if tag.startswith(('no ', 'bad ', 'worst ')):
                    if not any(neg in tag for neg in ['negative space', 'negative prompt']):
                        tag = tag.replace('no ', '').replace('bad ', '').replace('worst ', '')
                        tag = f"lowquality {tag}"
                
                formatted_tags.append(tag)
            
            # Join tags with standardized separator
            return ', '.join(formatted_tags)
            
        except Exception as e:
            logger.error(f"Caption formatting error: {str(e)}")
            return caption  # Return original if formatting fails

    def _calculate_tag_importance(self, tag: str, tags: List[str]) -> float:
        """
        Calculate importance score for a tag based on context.
        
        Args:
            tag (str): Target tag
            tags (List[str]): All tags in the image
            
        Returns:
            float: Importance score
        """
        class_name = self._classify_tag(tag)
        if not class_name:
            return 1.0
        
        # Get base class weight
        importance = self.class_base_weights.get(class_name, 1.0)
        
        # Apply character emphasis
        if class_name == 'character':
            importance *= self.character_emphasis
        
        # Apply emphasis for emphasized tags
        if tag in self.tag_classes['emphasis']:
            importance *= self.emphasis_factor
        
        # Apply rarity bonus
        rarity_score = self.cache.get_rarity(tag, 1.0)
        importance *= rarity_score
        
        # Apply quality bonus for high-quality images
        if class_name == 'quality' and any(q in tags for q in ['masterpiece', 'best quality', 'high quality']):
            importance *= (1.0 + self.quality_bonus)
        
        return importance
    
    def _calculate_tag_weights(self, tags_tuple: Tuple[str, ...]) -> float:
        """
        Calculate tag weights with improved weighting scheme.
        
        Args:
            tags_tuple (tuple): Tuple of tags for weight calculation
            
        Returns:
            float: Calculated weight value
        """
        try:
            tags = list(tags_tuple)
            weights = []
            
            # Calculate importance for each tag
            for tag in tags:
                importance = self._calculate_tag_importance(tag, tags)
                weights.append(importance)
            
            if not weights:
                return 1.0
            
            # Calculate final weight using weighted geometric mean
            weights = torch.tensor(weights, dtype=torch.float32)
            final_weight = torch.exp(torch.log(weights + 1e-6).mean())
            
            # Clamp between min and max
            return torch.clamp(final_weight, self.min_weight, self.max_weight).item()
            
        except Exception as e:
            logger.error(f"Tag weight calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_batch_weights(self, batch_tags: List[List[str]]) -> List[float]:
        """
        Calculate weights for a batch of tag lists in parallel.
        
        Args:
            batch_tags (List[List[str]]): List of tag lists to process
            
        Returns:
            List[float]: List of calculated weights
        """
        if len(batch_tags) > 50:  # Only parallelize for larger batches
            futures = [self.executor.submit(self._calculate_tag_weights, tuple(tags)) for tags in batch_tags]
            return [future.result() for future in futures]
        else:
            return [self._calculate_tag_weights(tuple(tags)) for tags in batch_tags]

    def calculate_weights(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Calculate weights with proper no_cache handling"""
        if self.no_cache:
            # Calculate weights directly without caching
            return self._calculate_weights_no_cache(tags, special_tags)
        else:
            # Use cached calculation
            return self._calculate_weights_cached(tags, special_tags)

    def _calculate_weights_no_cache(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Direct weight calculation without caching"""
        if special_tags is None:
            special_tags = {}
            
        weights = {}
        base_weight = 1.0
        
        # Apply modifiers directly without caching
        if 'masterpiece' in tags:
            base_weight *= 1.3
        if special_tags.get('niji', False):
            base_weight *= 1.2
        if 'stylize' in special_tags:
            stylize_value = special_tags['stylize']
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if 'chaos' in special_tags:
            chaos_value = special_tags['chaos']
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor
            
        # Calculate individual weights
        for i, tag in enumerate(tags):
            # Get tag class importance
            class_name = self._classify_tag(tag)
            class_weight = self.class_base_weights.get(class_name, 1.0)
            
            # Apply position decay
            position_weight = 1.0 - (i * 0.05)
            
            # Apply rarity bonus
            rarity_score = self.cache.get_rarity(tag, 1.0)
            
            # Combine all factors
            final_weight = base_weight * class_weight * position_weight * rarity_score
            
            # Apply any explicit tag weights
            if f'{tag}_weight' in special_tags:
                final_weight *= special_tags[f'{tag}_weight']
            
            weights[tag] = max(self.min_weight, min(self.max_weight, final_weight))
            
        return weights

    def _calculate_weights_cached(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Cached weight calculation"""
        # Convert tags to tuple for caching
        tags_tuple = tuple(sorted(tags))
        return self.calculate_tag_weights(tags_tuple, frozenset(special_tags.items()) if special_tags else None)

    def calculate_weights(self, tags: List[str]) -> torch.Tensor:
        """
        Calculate tag weights with efficient caching and error handling.
        
        Args:
            tags (list): Tags to calculate weight for
            
        Returns:
            torch.Tensor: Calculated weight
        """
        try:
            tags_tuple = tuple(sorted(tags))  # Sort for consistent caching
            weight = self.calculate_tag_weights(tags_tuple)
            return torch.tensor(weight, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Tag weight calculation error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return torch.tensor(1.0)
    
    def update_training_loss(self, loss: torch.Tensor, tags: List[str]) -> torch.Tensor:
        """
        Apply tag-based weighting to the training loss.
        
        Args:
            loss (torch.Tensor): Original loss value
            tags (list): List of tags for the current image
            
        Returns:
            torch.Tensor: Weighted loss value
        """
        try:
            weight = self.calculate_weights(tags)
            return loss * weight
        except Exception as e:
            logger.error(f"Loss update failed: {str(e)}")
            return loss
    
    def reset_statistics(self):
        """Reset all statistical tracking"""
        self.stats.frequencies.clear()
        self.stats.class_counts.clear()
        self.stats.cooccurrence.clear()
        self.cache.rarity_scores.clear()
        self.cache.importance_scores.clear()
        
        # Clear caches
        self.calculate_tag_weights.cache_clear()
        self._classify_tag.cache_clear()

    def __del__(self):
        """Cleanup worker pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)