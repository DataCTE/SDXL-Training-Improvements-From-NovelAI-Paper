import torch
import logging
import traceback
from typing import Dict, Set, Optional, List, Tuple
from functools import lru_cache
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class TagBasedLossWeighter:
    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
        cache_size: int = 1024,
        emphasis_factor: float = 1.1,
        rarity_factor: float = 0.9,
        quality_bonus: float = 0.2,
        character_emphasis: float = 1.2
    ):
        """
        Initialize the tag-based loss weighting system with NovelAI improvements.
        
        Args:
            tag_classes (dict): Dictionary mapping tag class names to lists of tags
            min_weight (float): Minimum weight multiplier for any image
            max_weight (float): Maximum weight multiplier for any image
            cache_size (int): Size of LRU cache for tag classification and weight calculation
            emphasis_factor (float): Multiplier for emphasized tags
            rarity_factor (float): Multiplier for rare tags
            quality_bonus (float): Additional weight for high-quality images
            character_emphasis (float): Special multiplier for character tags
        """
        self.tag_classes = tag_classes or {
            'character': set(),  # Character-specific tags
            'style': set(),     # Artistic style tags
            'setting': set(),   # Background and environment tags
            'action': set(),    # Pose and action tags
            'object': set(),    # Props and objects
            'quality': set(),   # Image quality indicators
            'emphasis': set(),  # Tags that should receive extra weight
            'meta': set()       # Meta tags for special handling
        }
        
        # Advanced weighting parameters
        self.emphasis_factor = emphasis_factor
        self.rarity_factor = rarity_factor
        self.quality_bonus = quality_bonus
        self.character_emphasis = character_emphasis
        
        # Initialize frequency tracking with defaultdict
        self.tag_frequencies = defaultdict(lambda: defaultdict(int))
        self.class_total_counts = defaultdict(int)
        self.tag_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Precompute mappings
        self._initialize_mappings()
        
        # Caching parameters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cache_size = cache_size
        
        # Initialize caches
        self.calculate_tag_weights = lru_cache(maxsize=cache_size)(self._calculate_tag_weights)
        self._tag_rarity_scores = {}
        self._tag_importance_scores = {}
    
    def _initialize_mappings(self):
        """Initialize tag mappings and importance scores"""
        self.tag_to_class = {
            tag: class_name 
            for class_name, tags in self.tag_classes.items() 
            for tag in tags
        }
        
        # Initialize importance scores based on tag classes
        self.class_base_weights = {
            'character': 1.2,    # Character tags get higher base weight
            'style': 1.1,       # Style tags are important for consistency
            'setting': 0.9,     # Background elements get slightly lower weight
            'action': 1.0,      # Action tags get normal weight
            'object': 0.8,      # Object tags get lower weight
            'quality': 1.3,     # Quality tags get higher weight
            'emphasis': 1.4,    # Emphasized tags get highest weight
            'meta': 0.7         # Meta tags get lowest weight
        }
    
    def update_tag_statistics(self, batch_tags: List[List[str]]):
        """
        Update tag statistics with a batch of tags, including co-occurrence.
        
        Args:
            batch_tags (List[List[str]]): List of tag lists from a batch of images
        """
        for tags in batch_tags:
            # Update individual tag frequencies
            for tag in tags:
                class_name = self._classify_tag(tag)
                if class_name:
                    self.tag_frequencies[class_name][tag] += 1
                    self.class_total_counts[class_name] += 1
            
            # Update co-occurrence matrix
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    self.tag_cooccurrence[tag1][tag2] += 1
                    self.tag_cooccurrence[tag2][tag1] += 1
        
        # Recalculate rarity scores
        self._update_rarity_scores()
    
    def _update_rarity_scores(self):
        """Update tag rarity scores based on frequency distribution"""
        total_images = max(sum(self.class_total_counts.values()), 1)
        
        for class_name, tags in self.tag_frequencies.items():
            for tag, freq in tags.items():
                # Calculate normalized frequency
                norm_freq = freq / total_images
                
                # Calculate rarity score with smoothing
                rarity = 1.0 - np.sqrt(norm_freq)
                rarity = np.clip(rarity * self.rarity_factor, 0.5, 2.0)
                
                self._tag_rarity_scores[tag] = rarity
    
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
        rarity_score = self._tag_rarity_scores.get(tag, 1.0)
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
        self.tag_frequencies.clear()
        self.class_total_counts.clear()
        self.tag_cooccurrence.clear()
        self._tag_rarity_scores.clear()
        self._tag_importance_scores.clear()
        
        # Clear caches
        self.calculate_tag_weights.cache_clear()
        self._classify_tag.cache_clear()