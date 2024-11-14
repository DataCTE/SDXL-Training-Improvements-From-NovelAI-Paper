import torch
import logging
import traceback
from typing import Dict, Set, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class TagBasedLossWeighter:
    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
        cache_size: int = 1024
    ):
        """
        Initialize the tag-based loss weighting system with advanced caching.
        
        Args:
            tag_classes (dict): Dictionary mapping tag class names to lists of tags
            min_weight (float): Minimum weight multiplier for any image
            max_weight (float): Maximum weight multiplier for any image
            cache_size (int): Size of LRU cache for tag classification and weight calculation
        """
        self.tag_classes = tag_classes or {
            'character': set(),
            'style': set(),
            'setting': set(),
            'action': set(),
            'object': set(),
            'quality': set()
        }
        
        # Precompute tag to class mapping for faster lookups
        self.tag_to_class = {
            tag: class_name 
            for class_name, tags in self.tag_classes.items() 
            for tag in tags
        }
        
        # Initialize class weights and tracking
        self.class_weights = {class_name: 1.0 for class_name in self.tag_classes}
        self.tag_frequencies = {class_name: {} for class_name in self.tag_classes}
        self.class_total_counts = {class_name: 0 for class_name in self.tag_classes}
        
        # Caching parameters
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # LRU cache for tag classification and weight calculation
        self.calculate_tag_weights = lru_cache(maxsize=cache_size)(self._calculate_tag_weights)
    
    @lru_cache(maxsize=1024)
    def _classify_tag(self, tag: str) -> Optional[str]:
        """
        Efficiently classify a tag to its class with caching.
        
        Args:
            tag (str): Tag to classify
        
        Returns:
            str or None: Class name for the tag, or None if not found
        """
        return self.tag_to_class.get(tag)
    
    def add_tags_to_frequency(self, tags, count=1):
        """
        Update tag frequencies with new tags, using precomputed tag classification.
        
        Args:
            tags (list): List of tags from an image
            count (int): How many times to count these tags (default: 1)
        """
        for tag in tags:
            class_name = self._classify_tag(tag)
            if class_name:
                self.tag_frequencies[class_name][tag] = (
                    self.tag_frequencies[class_name].get(tag, 0) + count
                )
                self.class_total_counts[class_name] += count
    
    def _calculate_tag_weights(self, tags_tuple):
        """
        Internal method for calculating tag weights with efficient processing.
        
        Args:
            tags_tuple (tuple): Tuple of tags for weight calculation
        
        Returns:
            float: Calculated weight value
        """
        try:
            weights = []
            for class_name in self.tag_classes:
                class_tags = self.tag_classes[class_name]
                tag_set = set(tags_tuple)
                
                class_intersection = class_tags.intersection(tag_set)
                weight = self.class_weights.get(class_name, 1.0)
                weights.append(weight if class_intersection else 1.0)
            
            # Calculate final weight and clamp between min and max
            final_weight = torch.tensor(weights, dtype=torch.float32).mean()
            return torch.clamp(final_weight, self.min_weight, self.max_weight).item()
            
        except Exception as e:
            logger.error(f"Tag weight calculation failed: {str(e)}")
            return 1.0
    
    def calculate_tag_weights(self, tags):
        """
        Calculate tag weights with efficient caching and error handling.
        
        Args:
            tags (list or tuple): Tags to calculate weight for
        
        Returns:
            torch.Tensor: Calculated weight
        """
        try:
            # Normalize input to tuple for caching
            if isinstance(tags, list):
                tags_tuple = tuple(tuple(t) if isinstance(t, list) else t for t in tags)
            else:
                tags_tuple = tags
            
            weight = self.calculate_tag_weights(tags_tuple)
            return torch.tensor(weight, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Tag weight calculation error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return torch.tensor(1.0)
    
    def update_training_loss(self, loss, tags):
        """
        Apply tag-based weighting to the training loss with efficient processing.
        
        Args:
            loss (torch.Tensor): Original loss value
            tags (list): List of tags for the current image/batch
            
        Returns:
            torch.Tensor: Weighted loss value
        """
        try:
            weight = self.calculate_tag_weights(tags)
            return loss * weight
        except Exception as e:
            logger.error(f"Loss update failed: {str(e)}")
            return loss  # Return original loss on error
    
    def reset_frequencies(self):
        """
        Reset tag frequencies and class total counts.
        """
        self.tag_frequencies = {class_name: {} for class_name in self.tag_classes}
        self.class_total_counts = {class_name: 0 for class_name in self.tag_classes}
        
        # Clear caches to ensure fresh calculations
        self.calculate_tag_weights.cache_clear()
        self._classify_tag.cache_clear()