import torch
import logging
import traceback
from typing import Dict, Set, Optional

logger = logging.getLogger(__name__)

class TagBasedLossWeighter:
    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        min_weight: float = 0.1,
        max_weight: float = 3.0
    ):
        """
        Initialize the tag-based loss weighting system.
        
        Args:
            tag_classes (dict): Dictionary mapping tag class names to lists of tags
                              e.g., {'character': ['girl', 'boy'], 'style': ['anime', 'sketch']}
            min_weight (float): Minimum weight multiplier for any image
            max_weight (float): Maximum weight multiplier for any image
        """
        self.tag_classes = tag_classes or {
            'character': set(),
            'style': set(),
            'setting': set(),
            'action': set(),
            'object': set(),
            'quality': set()
        }
        
        # Initialize class weights with default value of 1.0
        self.class_weights = {class_name: 1.0 for class_name in self.tag_classes}
        
        # Track tag frequencies per class
        self.tag_frequencies = {class_name: {} for class_name in self.tag_classes}
        self.class_total_counts = {class_name: 0 for class_name in self.tag_classes}
        
        # Parameters for weight calculation
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Cache for tag classifications
        self.tag_to_class = {}
        
    def add_tags_to_frequency(self, tags, count=1):
        """
        Update tag frequencies with new tags.
        
        Args:
            tags (list): List of tags from an image
            count (int): How many times to count these tags (default: 1)
        """
        for tag in tags:
            # Find which class this tag belongs to (if not already cached)
            if tag not in self.tag_to_class:
                for class_name, tag_set in self.tag_classes.items():
                    if tag in tag_set:
                        self.tag_to_class[tag] = class_name
                        break
                else:
                    continue  # Skip tags that don't belong to any class
            
            # Update frequencies
            class_name = self.tag_to_class.get(tag)
            if class_name:
                self.tag_frequencies[class_name][tag] = (
                    self.tag_frequencies[class_name].get(tag, 0) + count
                )
                self.class_total_counts[class_name] += count

    def set_class_weight(self, class_name, weight):
        """
        Set weight for a specific class.
        
        Args:
            class_name (str): Name of the class
            weight (float): Weight value
        """
        if class_name in self.class_weights:
            self.class_weights[class_name] = weight

    def calculate_tag_weights(self, tags):
        try:
            # Convert nested lists to tuples for hashing
            if isinstance(tags, list):
                # Handle nested lists by converting inner lists to tuples
                tags = tuple(tuple(t) if isinstance(t, list) else t for t in tags)
            
            weights = []
            for class_name in self.tag_classes:
                try:
                    class_tags = set(self.tag_classes[class_name])
                    tag_set = set(tags) if isinstance(tags, (list, tuple)) else {tags}
                    
                    class_intersection = class_tags.intersection(tag_set)
                    weight = self.class_weights.get(class_name, 1.0)
                    weights.append(weight if class_intersection else 1.0)
                    
                except Exception as class_error:
                    logger.error(f"Error processing class {class_name}: {str(class_error)}")
                    logger.error(f"Class traceback: {traceback.format_exc()}")
                    weights.append(1.0)
            
            # Calculate final weight and clamp between min and max
            final_weight = torch.tensor(weights).mean()
            final_weight = torch.clamp(final_weight, self.min_weight, self.max_weight)
            
            return final_weight
            
        except Exception as e:
            logger.error(f"Tag weight calculation failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return torch.tensor(1.0)

    def update_training_loss(self, loss, tags):
        """
        Apply tag-based weighting to the training loss.
        
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
            logger.error(f"Loss update traceback: {traceback.format_exc()}")
            return loss  # Return original loss on error
    