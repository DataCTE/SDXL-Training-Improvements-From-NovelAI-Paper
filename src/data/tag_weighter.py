from typing import List
import torch
from dataclasses import dataclass

def parse_tags(caption: str) -> List[str]:
    """Extract tags from caption.
    
    Args:
        caption: Comma-separated string of tags
        
    Returns:
        List of cleaned and normalized tags
    """
    parts = caption.lower().split(',')
    tags = [tag.strip() for tag in parts]
    return tags

@dataclass
class TagWeightingConfig:
    min_weight: float
    max_weight: float
    default_weight: float
    enabled: bool
    update_frequency: int
    smoothing_factor: float

class TagWeighter:
    def __init__(self, config: TagWeightingConfig):
        self.config = config
        self.min_weight = config.min_weight
        self.max_weight = config.max_weight
        self.default_weight = config.default_weight
        self.smoothing_factor = config.smoothing_factor
        
        # Initialize tag tracking
        self.tag_counts = {}
        self.total_count = 0
        self.tag_weights = {}
        self.steps_since_update = 0
        
        # For smoothing
        self.previous_weights = {}

    def update_frequencies(self, tags: List[str]):
        """Update tag frequency counters"""
        if not self.config.enabled:
            return
            
        for tag in tags:
            if tag not in self.tag_counts:
                self.tag_counts[tag] = 0
            self.tag_counts[tag] += 1
            self.total_count += 1
        
        self.steps_since_update += 1
        
        # Check if we should recompute weights
        if self.steps_since_update >= self.config.update_frequency:
            self.compute_weights()
            self.steps_since_update = 0
            
    def compute_weights(self):
        """Compute weights for all seen tags with smoothing"""
        if not self.total_count or not self.config.enabled:
            return
            
        # Calculate average frequency
        avg_freq = self.total_count / len(self.tag_counts) if self.tag_counts else 1.0
        
        # Store current weights for smoothing
        self.previous_weights = self.tag_weights.copy()
        
        # Compute new weights for each tag
        for tag, count in self.tag_counts.items():
            raw_weight = avg_freq / count
            new_weight = min(self.max_weight, max(self.min_weight, raw_weight))
            
            # Apply exponential moving average if we have previous weights
            if tag in self.previous_weights:
                smoothed_weight = (
                    self.smoothing_factor * new_weight + 
                    (1 - self.smoothing_factor) * self.previous_weights[tag]
                )
                self.tag_weights[tag] = smoothed_weight
            else:
                self.tag_weights[tag] = new_weight
                
    def get_weight(self, tags: List[str]) -> float:
        """Get combined weight for a set of tags"""
        if not self.config.enabled or not tags:
            return self.default_weight
            
        weights = [self.tag_weights.get(tag, self.default_weight) for tag in tags]
        return torch.tensor(weights).mean().item()

    def save_weights(self, path: str):
        """Save current tag weights to file"""
        if not self.config.enabled:
            return
            
        torch.save({
            'tag_weights': self.tag_weights,
            'tag_counts': self.tag_counts,
            'total_count': self.total_count
        }, path)
        
    def load_weights(self, path: str):
        """Load tag weights from file"""
        if not self.config.enabled:
            return
            
        state = torch.load(path)
        self.tag_weights = state['tag_weights']
        self.tag_counts = state['tag_counts']
        self.total_count = state['total_count']
        self.previous_weights = self.tag_weights.copy()