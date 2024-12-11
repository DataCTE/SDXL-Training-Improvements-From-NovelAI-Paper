from typing import List
import torch

class TagWeighter:
    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        default_weight: float = 1.0
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.default_weight = default_weight
        
        self.tag_counts = {}
        self.total_count = 0
        self.tag_weights = {}

    def update_frequencies(self, tags: List[str]):
        """Update tag frequency counters"""
        for tag in tags:
            if tag not in self.tag_counts:
                self.tag_counts[tag] = 0
            self.tag_counts[tag] += 1
            self.total_count += 1
            
    def compute_weights(self):
        """Compute weights for all seen tags"""
        if not self.total_count:
            return
            
        avg_freq = self.total_count / len(self.tag_counts) if self.tag_counts else 1.0
        
        for tag, count in self.tag_counts.items():
            raw_weight = avg_freq / count
            weight = min(self.max_weight, max(self.min_weight, raw_weight))
            self.tag_weights[tag] = weight
                
    def get_weight(self, tags: List[str]) -> float:
        """Get combined weight for a set of tags"""
        if not tags:
            return self.default_weight
            
        weights = [self.tag_weights.get(tag, self.default_weight) for tag in tags]
        return torch.tensor(weights).mean().item()

def parse_tags(caption: str) -> List[str]:
    """Extract tags from caption"""
    parts = caption.lower().split(',')
    return [tag.strip() for tag in parts] 