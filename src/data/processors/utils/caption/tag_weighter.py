import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from pathlib import Path
import torch
import numpy as np
from src.config.config import TagWeighterConfig
logger = logging.getLogger(__name__)

def parse_tags(text: str) -> Dict[str, List[str]]:
    """Parse text into tag dictionary by category."""
    tag_dict = {
        'character': [],
        'style': [],
        'quality': [],
        'artist': []
    }
    
    # Simple parsing - can be enhanced later
    lines = text.strip().split('\n')
    for line in lines:
        if ':' in line:
            category, tags = line.split(':', 1)
            category = category.strip().lower()
            if category in tag_dict:
                tag_dict[category].extend(
                    [t.strip() for t in tags.split(',') if t.strip()]
                )
    
    return tag_dict

class TagWeighter:
    """Handles tag weighting and frequency tracking for captions."""
    
    def __init__(
        self,
        config: Optional[TagWeighterConfig] = None,
        initial_weights: Optional[Dict[str, Dict[str, float]]] = None,
        weight_ranges: Optional[Dict[str, tuple]] = None,
        save_path: Optional[str] = None
    ):
        """Initialize tag weighter with configuration."""
        # Initialize from config if provided
        if config:
            self.tag_weights = initial_weights or {}
            self.weight_ranges = weight_ranges or {
                'character': (config.min_weight, config.max_weight),
                'style': (config.min_weight, config.max_weight),
                'quality': (config.min_weight, config.max_weight),
                'artist': (config.min_weight, config.max_weight)
            }
            self.default_weight = config.default_weight
            self.smoothing_factor = config.smoothing_factor
        else:
            # Use default initialization
            self.tag_weights = initial_weights or {}
            self.weight_ranges = weight_ranges or {
                'character': (0.8, 1.2),
                'style': (0.7, 1.3),
                'quality': (0.6, 1.4),
                'artist': (0.5, 1.5)
            }
            self.default_weight = 1.0
            self.smoothing_factor = 1e-4

        # Initialize frequency counters
        self.tag_frequencies = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0
        self.save_path = Path(save_path) if save_path else None
        
        logger.info(
            f"Initialized TagWeighter:\n"
            f"- Weight ranges: {self.weight_ranges}\n"
            f"- Initial weights: {len(self.tag_weights)} categories\n"
            f"- Save path: {self.save_path or 'None'}"
        )

    def update_frequencies(self, tag_class: str, tag: str) -> None:
        """Update frequency counters for a tag."""
        self.tag_frequencies[tag_class][tag] += 1
        self.total_samples += 1

    def get_tag_weight(self, tag_class: str, tag: str) -> float:
        """Get weight for a tag, calculating if needed."""
        if tag_class not in self.tag_weights or tag not in self.tag_weights[tag_class]:
            self._calculate_weight(tag_class, tag)
        return self.tag_weights.get(tag_class, {}).get(tag, 1.0)

    def _calculate_weight(self, tag_class: str, tag: str) -> None:
        """Calculate weight for a tag based on frequency and class."""
        try:
            if tag_class not in self.tag_weights:
                self.tag_weights[tag_class] = {}
                
            # Get frequency
            freq = self.tag_frequencies[tag_class][tag]
            if freq == 0 or self.total_samples == 0:
                self.tag_weights[tag_class][tag] = 1.0
                return
                
            # Calculate relative frequency
            rel_freq = freq / self.total_samples
            
            # Get weight range for class
            min_weight, max_weight = self.weight_ranges.get(
                tag_class, 
                (0.5, 1.5)  # Default range
            )
            
            # Calculate weight using inverse frequency
            # More frequent tags get lower weights
            weight_range = max_weight - min_weight
            weight = max_weight - (weight_range * rel_freq)
            
            # Apply SDXL-style conditioning strength
            # This helps balance the tag weights with SDXL's text conditioning
            if tag_class == 'quality':
                weight *= 1.2  # Boost quality tags slightly
            elif tag_class == 'character':
                weight *= 1.1  # Slight boost for character tags
                
            # Store calculated weight
            self.tag_weights[tag_class][tag] = float(weight)
            
        except Exception as e:
            logger.error(f"Error calculating weight for {tag_class}:{tag}: {e}")
            self.tag_weights[tag_class][tag] = 1.0

    def get_weights_tensor(self, tag_dict: Dict[str, List[str]]) -> torch.Tensor:
        """Convert tag dictionary to weight tensor for SDXL conditioning."""
        weights = []
        
        for tag_class, tags in tag_dict.items():
            for tag in tags:
                weight = self.get_tag_weight(tag_class, tag)
                weights.append(weight)
                
        if not weights:
            return torch.ones(1, dtype=torch.float32)
            
        return torch.tensor(weights, dtype=torch.float32)

    def save_weights(self) -> None:
        """Save current weights to file."""
        if self.save_path:
            try:
                save_data = {
                    'weights': self.tag_weights,
                    'frequencies': {
                        k: dict(v) for k, v in self.tag_frequencies.items()
                    },
                    'total_samples': self.total_samples,
                    'weight_ranges': self.weight_ranges
                }
                
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.save_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                    
                logger.info(f"Saved tag weights to {self.save_path}")
                
            except Exception as e:
                logger.error(f"Error saving weights: {e}")

    @classmethod
    def load(cls, path: str) -> "TagWeighter":
        """Load tag weighter from saved file."""
        try:
            with open(path) as f:
                data = json.load(f)
                
            weighter = cls(
                initial_weights=data['weights'],
                weight_ranges=data['weight_ranges'],
                save_path=path
            )
            
            # Restore frequencies
            weighter.tag_frequencies.update(
                {k: defaultdict(int, v) for k, v in data['frequencies'].items()}
            )
            weighter.total_samples = data['total_samples']
            
            logger.info(
                f"Loaded TagWeighter from {path}:\n"
                f"- Categories: {len(data['weights'])}\n"
                f"- Total samples: {data['total_samples']:,}"
            )
            
            return weighter
            
        except Exception as e:
            logger.error(f"Error loading weights from {path}: {e}")
            return cls()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tag weights and frequencies."""
        stats = {
            'total_samples': self.total_samples,
            'categories': {},
            'weight_ranges': self.weight_ranges
        }
        
        for tag_class in self.tag_frequencies:
            freqs = self.tag_frequencies[tag_class]
            weights = self.tag_weights.get(tag_class, {})
            
            stats['categories'][tag_class] = {
                'unique_tags': len(freqs),
                'total_occurrences': sum(freqs.values()),
                'avg_weight': np.mean(list(weights.values())) if weights else 1.0,
                'min_weight': min(weights.values()) if weights else 1.0,
                'max_weight': max(weights.values()) if weights else 1.0
            }
            
        return stats


