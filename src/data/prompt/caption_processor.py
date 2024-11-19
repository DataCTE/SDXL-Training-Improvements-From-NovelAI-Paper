import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
from threading import Lock
import random

logger = logging.getLogger(__name__)

@dataclass
class TagStats:
    """Thread-safe container for tag statistics tracking."""
    frequencies: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total_occurrences: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    _lock: Lock = field(default_factory=Lock)
    
    def update(self, tag: str) -> None:
        """Thread-safe update of tag frequencies."""
        with self._lock:
            self.total_occurrences[tag] += 1
    
    def update_cooccurrence(self, tag1: str, tag2: str) -> None:
        """Thread-safe update of tag co-occurrence."""
        with self._lock:
            self.cooccurrence[tag1][tag2] += 1
            self.cooccurrence[tag2][tag1] += 1
    
    def get_rarity(self, tag: str, default: float = 1.0) -> float:
        """Calculate tag rarity based on inverse frequency."""
        with self._lock:
            total_freq = self.total_occurrences.get(tag, 0)
            if total_freq == 0:
                return default
            return default / (math.log1p(total_freq) + 1.0)

class CaptionProcessor:
    """Handles caption parsing, tag processing, and weight computation."""
    
    MIN_WEIGHT: float = 0.1
    MAX_WEIGHT: float = 3.0
    MAX_LENGTH: int = 77  # SDXL's max token length
    
    def __init__(self, 
                 token_dropout_rate: float = 0.1,
                 caption_dropout_rate: float = 0.1,
                 rarity_factor: float = 0.9,
                 emphasis_factor: float = 1.2,
                 num_workers: int = 4):
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        self.rarity_factor = rarity_factor
        self.emphasis_factor = emphasis_factor
        self.num_workers = num_workers
        self.tag_stats = TagStats()
        self.weight_cache: Dict[str, float] = {}
        
    def process_caption(self, caption: str, training: bool = True) -> Tuple[List[str], List[float]]:
        """Process caption into tags and weights with length normalization."""
        tags = self.parse_tags(caption)
        if not tags or (training and random.random() < self.caption_dropout_rate):
            # Return padded empty tags
            return [''] * self.MAX_LENGTH, [0.0] * self.MAX_LENGTH
            
        # Apply token dropout during training
        if training and self.token_dropout_rate > 0:
            tags = [tag for tag in tags if random.random() > self.token_dropout_rate]
        
        # Update statistics
        for tag in tags:
            self.tag_stats.update(tag)
        
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                self.tag_stats.update_cooccurrence(tag1, tag2)
        
        # Get weights
        weights = [self.get_tag_weight(tag) for tag in tags]
        
        # Pad or truncate to MAX_LENGTH
        if len(tags) < self.MAX_LENGTH:
            padding_length = self.MAX_LENGTH - len(tags)
            tags.extend([''] * padding_length)
            weights.extend([0.0] * padding_length)
        else:
            tags = tags[:self.MAX_LENGTH]
            weights = weights[:self.MAX_LENGTH]
            
        return tags, weights
    
    def parse_tags(self, caption: str) -> List[str]:
        """Parse caption into individual tags."""
        if not caption:
            return []
        return [t.strip() for t in caption.split(',') if t.strip()]
    
    def get_tag_weight(self, tag: str) -> float:
        """Compute the weight for a tag based on rarity and emphasis."""
        # Check cache first
        if tag in self.weight_cache:
            return self.weight_cache[tag]
            
        # Get base components
        rarity = self.tag_stats.get_rarity(tag)
        
        # Compute final weight
        weight = 1.0 * (1.0 + (rarity * self.rarity_factor))
        
        # Apply emphasis if present
        emphasis_level = self._get_emphasis_level(tag)
        if emphasis_level > 0:
            weight *= (self.emphasis_factor ** emphasis_level)
            
        # Clamp to valid range
        weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weight))
        
        # Cache and return
        self.weight_cache[tag] = weight
        return weight
        
    def _get_emphasis_level(self, tag: str) -> int:
        """Get the emphasis level based on curly brace markers."""
        emphasis = 0
        while tag.startswith('{') and tag.endswith('}'):
            emphasis += 1
            tag = tag[1:-1].strip()
        return emphasis
    
    def load_caption(self, image_path: str) -> str:
        """Load caption from corresponding text file."""
        try:
            caption_path = Path(image_path).with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                logger.warning(f"Caption file not found for {image_path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading caption for {image_path}: {str(e)}")
            return ""

def load_captions(image_paths: List[str]) -> Dict[str, str]:
    """Load captions for a list of image paths."""
    processor = CaptionProcessor()
    return {path: processor.load_caption(path) for path in image_paths}
