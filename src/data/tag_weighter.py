import torch
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class TagStats:
    """Thread-safe container for tag statistics tracking."""
    
    # Track tag frequencies per class
    frequencies: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    
    # Track total counts per class
    class_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    # Track tag co-occurrence
    cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    
    # Thread safety lock
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
    
    def get_rarity(self, tag: str, default: float = 1.0) -> float:
        """Calculate tag rarity based on inverse frequency."""
        with self._lock:
            total_freq = sum(
                self.frequencies[class_name][tag]
                for class_name in self.frequencies
            )
            if total_freq == 0:
                return default
            # Use log scaling to prevent extreme values
            return default / (math.log1p(total_freq) + 1.0)


class TagBasedLossWeighter:
    """Tag-based loss weighting system following NovelAI paper specifications."""
    
    # Weight boundaries from paper
    MIN_WEIGHT: float = 0.1
    MAX_WEIGHT: float = 3.0
    
    # Weighting factors
    DEFAULT_RARITY_FACTOR: float = 0.9
    DEFAULT_EMPHASIS_FACTOR: float = 1.2
    DEFAULT_CACHE_SIZE: int = 1024
    
    # Tag categories based on paper
    TAG_CATEGORIES = {
        "quality": {
            "description": "Image quality indicators",
            "keywords": ["masterpiece", "best quality", "high quality", "highres"],
            "base_weight": 1.2
        },
        "character": {
            "description": "Character traits and features",
            "keywords": ["1girl", "1boy", "hair", "eyes", "face", "expression"],
            "base_weight": 1.3
        },
        "clothing": {
            "description": "Character attire",
            "keywords": ["dress", "shirt", "outfit", "uniform", "costume"],
            "base_weight": 1.0
        },
        "pose": {
            "description": "Character poses and actions",
            "keywords": ["standing", "sitting", "walking", "running", "pose"],
            "base_weight": 1.1
        },
        "background": {
            "description": "Scene and environment",
            "keywords": ["outdoors", "indoors", "room", "sky", "night", "day"],
            "base_weight": 0.9
        },
        "style": {
            "description": "Artistic style",
            "keywords": ["style", "colored", "monochrome", "sketch", "painting"],
            "base_weight": 1.1
        },
        "general": {
            "description": "General tags",
            "keywords": [],
            "base_weight": 1.0
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the weighting system with configuration."""
        self.config = config or {}
        
        # Initialize core parameters
        self.min_weight = float(self.config.get('min_weight', self.MIN_WEIGHT))
        self.max_weight = float(self.config.get('max_weight', self.MAX_WEIGHT))
        self.rarity_factor = float(self.config.get('rarity_factor', self.DEFAULT_RARITY_FACTOR))
        self.emphasis_factor = float(self.config.get('emphasis_factor', self.DEFAULT_EMPHASIS_FACTOR))
        self.cache_size = int(self.config.get('cache_size', self.DEFAULT_CACHE_SIZE))
        
        # Initialize statistics tracking
        self.stats = TagStats()
        
        # Initialize thread pool for parallel processing
        self.num_workers = min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info("TagBasedLossWeighter initialized with:")
        logger.info(f"- Min weight: {self.min_weight}")
        logger.info(f"- Max weight: {self.max_weight}")
        logger.info(f"- Rarity factor: {self.rarity_factor}")
        logger.info(f"- Emphasis factor: {self.emphasis_factor}")
        logger.info(f"- Cache size: {self.cache_size}")
        logger.info(f"- Workers: {self.num_workers}")
    
    def get_category_weight(self, tag: str) -> float:
        """Determine base weight from tag category."""
        for category, info in self.TAG_CATEGORIES.items():
            if any(keyword in tag.lower() for keyword in info["keywords"]):
                return info["base_weight"]
        return self.TAG_CATEGORIES["general"]["base_weight"]
    
    def calculate_emphasis_weight(self, tag: str) -> float:
        """Calculate weight based on tag emphasis markers."""
        emphasis_level = 0
        
        # Check for emphasis markers like (), [], {}, <>
        if tag.count("(") > tag.count(")"):  # Unbalanced parentheses check
            emphasis_level += 1
        elif tag.count("[") > tag.count("]"):
            emphasis_level += 2
        elif tag.count("{") > tag.count("}"):
            emphasis_level += 3
        elif tag.count("<") > tag.count(">"):
            emphasis_level += 4
            
        return 1.0 + (emphasis_level * self.emphasis_factor)
    
    def calculate_tag_weight(self, tag: str, class_name: str = "general") -> float:
        """Calculate final weight for a tag combining category, rarity and emphasis."""
        # Get base category weight
        category_weight = self.get_category_weight(tag)
        
        # Calculate rarity component
        rarity_weight = self.stats.get_rarity(tag) * self.rarity_factor
        
        # Calculate emphasis component
        emphasis_weight = self.calculate_emphasis_weight(tag)
        
        # Combine weights with proper scaling
        combined_weight = category_weight * rarity_weight * emphasis_weight
        
        # Clamp to boundaries
        final_weight = max(self.min_weight, min(self.max_weight, combined_weight))
        
        logger.debug(f"Tag weight calculation for '{tag}':")
        logger.debug(f"- Category weight: {category_weight:.3f}")
        logger.debug(f"- Rarity weight: {rarity_weight:.3f}")
        logger.debug(f"- Emphasis weight: {emphasis_weight:.3f}")
        logger.debug(f"- Final weight: {final_weight:.3f}")
        
        return final_weight
    
    def process_tag_batch(self, tags: List[str], class_name: str = "general") -> Dict[str, float]:
        """Process a batch of tags in parallel to get their weights."""
        # Update statistics for all tags
        for tag in tags:
            self.stats.update(class_name, tag)
            
        # Update co-occurrence for tag pairs
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                self.stats.update_cooccurrence(tag1, tag2)
        
        # Calculate weights in parallel
        future_to_tag = {
            self.executor.submit(self.calculate_tag_weight, tag, class_name): tag
            for tag in tags
        }
        
        # Collect results
        weights = {}
        for future in as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                weights[tag] = future.result()
            except Exception as e:
                logger.error(f"Error calculating weight for tag '{tag}': {e}")
                weights[tag] = 1.0  # Default weight on error
                
        return weights