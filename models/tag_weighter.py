from typing import List, Tuple
import torch
from utils.error_handling import error_handler
import re
import numpy as np


class TagCategory:
    """Tag categories with importance weights"""
    SUBJECT = "subject"
    STYLE = "style"
    QUALITY = "quality"
    COMPOSITION = "composition"
    COLOR = "color"
    LIGHTING = "lighting"
    TECHNICAL = "technical"
    MEDIUM = "medium"
    ARTIST = "artist"

class OptimizedTagClassifier:
    """Memory and compute optimized tag classifier"""
    def __init__(self):
        # Pre-compile all patterns into a single regex per category for faster matching
        self.category_patterns = {
            TagCategory.SUBJECT: (
                r"(person|man|woman|girl|boy|child|people|"
                r"landscape|nature|city|building|"
                r"animal|cat|dog|bird|"
                r"object|item|thing)",
                1.2
            ),
            TagCategory.STYLE: (
                r"(realistic|photorealistic|abstract|"
                r"anime|cartoon|digital art|"
                r"painting|sketch|drawing)",
                1.1
            ),
            TagCategory.QUALITY: (
                r"(high quality|masterpiece|best quality|"
                r"detailed|intricate|sharp|"
                r"professional|award winning)",
                0.9
            ),
            TagCategory.COMPOSITION: (
                r"(portrait|close-up|wide shot|"
                r"symmetrical|balanced|centered|"
                r"dynamic|action|motion)",
                1.0
            ),
            TagCategory.COLOR: (
                r"(colorful|vibrant|monochrome|"
                r"red|blue|green|yellow|purple|pink|"
                r"dark|light|bright|muted)",
                0.8
            ),
            TagCategory.LIGHTING: (
                r"(sunlight|natural light|artificial light|"
                r"dramatic lighting|soft lighting|"
                r"shadow|highlight|contrast)",
                1.0
            ),
            TagCategory.TECHNICAL: (
                r"(8k|4k|uhd|hdr|"
                r"raw photo|dslr|bokeh|"
                r"lens|camera|settings)",
                0.7
            ),
            TagCategory.MEDIUM: (
                r"(photograph|digital|traditional|"
                r"oil|watercolor|acrylic|"
                r"pencil|charcoal|ink)",
                0.9
            ),
            TagCategory.ARTIST: (
                r"(by \w+|style of \w+|inspired by|"
                r"artist:|photographer:|"
                r"school|movement|period)",
                0.8
            )
        }
        
        # Compile patterns once
        self.compiled_patterns = {
            category: (re.compile(pattern, re.IGNORECASE), importance)
            for category, (pattern, importance) in self.category_patterns.items()
        }
        
        # Cache for tag classifications
        self.classification_cache = {}
    
    def classify_tag(self, tag: str) -> Tuple[str, float]:
        """Classify a tag with caching for performance"""
        # Check cache first
        if tag in self.classification_cache:
            return self.classification_cache[tag]
        
        # Match against compiled patterns
        for category, (pattern, importance) in self.compiled_patterns.items():
            if pattern.search(tag):
                result = (category, importance)
                self.classification_cache[tag] = result
                return result
        
        # Cache and return default for unknown tags
        result = ("other", 1.0)
        self.classification_cache[tag] = result
        return result

class FastTagWeighter:
    """Performance optimized tag weighting system"""
    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        default_weight: float = 1.0,
        smoothing_factor: float = 0.1,
        rarity_scale: float = 0.5,
        cache_size: int = 10000
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.default_weight = default_weight
        self.smoothing_factor = smoothing_factor
        self.rarity_scale = rarity_scale
        
        # Use numpy arrays for faster computation
        self.tag_counts = np.zeros(cache_size, dtype=np.int32)
        self.category_counts = np.zeros(len(TagCategory.__dict__) - 2, dtype=np.int32)
        self.total_samples = 0
        
        # Fast lookup dictionaries
        self.tag_to_idx = {}
        self.next_tag_idx = 0
        self.category_to_idx = {
            category: idx for idx, category in enumerate(
                [attr for attr in dir(TagCategory) if not attr.startswith('_')]
            )
        }
        
        # Initialize classifier
        self.classifier = OptimizedTagClassifier()
        
        # Use numpy arrays for moving averages
        self.tag_moving_avg = np.ones(cache_size, dtype=np.float32) * default_weight
        self.category_moving_avg = np.ones(len(self.category_to_idx), dtype=np.float32) * default_weight
        
        # Weight computation cache
        self.weight_cache = {}
        self.weight_cache_size = 1000
        self.weight_cache_hits = 0
        self.weight_cache_misses = 0
    
    def _get_tag_idx(self, tag: str) -> int:
        """Get or create index for tag"""
        if tag not in self.tag_to_idx:
            if self.next_tag_idx >= len(self.tag_counts):
                # Double array size if needed
                self.tag_counts = np.pad(self.tag_counts, (0, len(self.tag_counts)))
                self.tag_moving_avg = np.pad(
                    self.tag_moving_avg, 
                    (0, len(self.tag_moving_avg)),
                    constant_values=self.default_weight
                )
            self.tag_to_idx[tag] = self.next_tag_idx
            self.next_tag_idx += 1
        return self.tag_to_idx[tag]
    
    @torch.jit.script
    def _compute_weights_fast(
        self,
        tag_freqs: torch.Tensor,
        category_freqs: torch.Tensor,
        base_importance: torch.Tensor,
        tag_counts: torch.Tensor,
        category_counts: torch.Tensor,
        rarity_scale: float,
        smoothing_factor: float
    ) -> torch.Tensor:
        """JIT-compiled weight computation"""
        # Calculate frequency weights
        tag_weights = 1.0 / (tag_freqs + smoothing_factor)
        category_weights = 1.0 / (category_freqs + smoothing_factor)
        
        # Calculate rarity bonus
        rarity_bonus = 1.0 + rarity_scale * (1.0 - tag_counts / category_counts.unsqueeze(1))
        
        # Combine weights
        combined_weights = (
            base_importance * 
            (tag_weights * 0.6 + category_weights * 0.4) * 
            rarity_bonus
        )
        
        return combined_weights
    
    def update_frequencies(self, tags: List[str]):
        """Vectorized frequency update"""
        self.total_samples += 1
        
        # Convert tags to indices
        tag_indices = np.array([self._get_tag_idx(tag) for tag in tags])
        
        # Get categories and importance values
        categories = []
        importance_values = []
        for tag in tags:
            category, importance = self.classifier.classify_tag(tag)
            categories.append(self.category_to_idx[category])
            importance_values.append(importance)
        
        category_indices = np.array(categories)
        
        # Vectorized updates
        np.add.at(self.tag_counts, tag_indices, 1)
        np.add.at(self.category_counts, category_indices, 1)
        
        # Update moving averages in batches
        tag_freqs = self.tag_counts[tag_indices] / self.total_samples
        category_freqs = self.category_counts[category_indices] / self.total_samples
        
        current_tag_weights = 1.0 / (tag_freqs + self.smoothing_factor)
        current_category_weights = 1.0 / (category_freqs + self.smoothing_factor)
        
        # Vectorized moving average update
        self.tag_moving_avg[tag_indices] = (
            self.tag_moving_avg[tag_indices] * (1 - self.smoothing_factor) +
            current_tag_weights * self.smoothing_factor
        )
        
        np.add.at(
            self.category_moving_avg,
            category_indices,
            (current_category_weights - self.category_moving_avg[category_indices]) * self.smoothing_factor
        )
        
        # Clear weight cache when frequencies update
        if len(self.weight_cache) > self.weight_cache_size:
            self.weight_cache.clear()
    
    def get_tag_weight(self, tag: str) -> float:
        """Get cached or compute tag weight"""
        # Check cache
        if tag in self.weight_cache:
            self.weight_cache_hits += 1
            return self.weight_cache[tag]
        
        self.weight_cache_misses += 1
        
        # Get tag index and category
        tag_idx = self._get_tag_idx(tag)
        category, importance = self.classifier.classify_tag(tag)
        category_idx = self.category_to_idx[category]
        
        # Convert to tensors for fast computation
        tag_freq = torch.tensor(self.tag_counts[tag_idx] / self.total_samples)
        category_freq = torch.tensor(self.category_counts[category_idx] / self.total_samples)
        
        # Compute weight using JIT-compiled function
        weight = self._compute_weights_fast(
            tag_freq.unsqueeze(0),
            category_freq.unsqueeze(0),
            torch.tensor([importance]),
            torch.tensor([self.tag_counts[tag_idx]]),
            torch.tensor([self.category_counts[category_idx]]),
            self.rarity_scale,
            self.smoothing_factor
        )[0].item()
        
        # Cache result
        self.weight_cache[tag] = weight
        
        # Clear cache if too large
        if len(self.weight_cache) > self.weight_cache_size:
            self.weight_cache.clear()
        
        return weight
    
    @torch.jit.script
    def _compute_combined_weight(self, weights: torch.Tensor) -> float:
        """JIT-compiled weight combination"""
        log_weights = torch.log(weights)
        return torch.exp(log_weights.mean()).item()
    
    def get_weights(self, tags: List[str]) -> float:
        """Get combined weights using vectorized operations"""
        if not tags:
            return self.default_weight
        
        # Convert to tensor for fast computation
        weights = torch.tensor([self.get_tag_weight(tag) for tag in tags])
        return self._compute_combined_weight(weights)
    
    def print_stats(self):
        """Print cache performance statistics"""
        total = self.weight_cache_hits + self.weight_cache_misses
        hit_rate = self.weight_cache_hits / total if total > 0 else 0
        print(f"Weight cache hit rate: {hit_rate:.2%}")
        print(f"Unique tags: {self.next_tag_idx}")
        print(f"Cache size: {len(self.weight_cache)}")

def parse_tags(caption: str) -> List[str]:
    """Enhanced tag parsing with better handling of special cases"""
    # Convert to lowercase and split on commas
    parts = caption.lower().split(',')
    
    # Clean and normalize tags
    tags = []
    for part in parts:
        # Basic cleaning
        tag = part.strip()
        
        # Skip empty tags
        if not tag:
            continue
            
        # Handle special cases
        if ':' in tag:  # Handle key:value pairs
            key, value = tag.split(':', 1)
            tags.extend([key.strip(), value.strip()])
        elif 'style of' in tag:  # Handle artist styles
            artist = tag.replace('style of', '').strip()
            tags.extend(['style', artist])
        elif tag.startswith(('by ', 'artist ')):  # Handle artist attribution
            artist = tag.replace('by', '').replace('artist', '').strip()
            tags.extend(['artist', artist])
        else:
            tags.append(tag)
    
    return tags