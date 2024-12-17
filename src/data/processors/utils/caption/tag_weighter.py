from typing import List, Dict, Optional
import torch
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import logging
from src.data.processors.utils.caption.text_embedder import TextEmbedder
from src.data.processors.utils.system_utils import calculate_optimal_batch_size

logger = logging.getLogger(__name__)

@dataclass
class TagWeightingConfig:
    """Configuration for tag weighting system."""
    min_weight: float = 0.1
    max_weight: float = 2.0
    default_weight: float = 1.0
    enabled: bool = True
    update_frequency: int = 1000
    smoothing_factor: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_weight <= 0:
            raise ValueError("min_weight must be positive")
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be greater than min_weight")
        if not 0 <= self.smoothing_factor <= 1:
            raise ValueError("smoothing_factor must be between 0 and 1")
        if self.update_frequency < 1:
            raise ValueError("update_frequency must be positive")

def parse_tags(text: str) -> List[str]:
    """Parse text into individual word tags.
    
    Args:
        text: Input text/prompt
        
    Returns:
        List of individual word tags
    """
    # Split text into words and clean each word
    words = text.lower().split()
    
    # Clean and filter words
    tags = []
    for word in words:
        # Remove punctuation and special characters
        word = ''.join(c for c in word if c.isalnum() or c == '-')
        if word and len(word) > 1:  # Keep words with at least 2 characters
            tags.append(word)
            
    return tags

class TagWeighter:
    def __init__(self, config: TagWeightingConfig, text_embedder: Optional[TextEmbedder] = None):
        """Initialize with optional text embedder for advanced features."""
        self.config = config
        self.text_embedder = text_embedder
        
        # Initialize tracking state
        self.tag_counts: Dict[str, int] = {}
        self.total_count: int = 0
        self.tag_weights: Dict[str, float] = {}
        self.previous_weights: Dict[str, float] = {}
        self.steps_since_update: int = 0
        
        # Runtime statistics
        self.updates_performed: int = 0
        self.total_samples_processed: int = 0
        
        logger.info(f"Initialized TagWeighter with config: {config}")

    def update_frequencies(self, text: str) -> None:
        """Update tag frequency counters from full text.
        
        Args:
            text: Full text/prompt to process
        """
        if not self.config.enabled:
            return
            
        try:
            # Get tags from full text
            tags = parse_tags(text)
            
            # Update counts for each tag
            for tag in tags:
                if tag not in self.tag_counts:
                    self.tag_counts[tag] = 0
                self.tag_counts[tag] += 1
                self.total_count += 1
            
            self.steps_since_update += 1
            self.total_samples_processed += 1
            
            # Check if we should recompute weights
            if self.steps_since_update >= self.config.update_frequency:
                self.compute_weights()
                self.steps_since_update = 0
                self.updates_performed += 1
                
        except Exception as e:
            logger.error(f"Error updating frequencies: {e}")
            raise

    def compute_weights(self) -> None:
        """Compute weights for all seen tags with smoothing."""
        if not self.config.enabled or not self.total_count:
            return
            
        try:
            # Calculate average frequency
            num_tags = len(self.tag_counts)
            if num_tags == 0:
                return
                
            avg_freq = self.total_count / num_tags
            
            # Store current weights for smoothing
            self.previous_weights = self.tag_weights.copy()
            
            # Compute new weights efficiently using numpy
            frequencies = np.array(list(self.tag_counts.values()))
            raw_weights = avg_freq / frequencies
            
            # Clip weights to bounds
            clipped_weights = np.clip(
                raw_weights, 
                self.config.min_weight,
                self.config.max_weight
            )
            
            # Update weights with smoothing
            for tag, new_weight in zip(self.tag_counts.keys(), clipped_weights):
                if tag in self.previous_weights:
                    smoothed_weight = (
                        self.config.smoothing_factor * new_weight +
                        (1 - self.config.smoothing_factor) * self.previous_weights[tag]
                    )
                    self.tag_weights[tag] = float(smoothed_weight)
                else:
                    self.tag_weights[tag] = float(new_weight)
                    
        except Exception as e:
            logger.error(f"Error computing weights: {e}")
            raise

    def get_weight(self, tags: List[str]) -> float:
        """Get weight for tags with embeddings-based similarity if available."""
        base_weight = self._calculate_base_weight(tags)
        
        if self.text_embedder is not None:
            # Use text embedder to calculate similarity-based adjustment
            try:
                embeddings = self.text_embedder.encode_prompt_list(tags)
                similarity_factor = self._calculate_similarity_factor(embeddings)
                return base_weight * similarity_factor
            except Exception as e:
                logger.warning(f"Error calculating embedding similarity: {e}")
                
        return base_weight

    def save_weights(self, path: str) -> None:
        """Save current state to file.
        
        Args:
            path: Path to save state to
        """
        if not self.config.enabled:
            return
            
        path = Path(path)
        try:
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'tag_weights': self.tag_weights,
                'tag_counts': self.tag_counts,
                'total_count': self.total_count,
                'steps_since_update': self.steps_since_update,
                'updates_performed': self.updates_performed,
                'total_samples_processed': self.total_samples_processed,
                'config': vars(self.config)
            }
            
            # Save atomically
            temp_path = path.with_suffix('.tmp')
            torch.save(state, temp_path)
            temp_path.rename(path)
            
            logger.info(f"Saved weights to {path}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            raise

    def load_weights(self, path: str) -> None:
        """Load state from file.
        
        Args:
            path: Path to load state from
        """
        if not self.config.enabled:
            return
            
        try:
            state = torch.load(path)
            
            # Restore state
            self.tag_weights = state['tag_weights']
            self.tag_counts = state['tag_counts']
            self.total_count = state['total_count']
            self.steps_since_update = state.get('steps_since_update', 0)
            self.updates_performed = state.get('updates_performed', 0)
            self.total_samples_processed = state.get('total_samples_processed', 0)
            
            # Keep previous weights for smoothing
            self.previous_weights = self.tag_weights.copy()
            
            logger.info(f"Loaded weights from {path}")
            logger.info(f"Restored {len(self.tag_weights)} tag weights")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise
            
    def get_stats(self) -> Dict:
        """Get current statistics about the weighting system."""
        return {
            'total_tags': len(self.tag_counts),
            'total_samples': self.total_samples_processed,
            'updates_performed': self.updates_performed,
            'unique_tags_seen': len(self.tag_weights),
            'min_weight': min(self.tag_weights.values()) if self.tag_weights else None,
            'max_weight': max(self.tag_weights.values()) if self.tag_weights else None,
            'avg_weight': np.mean(list(self.tag_weights.values())) if self.tag_weights else None
        }

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            if self.device.type == "cuda":
                return calculate_optimal_batch_size(
                    device=self.device,
                    min_batch_size=1,
                    max_batch_size=32,  # Cap at 32 as per original implementation
                    target_memory_usage=self.max_memory_usage,
                    growth_factor=0.3
                )
            else:
                return 8  # Default CPU batch size
        except Exception as e:
            logger.warning(f"Error calculating batch size: {e}, using default")
            return 8


