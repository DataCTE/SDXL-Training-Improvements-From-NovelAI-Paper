import torch
import logging
import traceback
from typing import Dict, Set, Optional, List, Tuple
from functools import lru_cache
from collections import defaultdict
import numpy as np
import re
import math
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

def _default_zero():
    """Default function to return 0 for defaultdict"""
    return 0

class TagBasedLossWeighter:
    min_weight = 0.1
    max_weight = 3.0
    
    @staticmethod
    def parse_tags(caption: str) -> Tuple[List[str], Dict[str, any]]:
        """
        Static method to parse caption into tags and special tags with improved Midjourney compatibility.
        
        Args:
            caption (str): Raw caption text
            
        Returns:
            Tuple[List[str], Dict]: Regular tags and special tag parameters
        """
        if not caption:
            return [], {}
            
        tags = []
        special_tags = {}
        
        # Split and process tags
        raw_tags = [t.strip() for t in caption.split(',')]
        
        # Check for MJ-specific tags
        has_mj_tags = any('niji' in t.lower() or t.strip() in ['4', '5', '6'] for t in raw_tags)
        
        if has_mj_tags:
            # Handle anime style/niji at start
            if raw_tags and ('anime style' in raw_tags[0].lower() or 'niji' in raw_tags[0].lower()):
                special_tags['niji'] = True
                raw_tags = raw_tags[1:]
                
            # Handle version number
            if raw_tags and raw_tags[-1].strip() in ['4', '5', '6']:
                raw_tags = raw_tags[:-1]
                tags.append('masterpiece')
        
        for tag in raw_tags:
            tag = tag.lower().strip()
            
            if not tag:
                continue
            
            # Handle compound tags with weights
            if '::' in tag:
                parts = tag.split('::')
                tag = parts[0].strip()
                try:
                    weight = float(parts[1])
                    special_tags[f'{tag}_weight'] = weight
                except:
                    pass

            # Handle style references
            if 'sref' in tag:
                refs = re.findall(r'[a-f0-9]{8}|https?://[^\s>]+', tag)
                if refs:
                    special_tags['sref'] = refs
                    continue

            # Handle MJ style parameters
            if has_mj_tags:
                is_param = False
                for param in ['stylize', 'chaos', 'sw', 'sv']:
                    if param in tag:
                        try:
                            value = float(re.search(r'[\d.]+', tag).group())
                            special_tags[param] = value
                            is_param = True
                        except:
                            continue
                if is_param:
                    continue

            # Clean up tag
            if tag.startswith(('a ', 'an ', 'the ')):
                tag = ' '.join(tag.split()[1:])
            
            if tag:
                tags.append(tag)
        
        return tags, special_tags

    @staticmethod
    def calculate_static_weights(tags: List[str], special_tags: Dict[str, any] = None) -> float:
        """
        Static method to calculate basic weights without instance-specific data.
        
        Args:
            tags (List[str]): List of tags
            special_tags (Dict): Special tag parameters
            
        Returns:
            float: Basic weight value
        """
        if special_tags is None:
            special_tags = {}
            
        base_weight = 1.0
        
        # Apply basic modifiers
        if 'masterpiece' in tags:
            base_weight *= 1.3
        if special_tags.get('niji', False):
            base_weight *= 1.2
        if 'stylize' in special_tags:
            stylize_value = special_tags['stylize']
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if 'chaos' in special_tags:
            chaos_value = special_tags['chaos']
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor
            
        # Clamp between min and max
        return max(TagBasedLossWeighter.min_weight, 
                  min(TagBasedLossWeighter.max_weight, base_weight))

    @staticmethod
    def format_caption(caption: str) -> str:
        """
        Static method to format caption text with standardized formatting.
        
        Args:
            caption (str): Raw caption text
            
        Returns:
            str: Formatted caption text
        """
        if not caption:
            return ""
            
        try:
            # Split into tags
            tags = [t.strip() for t in caption.split(',')]
            
            # Remove empty tags
            tags = [t for t in tags if t]
            
            # Basic cleanup for each tag
            formatted_tags = []
            for tag in tags:
                # Convert to lowercase
                tag = tag.lower()
                
                # Remove extra spaces
                tag = ' '.join(tag.split())
                
                # Remove articles from start
                if tag.startswith(('a ', 'an ', 'the ')):
                    tag = ' '.join(tag.split()[1:])
                
                # Handle special formatting for quality tags
                if any(q in tag for q in ['masterpiece', 'best quality', 'high quality']):
                    formatted_tags.insert(0, tag)  # Move to front
                    continue
                    
                # Handle special formatting for negative tags
                if tag.startswith(('no ', 'bad ', 'worst ')):
                    if not any(neg in tag for neg in ['negative space', 'negative prompt']):
                        tag = tag.replace('no ', '').replace('bad ', '').replace('worst ', '')
                        tag = f"lowquality {tag}"
                
                formatted_tags.append(tag)
            
            # Join tags with standardized separator
            return ', '.join(formatted_tags)
            
        except Exception as e:
            logger.error(f"Caption formatting error: {str(e)}")
            return caption  # Return original if formatting fails

    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
        cache_size: int = 1024,
        no_cache: bool = False,
        emphasis_factor: float = 1.1,
        rarity_factor: float = 0.9,
        quality_bonus: float = 0.2,
        character_emphasis: float = 1.2,
        num_workers: Optional[int] = None
    ):
        """
        Initialize the tag-based loss weighting system with NovelAI improvements.
        
        Args:
            tag_classes (dict): Dictionary mapping tag class names to lists of tags
            min_weight (float): Minimum weight multiplier for any image
            max_weight (float): Maximum weight multiplier for any image
            cache_size (int): Size of LRU cache for tag classification and weight calculation
            no_cache (bool): Flag to disable caching
            emphasis_factor (float): Multiplier for emphasized tags
            rarity_factor (float): Multiplier for rare tags
            quality_bonus (float): Additional weight for high-quality images
            character_emphasis (float): Special multiplier for character tags
            num_workers (int, optional): Number of worker threads for parallel processing
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
        self.tag_frequencies = defaultdict(lambda: defaultdict(_default_zero))
        self.class_total_counts = defaultdict(_default_zero)
        self.tag_cooccurrence = defaultdict(lambda: defaultdict(_default_zero))
        
        # Precompute mappings
        self._initialize_mappings()
        
        # Caching parameters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cache_size = cache_size
        
        # Modify caching behavior based on no_cache flag
        self.no_cache = no_cache
        if no_cache:
            # Don't use caching when no_cache is True
            self.calculate_tag_weights = self._calculate_tag_weights
        else:
            # Use LRU cache when caching is enabled
            self.calculate_tag_weights = lru_cache(maxsize=cache_size)(self._calculate_tag_weights)
        
        # Initialize caches
        self._tag_rarity_scores = {}
        self._tag_importance_scores = {}
        
        # Initialize worker pool
        self.num_workers = num_workers if num_workers is not None else min(8, os.cpu_count() or 1)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
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
        Uses parallel processing for large batches.
        
        Args:
            batch_tags (List[List[str]]): List of tag lists from a batch of images
        """
        def process_tag_list(tags):
            result = {
                'frequencies': defaultdict(lambda: defaultdict(_default_zero)),
                'class_counts': defaultdict(_default_zero),
                'cooccurrence': defaultdict(lambda: defaultdict(_default_zero))
            }
            
            # Update individual tag frequencies
            for tag in tags:
                class_name = self._classify_tag(tag)
                if class_name:
                    result['frequencies'][class_name][tag] += 1
                    result['class_counts'][class_name] += 1
            
            # Update co-occurrence matrix
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    result['cooccurrence'][tag1][tag2] += 1
                    result['cooccurrence'][tag2][tag1] += 1
                    
            return result

        # Process tags in parallel for large batches
        if len(batch_tags) > 100:  # Only parallelize for large batches
            futures = []
            chunk_size = max(1, len(batch_tags) // self.num_workers)
            
            for i in range(0, len(batch_tags), chunk_size):
                chunk = batch_tags[i:i + chunk_size]
                futures.extend([self.executor.submit(process_tag_list, tags) for tags in chunk])
            
            # Combine results
            for future in futures:
                result = future.result()
                for class_name, tags in result['frequencies'].items():
                    for tag, count in tags.items():
                        self.tag_frequencies[class_name][tag] += count
                for class_name, count in result['class_counts'].items():
                    self.class_total_counts[class_name] += count
                for tag1, coocs in result['cooccurrence'].items():
                    for tag2, count in coocs.items():
                        self.tag_cooccurrence[tag1][tag2] += count
        else:
            # Process sequentially for small batches
            for tags in batch_tags:
                result = process_tag_list(tags)
                for class_name, tags in result['frequencies'].items():
                    for tag, count in tags.items():
                        self.tag_frequencies[class_name][tag] += count
                for class_name, count in result['class_counts'].items():
                    self.class_total_counts[class_name] += count
                for tag1, coocs in result['cooccurrence'].items():
                    for tag2, count in coocs.items():
                        self.tag_cooccurrence[tag1][tag2] += count
        
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
    
    def _calculate_batch_weights(self, batch_tags: List[List[str]]) -> List[float]:
        """
        Calculate weights for a batch of tag lists in parallel.
        
        Args:
            batch_tags (List[List[str]]): List of tag lists to process
            
        Returns:
            List[float]: List of calculated weights
        """
        if len(batch_tags) > 50:  # Only parallelize for larger batches
            futures = [self.executor.submit(self._calculate_tag_weights, tuple(tags)) for tags in batch_tags]
            return [future.result() for future in futures]
        else:
            return [self._calculate_tag_weights(tuple(tags)) for tags in batch_tags]

    def calculate_weights(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Calculate weights with proper no_cache handling"""
        if self.no_cache:
            # Calculate weights directly without caching
            return self._calculate_weights_no_cache(tags, special_tags)
        else:
            # Use cached calculation
            return self._calculate_weights_cached(tags, special_tags)

    def _calculate_weights_no_cache(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Direct weight calculation without caching"""
        if special_tags is None:
            special_tags = {}
            
        weights = {}
        base_weight = 1.0
        
        # Apply modifiers directly without caching
        if 'masterpiece' in tags:
            base_weight *= 1.3
        if special_tags.get('niji', False):
            base_weight *= 1.2
        if 'stylize' in special_tags:
            stylize_value = special_tags['stylize']
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if 'chaos' in special_tags:
            chaos_value = special_tags['chaos']
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor
            
        # Calculate individual weights
        for i, tag in enumerate(tags):
            # Get tag class importance
            class_name = self._classify_tag(tag)
            class_weight = self.class_base_weights.get(class_name, 1.0)
            
            # Apply position decay
            position_weight = 1.0 - (i * 0.05)
            
            # Apply rarity bonus
            rarity_score = self._tag_rarity_scores.get(tag, 1.0)
            
            # Combine all factors
            final_weight = base_weight * class_weight * position_weight * rarity_score
            
            # Apply any explicit tag weights
            if f'{tag}_weight' in special_tags:
                final_weight *= special_tags[f'{tag}_weight']
            
            weights[tag] = max(self.min_weight, min(self.max_weight, final_weight))
            
        return weights

    def _calculate_weights_cached(self, tags: List[str], special_tags: Dict[str, any] = None) -> Dict[str, float]:
        """Cached weight calculation"""
        # Convert tags to tuple for caching
        tags_tuple = tuple(sorted(tags))
        return self.calculate_tag_weights(tags_tuple, frozenset(special_tags.items()) if special_tags else None)

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

    def __del__(self):
        """Cleanup worker pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)