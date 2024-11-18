import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Any
from concurrent.futures import ThreadPoolExecutor
import os
from functools import lru_cache
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
from threading import Lock
import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

@dataclass
class TagStats:
    """Thread-safe container for tag statistics tracking.
    
    Attributes:
        frequencies (Dict[str, Dict[str, int]]): Nested dictionary tracking tag frequencies by class.
        class_counts (Dict[str, int]): Dictionary tracking counts per class.
        cooccurrence (Dict[str, Dict[str, int]]): Nested dictionary tracking tag co-occurrences.
        total_occurrences (Dict[str, int]): Dictionary tracking total occurrences of each tag.
        _lock (Lock): Thread lock for synchronization.
    """
    frequencies: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    class_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total_occurrences: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _lock: Lock = field(default_factory=Lock)
    
    def update(self, class_name: str, tag: str) -> None:
        """Thread-safe update of tag frequencies."""
        with self._lock:
            self.frequencies[class_name][tag] += 1
            self.class_counts[class_name] += 1
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
            
    def get_common_cooccurrences(self, tag: str, min_occurrences: int = 5) -> List[str]:
        """Get tags that commonly co-occur with the given tag."""
        with self._lock:
            cooccur_dict = self.cooccurrence.get(tag, {})
            return [t for t, count in cooccur_dict.items() 
                   if count >= min_occurrences]
            
    def get_most_common_tags(self, n: int = 100) -> List[Tuple[str, int]]:
        """Get the n most common tags."""
        with self._lock:
            return sorted(
                self.total_occurrences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]

class CaptionProcessor:
    """Handles caption loading, parsing, tag processing, and weight computation.
    
    This class implements advanced caption processing techniques based on the NovelAI paper,
    including tag categorization, weight computation, and dynamic tag dropout.
    
    Attributes:
        token_dropout_rate (float): Rate at which individual tokens are dropped during training.
        caption_dropout_rate (float): Rate at which entire captions are dropped.
        rarity_factor (float): Factor controlling the impact of tag rarity on weights.
        emphasis_factor (float): Factor controlling the impact of tag emphasis markers.
        num_workers (int): Number of worker threads for parallel processing.
        tag_stats (TagStats): Thread-safe container for tag statistics.
        weight_cache (Dict): Cache for computed tag weights.
        _category_cache (Dict): Internal cache for tag categories.
        tag_categories (Dict): Mapping of tag categories and their properties.
    """
    
    # Weight boundaries from NovelAI paper
    MIN_WEIGHT: float = 0.1
    MAX_WEIGHT: float = 3.0
    
    def __init__(self, 
                 token_dropout_rate: float = 0.1,
                 caption_dropout_rate: float = 0.1,
                 rarity_factor: float = 0.9,
                 emphasis_factor: float = 1.2,
                 min_tag_freq: int = 10,
                 min_cluster_size: int = 5,
                 similarity_threshold: float = 0.3):
        """Initialize the CaptionProcessor.
        
        Args:
            token_dropout_rate: Rate at which individual tokens are dropped.
            caption_dropout_rate: Rate at which entire captions are dropped.
            rarity_factor: Controls impact of tag rarity on weights (0.0-1.0).
            emphasis_factor: Controls impact of emphasis markers (>1.0).
            min_tag_freq: Minimum frequency for a tag to be considered in clustering.
            min_cluster_size: Minimum size for a tag cluster.
            similarity_threshold: DBSCAN eps parameter for clustering.
        """
        self.token_dropout_rate = max(0.0, min(1.0, token_dropout_rate))
        self.caption_dropout_rate = max(0.0, min(1.0, caption_dropout_rate))
        self.rarity_factor = max(0.0, min(1.0, rarity_factor))
        self.emphasis_factor = max(1.0, emphasis_factor)
        self.num_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # Initialize stats tracking with thread safety
        self.tag_stats = TagStats()
        self.weight_cache: Dict[str, float] = {}
        self._category_cache: Dict[str, Any] = {}
        self.tag_categories: Dict[str, Dict[str, Any]] = {
            "general": {
                "keywords": [],
                "base_weight": 1.0,
                "representative_tags": []
            }
        }
        
        # Clustering parameters with validation
        self.min_tag_freq = max(1, min_tag_freq)
        self.min_cluster_size = max(2, min_cluster_size)
        self.similarity_threshold = max(0.1, min(0.9, similarity_threshold))
        
        # Initialize logging
        logger.info(
            f"Initialized CaptionProcessor with token_dropout={token_dropout_rate}, "
            f"caption_dropout={caption_dropout_rate}, rarity_factor={rarity_factor}"
        )
    
    def analyze_dataset(self, captions: List[str]) -> None:
        """Analyze dataset to build tag statistics and categories."""
        logger.info("Analyzing dataset to build tag categories...")
        
        # Process all captions to build statistics
        all_tags = []
        for caption in captions:
            tags = self.parse_tags(caption)
            all_tags.extend(tags)
            
            # Update tag statistics
            for tag in tags:
                self.tag_stats.total_occurrences[tag] += 1
                
            # Update co-occurrence
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    self.tag_stats.update_cooccurrence(tag1, tag2)
        
        # Get frequent tags for clustering
        common_tags = [tag for tag, count in self.tag_stats.get_most_common_tags(1000)
                      if count >= self.min_tag_freq]
        
        if not common_tags:
            logger.warning("No common tags found in dataset")
            return
            
        # Create tag clusters based on co-occurrence
        self._create_tag_categories(common_tags)
        
    def _create_tag_categories(self, common_tags: List[str]) -> None:
        """Create tag categories using clustering on co-occurrence patterns and TF-IDF features.
        
        This method implements an advanced clustering approach that combines:
        1. Tag co-occurrence patterns
        2. TF-IDF feature analysis
        3. DBSCAN clustering
        4. Category naming based on representative tags
        
        Args:
            common_tags: List of tags that meet the minimum frequency threshold.
        """
        # Create co-occurrence matrix
        matrix_size = len(common_tags)
        cooccurrence_matrix = np.zeros((matrix_size, matrix_size))
        
        # Get all captions for TF-IDF
        tag_documents = []
        for tag in common_tags:
            # Get co-occurring tags as a document
            cooccurring = self.tag_stats.get_common_cooccurrences(tag)
            tag_documents.append(" ".join(cooccurring))
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features to most important ones
            stop_words='english',
            token_pattern=r'[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*'  # Handle compound tags
        )
        tfidf_matrix = vectorizer.fit_transform(tag_documents).toarray()
        
        # Combine co-occurrence and TF-IDF features
        for i, tag1 in enumerate(common_tags):
            for j, tag2 in enumerate(common_tags[i:], i):
                # Co-occurrence score
                count = self.tag_stats.cooccurrence[tag1][tag2]
                norm_count = count / math.sqrt(
                    self.tag_stats.total_occurrences[tag1] * 
                    self.tag_stats.total_occurrences[tag2]
                )
                
                # TF-IDF similarity
                tfidf_sim = np.dot(tfidf_matrix[i], tfidf_matrix[j]) / (
                    np.linalg.norm(tfidf_matrix[i]) * np.linalg.norm(tfidf_matrix[j])
                    + 1e-8  # Avoid division by zero
                )
                
                # Combine scores (weighted average)
                combined_score = 0.7 * norm_count + 0.3 * tfidf_sim
                
                cooccurrence_matrix[i, j] = combined_score
                cooccurrence_matrix[j, i] = combined_score
        
        # Cluster tags using DBSCAN
        clustering = DBSCAN(
            eps=self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='precomputed'
        ).fit(1 - cooccurrence_matrix)  # Convert similarity to distance
        
        # Create categories from clusters
        categories = defaultdict(list)
        for tag, cluster_id in zip(common_tags, clustering.labels_):
            if cluster_id != -1:  # Not noise
                categories[f"category_{cluster_id}"].append(tag)
        
        # Analyze and name categories
        self.tag_categories = {}
        for cluster_id, tags in categories.items():
            # Get most representative tags using TF-IDF scores
            cluster_docs = [" ".join(self.tag_stats.get_common_cooccurrences(tag))
                          for tag in tags]
            cluster_tfidf = vectorizer.transform(cluster_docs).toarray()
            importance_scores = np.mean(cluster_tfidf, axis=0)
            top_terms_idx = np.argsort(importance_scores)[-5:]  # Top 5 terms
            
            # Compute base weight based on average tag frequency and importance
            avg_freq = np.mean([
                self.tag_stats.total_occurrences[tag]
                for tag in tags
            ])
            total_freq = sum(self.tag_stats.total_occurrences.values())
            base_weight = 1.0 + math.log1p(avg_freq / total_freq)
            
            # Adjust weight based on tag importance
            avg_importance = np.mean(importance_scores[top_terms_idx])
            base_weight *= (1.0 + 0.2 * avg_importance)  # Small boost from importance
            
            self.tag_categories[cluster_id] = {
                "keywords": tags,
                "representative_tags": [vectorizer.get_feature_names_out()[i] 
                                     for i in top_terms_idx],
                "base_weight": min(self.MAX_WEIGHT, max(self.MIN_WEIGHT, base_weight))
            }
        
        # Add general category for unclustered tags
        self.tag_categories["general"] = {
            "keywords": [],
            "representative_tags": [],
            "base_weight": 1.0
        }
        
        logger.info(f"Created {len(self.tag_categories)} tag categories from dataset")
        for category, info in self.tag_categories.items():
            if category != "general":
                logger.debug(f"Category {category}: {info['representative_tags'][:3]}")
    
    def get_tag_category(self, tag: str) -> str:
        """Get the category for a tag."""
        if tag in self._category_cache:
            return self._category_cache[tag]
            
        # Check each category's keywords
        for category, info in self.tag_categories.items():
            if tag in info["keywords"]:
                self._category_cache[tag] = category
                return category
        
        # Get commonly co-occurring tags and their categories
        cooccurring_tags = self.tag_stats.get_common_cooccurrences(tag)
        if cooccurring_tags:
            # Assign to most common category among co-occurring tags
            category_counts = Counter(
                self.get_tag_category(t) for t in cooccurring_tags
            )
            if category_counts:
                most_common = category_counts.most_common(1)[0][0]
                self._category_cache[tag] = most_common
                return most_common
        
        self._category_cache[tag] = "general"
        return "general"
    
    def process_caption(self, caption: str, training: bool = True) -> Tuple[List[str], List[float]]:
        """Process a caption into tags and weights.
        
        Implements the tag processing pipeline from the NovelAI paper:
        1. Tag parsing and normalization
        2. Weight computation based on rarity and emphasis
        3. Optional token dropout during training
        
        Args:
            caption: Raw caption string to process.
            training: Whether to apply dropout during processing.
            
        Returns:
            Tuple containing:
            - List of processed tags
            - List of corresponding weights
        """
        # Parse and normalize tags
        tags = self.parse_tags(caption)
        if not tags:
            return [], []
            
        # Apply caption dropout during training
        if training and random.random() < self.caption_dropout_rate:
            return [], []
            
        # Process tags and compute weights
        processed_tags = []
        weights = []
        
        for tag in tags:
            # Apply token dropout during training
            if training and random.random() < self.token_dropout_rate:
                continue
                
            # Get or compute tag weight
            weight = self.get_tag_weight(tag)
            
            # Add to results if valid
            if weight > 0:
                processed_tags.append(tag)
                weights.append(weight)
                
        return processed_tags, weights
        
    def get_tag_weight(self, tag: str) -> float:
        """Compute the weight for a tag based on rarity and category.
        
        Implements the weight computation formula from the NovelAI paper:
        weight = base_weight * rarity_factor * emphasis_multiplier
        
        Args:
            tag: Tag to compute weight for.
            
        Returns:
            Computed weight, clamped to [MIN_WEIGHT, MAX_WEIGHT].
        """
        # Check cache first
        if tag in self.weight_cache:
            return self.weight_cache[tag]
            
        # Get base components
        rarity = self.tag_stats.get_rarity(tag)
        category_weight = self._get_category_weight(tag)
        
        # Compute final weight
        weight = category_weight * (1.0 + (rarity * self.rarity_factor))
        
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
        """Get the emphasis level of a tag based on curly brace markers.
        
        Args:
            tag: Tag to check for emphasis markers.
            
        Returns:
            Number of emphasis levels (number of curly brace pairs).
        """
        emphasis = 0
        while tag.startswith('{') and tag.endswith('}'):
            emphasis += 1
            tag = tag[1:-1].strip()
        return emphasis
        
    def _get_category_weight(self, tag: str) -> float:
        """Get the base weight for a tag's category.
        
        Args:
            tag: Tag to get category weight for.
            
        Returns:
            Base weight for the tag's category (defaults to 1.0).
        """
        # Check category cache
        if tag in self._category_cache:
            return self._category_cache[tag]
            
        # Find matching category
        for category, info in self.tag_categories.items():
            if tag in info["keywords"]:
                self._category_cache[tag] = info['base_weight']
                return info['base_weight']
                
        # Default to general category
        self._category_cache[tag] = 1.0
        return 1.0
    
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
    
    @lru_cache(maxsize=1024)
    def parse_tags(self, caption: str) -> List[str]:
        """Parse caption into individual tags with caching."""
        if not caption:
            return []
        return [t.strip() for t in caption.split(',') if t.strip()]
    
    def process_caption_batch(self, captions: List[str], num_workers: Optional[int] = None) -> List[List[str]]:
        """Process multiple captions in parallel."""
        if not captions:
            return []
            
        if num_workers is None:
            num_workers = self.num_workers
            
        def process_batch(caption_batch):
            return [self.parse_tags(caption) for caption in caption_batch]
            
        # Process in parallel for large batches
        if len(captions) > 100:
            batch_size = max(50, len(captions) // (num_workers * 4))
            batches = [captions[i:i + batch_size] 
                      for i in range(0, len(captions), batch_size)]
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                results = [r for f in futures for r in f.result()]
            return results
            
        # Process sequentially for small batches
        return process_batch(captions)
    
    def format_caption(self, caption: str) -> str:
        """Format caption for training, applying dropout if needed."""
        if not caption or self.caption_dropout_rate <= 0:
            return caption
            
        tags = self.parse_tags(caption)
        if not tags:
            return caption
            
        # Apply token dropout
        if self.token_dropout_rate > 0:
            tags = [tag for tag in tags 
                   if random.random() > self.token_dropout_rate]
            
        return ", ".join(tags)
    
    def process_tags_with_weights(self, caption: str) -> Tuple[List[str], List[float]]:
        """Process caption to extract tags and compute their weights."""
        tags = self.parse_tags(caption)
        if not tags:
            return [], []
            
        # Update statistics
        for tag in tags:
            category = self.get_tag_category(tag)
            self.tag_stats.update(category, tag)
            
        # Update co-occurrence for all tag pairs
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                self.tag_stats.update_cooccurrence(tag1, tag2)
        
        # Compute weights
        weights = [self.get_tag_weight(tag) for tag in tags]
        
        return tags, weights
    
    def process_tag_batch(self, tags: List[str]) -> List[float]:
        """Process a batch of tags in parallel to compute weights."""
        return [self.get_tag_weight(tag) for tag in tags]
    
    def process_caption_file(self, caption_path: Path) -> Tuple[float, Optional[str]]:
        """Process a caption file and return its weight and any error message.
        
        Args:
            caption_path: Path to the caption file
            
        Returns:
            Tuple of (weight, error_message). Weight defaults to 1.0 on error.
            Error message is None if processing succeeded.
        """
        try:
            if not caption_path.exists():
                return 1.0, "Caption file not found"

            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            if not caption:
                return 1.0, "Empty caption file"

            # Split caption into tags and calculate weights for each
            tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
            if not tags:
                return 1.0, "No valid tags found in caption"

            tag_weights = [self.get_tag_weight(tag) for tag in tags]
            # Use mean of tag weights as the overall weight
            weight = sum(tag_weights) / len(tag_weights) if tag_weights else 1.0
            return weight, None

        except Exception as e:
            return 1.0, str(e)
            
    def calculate_weights_batch(self, tag_pairs: List[Tuple[List[str], Dict[str, float]]], num_workers: Optional[int] = None) -> List[Dict[str, float]]:
        """Process multiple tag pairs in parallel.
        
        Args:
            tag_pairs: List of (tags, special_tags) pairs where:
                - tags is a list of regular tags
                - special_tags is a dict mapping tags to their weights
            num_workers: Optional number of workers for parallel processing
            
        Returns:
            List of dictionaries mapping tags to their computed weights
        """
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
            
        # Prepare tag lists for batch processing
        all_tag_lists = []
        for tags, special_tags in tag_pairs:
            # Add special tags with weights to the tag list
            all_tags = list(tags)
            for tag, weight in special_tags.items():
                if isinstance(weight, (int, float)):
                    all_tags.append(f"{tag}::{weight}")
            all_tag_lists.append(all_tags)
            
        # Process in parallel for large batches
        if len(tag_pairs) > 100:
            batch_size = max(50, len(tag_pairs) // (num_workers * 4))
            batches = [all_tag_lists[i:i + batch_size] 
                      for i in range(0, len(all_tag_lists), batch_size)]
            
            all_weights = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_tag_batch, batch) 
                         for batch in batches]
                for future in futures:
                    all_weights.extend(future.result())
        else:
            # Process sequentially for small batches
            all_weights = self.process_tag_batch([tag for tags in all_tag_lists for tag in tags])
            
            # Split weights back into batches
            offset = 0
            split_weights = []
            for tags in all_tag_lists:
                split_weights.append(all_weights[offset:offset + len(tags)])
                offset += len(tags)
            all_weights = split_weights
            
        # Convert weights back to dictionaries
        results = []
        for tags, weights in zip(all_tag_lists, all_weights):
            result = {}
            for tag, weight in zip(tags, weights):
                if "::" in tag:
                    tag = tag.split("::")[0]
                result[tag] = weight
            results.append(result)
            
        return results

def load_captions(image_paths: List[str]) -> Dict[str, str]:
    """Load captions for a list of image paths.
    
    Args:
        image_paths: List of paths to images
        
    Returns:
        Dictionary mapping image paths to their captions
    """
    processor = CaptionProcessor()
    captions = {}
    
    for image_path in image_paths:
        captions[image_path] = processor.load_caption(image_path)
            
    return captions
