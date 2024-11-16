import torch
import logging
import traceback
from typing import Dict, Set, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor
import os
from threading import Lock
import re
import spacy
import sys
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class TagStats:
    """Thread-safe container for tag statistics."""

    frequencies: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    class_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
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
            
    def get_emphasis(self, tag: str, default: float = 1.0) -> float:
        """Get emphasis score for a tag based on frequency."""
        with self._lock:
            total_freq = sum(
                self.frequencies[class_name][tag]
                for class_name in self.frequencies
            )
            if total_freq == 0:
                return default
            return math.log1p(total_freq) / 10.0 + default
            
    def get_rarity(self, tag: str, default: float = 1.0) -> float:
        """Get rarity score for a tag based on inverse frequency."""
        with self._lock:
            total_freq = sum(
                self.frequencies[class_name][tag]
                for class_name in self.frequencies
            )
            if total_freq == 0:
                return default
            return default / (math.log1p(total_freq) + 1.0)


@dataclass
class TagCache:
    """Cache container for tag computations."""

    rarity_scores: Dict[str, float] = field(default_factory=dict)
    importance_scores: Dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def update_rarity(self, tag: str, score: float) -> None:
        with self._lock:
            self.rarity_scores[tag] = score

    def get_rarity(self, tag: str, default: float = 1.0) -> float:
        with self._lock:
            return self.rarity_scores.get(tag, default)


class TagBasedLossWeighter:
    """Improved tag-based loss weighting system with thread safety and optimized performance."""

    # Class-level constants for weight boundaries
    MIN_WEIGHT: float = 0.1
    MAX_WEIGHT: float = 3.0
    DEFAULT_EMPHASIS_FACTOR: float = 1.1
    DEFAULT_RARITY_FACTOR: float = 0.9
    DEFAULT_QUALITY_BONUS: float = 0.2
    DEFAULT_CHARACTER_EMPHASIS: float = 1.2
    DEFAULT_CACHE_SIZE: int = 1024

    # Class-level category definitions
    TAG_CATEGORIES = {
        "quality": {
            "description": "Image quality and technical aspects",
            "keywords": ["quality", "resolution", "detail", "clarity", "fidelity", "sharpness", "definition"],
            "patterns": [r"\d+k", r"high.*quality", r"best.*quality", r"master.*piece"]
        },
        "style": {
            "description": "Artistic style and rendering technique",
            "keywords": ["style", "art", "artistic", "painting", "drawing", "render", "illustration"],
            "patterns": [r".*style", r".*painting", r".*art"]
        },
        "character": {
            "description": "People, beings, and character traits",
            "keywords": ["person", "character", "human", "face", "portrait", "gender", "age"],
            "patterns": [r".*person", r".*gender", r".*hair", r".*eyes"]
        },
        "action": {
            "description": "Actions, poses, and movements",
            "keywords": ["action", "pose", "movement", "position", "gesture", "motion"],
            "patterns": [r".*ing$", r".*pose", r".*action"]
        },
        "setting": {
            "description": "Environment and location",
            "keywords": ["scene", "location", "place", "environment", "setting", "background"],
            "patterns": [r".*scene", r".*ground", r".*scape"]
        },
        "object": {
            "description": "Physical objects and items",
            "keywords": ["object", "item", "thing", "prop", "element", "material"],
            "patterns": [r".*able$", r".*ture$"]
        },
        "emphasis": {
            "description": "Emphasis and intensity modifiers",
            "keywords": ["very", "extremely", "highly", "super", "ultra", "perfect"],
            "patterns": [r".*ly$", r".*est$"]
        },
        "meta": {
            "description": "Technical and prompt-related",
            "keywords": ["prompt", "seed", "steps", "cfg", "sampler", "model"],
            "patterns": [r".*prompt", r".*scale", r".*steps"]
        }
    }

    def __init__(
        self,
        tag_classes: Optional[Dict[str, Set[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with additional NLP components."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        try:
            self._init_config(config or {})
            self._init_tag_classes(tag_classes)
            self._init_statistics()
            self._init_executor()

            logger.info("TagBasedLossWeighter initialized successfully")
            self._log_config()

        except Exception as e:
            logger.error("TagBasedLossWeighter initialization failed: %s", str(e))
            logger.error(traceback.format_exc())
            raise

    def _init_config(self, config: Dict[str, Any]) -> None:
        """Initialize configuration parameters."""
        # Core parameters
        self.min_weight = self.MIN_WEIGHT = float(config.get('min_weight', 0.1))
        self.max_weight = self.MAX_WEIGHT = float(config.get('max_weight', 3.0))
        self.emphasis_factor = float(config.get('emphasis_factor', self.DEFAULT_EMPHASIS_FACTOR))
        self.rarity_factor = float(config.get('rarity_factor', self.DEFAULT_RARITY_FACTOR))
        self.quality_bonus = float(config.get('quality_bonus', self.DEFAULT_QUALITY_BONUS))
        self.character_emphasis = float(config.get('character_emphasis', self.DEFAULT_CHARACTER_EMPHASIS))
        self.cache_size = int(config.get('cache_size', self.DEFAULT_CACHE_SIZE))
        self.num_workers = int(config.get('num_workers', min(32, (os.cpu_count() or 1) + 4)))
        self.no_cache = bool(config.get('no_cache', False))
        
        # Initialize base weights for tag classes
        self.class_base_weights = {
            'quality': 1.2,
            'style': 1.1,
            'character': 1.3,
            'action': 1.0,
            'setting': 0.9,
            'object': 0.8,
            'emphasis': 1.2,
            'meta': 0.5
        }
        
        # Update class weights from config if provided
        if 'class_weights' in config:
            self.class_base_weights.update(config['class_weights'])

    def _init_tag_classes(self, tag_classes: Optional[Dict[str, Set[str]]]) -> None:
        """Initialize tag classification system."""
        try:
            # Initialize base mapping
            self.tag_classes = tag_classes or {
                "quality": set(),
                "style": set(),
                "character": set(),
                "action": set(),
                "setting": set(),
                "object": set(),
                "emphasis": set(),
                "meta": set()
            }
            
            # Create reverse mapping
            self.tag_to_class = {}
            for class_name, tags in self.tag_classes.items():
                for tag in tags:
                    self.tag_to_class[tag] = class_name
                    
            # Validate categories
            for category in self.TAG_CATEGORIES.values():
                if not isinstance(category, dict):
                    raise TypeError(f"Invalid category format: {category}")
                if not all(k in category for k in ["description", "keywords", "patterns"]):
                    raise ValueError(f"Missing required category fields in: {category}")
                    
            logger.info("Tag classification system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tag classes: {str(e)}")
            raise

    def _init_statistics(self) -> None:
        """Initialize statistical tracking components."""
        try:
            self.stats = TagStats()
            self.cache = TagCache()
            
            # Initialize caches
            if not hasattr(self, 'tag_weight_cache'):
                self.tag_weight_cache = {}
                
            # Configure caching behavior for calculate_tag_weights
            if not self.no_cache:
                self.calculate_tag_weights = lru_cache(maxsize=self.cache_size)(
                    self._calculate_tag_weights
                )
            else:
                self.calculate_tag_weights = self._calculate_tag_weights
                
            logger.info("Statistics and caches initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize statistics: {str(e)}")
            raise

    def _init_executor(self) -> None:
        """Initialize thread pool executor."""
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def _log_config(self) -> None:
        """Log configuration settings."""
        logger.info("TagBasedLossWeighter configuration:")
        logger.info("- Min weight: %s", self.min_weight)
        logger.info("- Max weight: %s", self.max_weight)
        logger.info("- Emphasis factor: %s", self.emphasis_factor)
        logger.info("- Rarity factor: %s", self.rarity_factor)
        logger.info("- Cache size: %s", self.cache_size)
        logger.info("- Workers: %s", self.num_workers)

    def parse_tags(self, caption: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse caption into tags and special tags with improved error handling.

        Args:
            caption (str): The caption text to parse

        Returns:
            Tuple[List[str], Dict[str, Any]]: A tuple containing:
                - List of parsed tags
                - Dictionary of special tag parameters

        Raises:
            TypeError: If caption is not a string
            ValueError: If tag weight parsing fails
            AttributeError: If string operations fail
            IndexError: If list operations fail
        """
        if not isinstance(caption, str):
            raise TypeError("Caption must be a string")

        if not caption:
            return [], {}

        tags = []
        special_tags = {}

        # Process tags in a single pass
        raw_tags = [t.strip().lower() for t in caption.split(",")]

        # Early processing for MJ tags
        has_mj_tags = any("niji" in t or t in ["4", "5", "6"] for t in raw_tags)

        for i, tag in enumerate(raw_tags):
            # Skip empty tags
            if not tag:
                continue

            # Process special tag formats
            if "::" in tag:
                tag, weight = self._process_weighted_tag(tag)
                if weight is not None:
                    special_tags[f"{tag}_weight"] = weight

            elif has_mj_tags:
                tag = self._process_mj_tag(
                    tag, i, len(raw_tags), special_tags
                )

            # Clean and add tag
            if tag := self._clean_tag(tag):
                tags.append(tag)

        return tags, special_tags

    def _process_weighted_tag(self, tag: str) -> Tuple[Optional[str], float]:
        """Process a weighted tag format (tag:weight)."""
        try:
            if ':' not in tag:
                return self._clean_tag(tag), 1.0
                
            tag_part, weight_part = tag.rsplit(':', 1)
            tag_part = self._clean_tag(tag_part)
            if not tag_part:
                return None, 1.0
                
            try:
                weight = float(weight_part.strip())
                # Apply class-based weight modifiers
                tag_class = self._classify_tag(tag_part)
                if tag_class:
                    class_stats = self.stats.frequencies.get(tag_class, {})
                    tag_count = class_stats.get(tag_part, 0)
                    total_count = sum(class_stats.values())
                    
                    if total_count > 0:
                        # Calculate rarity factor based on tag frequency within its class
                        rarity = 1.0 - (tag_count / total_count)
                        weight *= (1.0 + (rarity * self.rarity_factor))
                
                return tag_part, max(self.min_weight, min(self.max_weight, weight))
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight value in tag '{tag}', using default")
                return tag_part, 1.0
                
        except Exception as e:
            logger.warning(f"Error processing weighted tag '{tag}': {str(e)}")
            return None, 1.0

    def process_caption(self, caption: str) -> Tuple[List[str], Dict[str, float]]:
        """Process a caption to extract tags and special tags with weights."""
        try:
            if not caption:
                return [], {}

            # Split caption into chunks for parallel processing
            chunks = [t.strip() for t in caption.split(',') if t.strip()]
            if not chunks:
                return [], {}

            # Process chunks in parallel if enough chunks
            if len(chunks) > 10 and self.num_workers > 1:  # Only parallelize for larger inputs
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Process each chunk in parallel
                    futures = [executor.submit(self._process_tag_chunk, chunk) for chunk in chunks]
                    results = [future.result() for future in futures]
                    
                    # Combine results
                    tags = []
                    special_tags = {}
                    for chunk_tags, chunk_special in results:
                        if chunk_tags:  # Only extend if there are tags
                            tags.extend(chunk_tags)
                        if chunk_special:  # Only update if there are special tags
                            special_tags.update(chunk_special)
            else:
                # Process sequentially for small inputs
                tags = []
                special_tags = {}
                for chunk in chunks:
                    chunk_tags, chunk_special = self._process_tag_chunk(chunk)
                    if chunk_tags:  # Only extend if there are tags
                        tags.extend(chunk_tags)
                    if chunk_special:  # Only update if there are special tags
                        special_tags.update(chunk_special)

            # Remove duplicates while preserving order
            seen = set()
            tags = [tag for tag in tags if tag and not (tag in seen or seen.add(tag))]

            return tags, special_tags

        except Exception as e:
            logger.error(f"Error processing caption: {str(e)}")
            return [], {}

    def calculate_caption_weight(self, caption: str) -> float:
        """Calculate the overall weight for a caption."""
        try:
            if not caption:
                return 1.0

            tags, special_tags = self.process_caption(caption)
            if not tags and not special_tags:
                return 1.0

            weights = self._calculate_weights_no_cache(tags, special_tags)
            if not weights:
                return 1.0

            # Calculate final weight as average of individual tag weights
            total_weight = sum(weights.values())
            num_weights = len(weights)
            
            if num_weights == 0:
                return 1.0
                
            avg_weight = total_weight / num_weights
            return max(self.min_weight, min(self.max_weight, avg_weight))

        except Exception as e:
            logger.warning(f"Error calculating caption weight: {str(e)}")
            return 1.0

    def calculate_weights(
        self, tags: List[str], special_tags: Dict[str, any] = None
    ) -> Dict[str, float]:
        """Calculate weights with proper no_cache handling"""
        try:
            if not tags:
                return {tag: 1.0 for tag in tags}
                
            special_tags = special_tags or {}
            
            if self.no_cache:
                return self._calculate_weights_no_cache(tags, special_tags)
            else:
                return self._calculate_weights_cached(tags, special_tags)
                
        except Exception as e:
            logger.warning(f"Error calculating weights: {str(e)}")
            return {tag: 1.0 for tag in tags}

    def _calculate_weights_no_cache(
        self, tags: List[str], special_tags: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate weights for tags without using cache."""
        try:
            weights = {}
            special_tags = special_tags or {}

            # Process regular tags
            for tag in tags:
                if not tag:
                    continue
                    
                # Get base weight (from explicit weighting or default 1.0)
                tag_clean = self._clean_tag(tag)
                if not tag_clean:
                    continue
                    
                base_weight = 1.0
                if ':' in tag:
                    tag_part, weight = self._process_weighted_tag(tag)
                    if tag_part:
                        base_weight = weight
                        tag_clean = tag_part

                # Apply class-based weighting
                tag_class = self._classify_tag(tag_clean)
                if tag_class:
                    class_stats = self.stats.frequencies.get(tag_class, {})
                    tag_count = class_stats.get(tag_clean, 0)
                    total_count = sum(class_stats.values())
                    
                    if total_count > 0:
                        # Calculate rarity factor
                        rarity = 1.0 - (tag_count / total_count)
                        # Apply emphasis and rarity factors
                        final_weight = base_weight * (1.0 + (rarity * self.rarity_factor))
                        weights[tag_clean] = max(self.min_weight, min(self.max_weight, final_weight))
                    else:
                        weights[tag_clean] = base_weight
                else:
                    weights[tag_clean] = base_weight

            # Add special tag weights
            for tag, weight in special_tags.items():
                if isinstance(weight, (int, float)):
                    weights[tag] = max(self.min_weight, min(self.max_weight, float(weight)))
                else:
                    weights[tag] = 1.0

            return weights

        except Exception as e:
            logger.warning(f"Error calculating weights: {str(e)}")
            return {tag: 1.0 for tag in tags}

    def calculate_static_weights(
        self,
        tags: List[str], special_tags: Dict[str, any] = None
    ) -> float:
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
        if "masterpiece" in tags:
            base_weight *= 1.3
        if special_tags.get("niji", False):
            base_weight *= 1.2
        if "stylize" in special_tags:
            stylize_value = special_tags["stylize"]
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if "chaos" in special_tags:
            chaos_value = special_tags["chaos"]
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor

        # Clamp between min and max
        return max(
            self.MIN_WEIGHT,
            min(self.MAX_WEIGHT, base_weight),
        )

    def _calculate_weights_cached(
        self, tags: List[str], special_tags: Dict[str, any] = None
    ) -> Dict[str, float]:
        """Cached weight calculation"""
        # Convert inputs to hashable types for caching
        tags_tuple = tuple(sorted(tags))
        return self.calculate_tag_weights(
            tags_tuple, frozenset(special_tags.items()) if special_tags else None
        )

    def update_training_loss(self, loss: torch.Tensor, tags: List[str]) -> torch.Tensor:
        """
        Apply tag-based weighting to the training loss.

        Args:
            loss (torch.Tensor): Original loss value
            tags (list): List of tags for the current image

        Returns:
            torch.Tensor: Weighted loss value

        Raises:
            TypeError: If loss is not a torch.Tensor or tags is not a list
            ValueError: If loss tensor has invalid shape or tags are malformed
            RuntimeError: If weight calculation fails due to computational error
        """
        # Input validation
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss must be a torch.Tensor, got {type(loss)}")
        if not isinstance(tags, list):
            raise TypeError(f"Tags must be a list, got {type(tags)}")
        if not loss.numel():
            raise ValueError("Loss tensor is empty")
            
        try:
            # Calculate weights with specific error handling
            weight = self.calculate_weights(tags)
            
            # Validate weight before applying
            if not isinstance(weight, (float, int, torch.Tensor)):
                raise TypeError(f"Invalid weight type: {type(weight)}")
            if isinstance(weight, torch.Tensor) and not weight.numel():
                raise ValueError("Weight tensor is empty")
                
            # Apply weight to loss
            weighted_loss = loss * weight
            
            # Verify result
            if not torch.isfinite(weighted_loss).all():
                raise RuntimeError(f"Weighted loss contains invalid values: {weighted_loss}")
                
            return weighted_loss
            
        except (KeyError, IndexError) as e:
            logger.error("Tag processing failed: %s", str(e))
            raise ValueError(f"Failed to process tags: {str(e)}")
        except torch.cuda.CudaError as e:
            logger.error("CUDA error during loss weighting: %s", str(e))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA error in loss weighting: {str(e)}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Out of memory during loss weighting")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise RuntimeError(f"Runtime error in loss weighting: {str(e)}")

    def reset_statistics(self):
        """Reset all statistical tracking"""
        self.stats.frequencies.clear()
        self.stats.class_counts.clear()
        self.stats.cooccurrence.clear()
        self.cache.rarity_scores.clear()
        self.cache.importance_scores.clear()

        # Clear caches
        self.calculate_tag_weights.cache_clear()
        self._classify_tag.cache_clear()

    def __del__(self):
        """Cleanup worker pool on deletion"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    @lru_cache(maxsize=1024)
    def _classify_tag(self, tag: str) -> str:
        """Classify a tag into its category."""
        try:
            # Input validation
            if not isinstance(tag, str):
                raise TypeError(f"Tag must be a string, got {type(tag)}")
            if not tag:
                return "default"

            # First check explicit mappings
            if tag in self.tag_to_class:
                return self.tag_to_class[tag]

            # Then try pattern matching
            for class_name, category in self.TAG_CATEGORIES.items():
                # Check keywords
                if any(keyword in tag.lower() for keyword in category["keywords"]):
                    return class_name
                # Check patterns
                if any(re.match(pattern, tag, re.IGNORECASE) for pattern in category["patterns"]):
                    return class_name

            # Default classification based on NLP analysis
            try:
                doc = self.nlp(tag)
                
                # Basic NLP-based classification
                if doc[0].pos_ in ["NOUN", "PROPN"]:
                    if any(ent.label_ == "PERSON" for ent in doc.ents):
                        return "character"
                    return "object"
            except Exception as nlp_error:
                logger.warning(f"NLP analysis failed for tag '{tag}': {str(nlp_error)}")
                
            return "default"  # Fallback to default category

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
        if class_name == "character":
            importance *= self.character_emphasis

        # Apply emphasis for emphasized tags
        if class_name == "emphasis" or tag in self.tag_classes.get("emphasis", set()):
            importance *= self.emphasis_factor

        # Apply rarity bonus
        rarity_score = self.cache.get_rarity(tag, 1.0)
        importance *= rarity_score

        # Apply quality bonus for high-quality images
        if class_name == "quality" and any(
            q in tags for q in ["masterpiece", "best quality", "high quality"]
        ):
            importance *= 1.0 + self.quality_bonus

        return importance

    def _calculate_tag_weights(self, tags_tuple: Tuple[str, ...]) -> float:
        """
        Calculate tag weights with improved weighting scheme.
        Raises exceptions on failure instead of returning default values.

        Args:
            tags_tuple (tuple): Tuple of tags for weight calculation

        Returns:
            float: Calculated weight value

        Raises:
            TypeError: For invalid data types
            ValueError: For invalid values
            RuntimeError: For PyTorch/CUDA errors
            KeyError: For tag lookup failures
            AttributeError: For missing attributes
        """
        tags = list(tags_tuple)
        weights = []

        # Calculate importance for each tag
        for tag in tags:
            importance = self._calculate_tag_importance(tag, tags)
            weights.append(importance)

        if not weights:
            raise ValueError("No valid weights calculated for tags")

        # Calculate final weight using weighted geometric mean
        weights = torch.tensor(weights, dtype=torch.float32)
        final_weight = torch.exp(torch.log(weights + 1e-6).mean())

        # Clamp between min and max
        return torch.clamp(final_weight, self.min_weight, self.max_weight).item()

    def _calculate_batch_weights(self, batch_tags: List[List[str]]) -> List[float]:
        """
        Calculate weights for a batch of tag lists in parallel.

        Args:
            batch_tags (List[List[str]]): List of tag lists to process

        Returns:
            List[float]: List of calculated weights
        """
        if len(batch_tags) > 50:  # Only parallelize for larger batches
            futures = [
                self.executor.submit(self._calculate_tag_weights, tuple(tags))
                for tags in batch_tags
            ]
            return [future.result() for future in futures]
        else:
            return [self._calculate_tag_weights(tuple(tags)) for tags in batch_tags]

    def format_caption(self, caption: str) -> str:
        """
        Static method to format caption text with standardized formatting.

        Args:
            caption (str): Raw caption text

        Returns:
            str: Formatted caption text

        Raises:
            TypeError: If caption is not a string
            AttributeError: If string operations fail
            IndexError: If list operations fail
        """
        if not isinstance(caption, str):
            logger.error("Caption must be a string, got %s", type(caption))
            return ""

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
                tag = " ".join(tag.split())

                # Remove articles from start
                if tag.startswith(("a ", "an ", "the ")):
                    tag = " ".join(tag.split()[1:])

                # Handle special formatting for quality tags
                if any(
                    q in tag for q in ["masterpiece", "best quality", "high quality"]
                ):
                    formatted_tags.insert(0, tag)  # Move to front
                    continue

                # Handle special formatting for negative tags
                if tag.startswith(("no ", "bad ", "worst ")):
                    if not any(
                        neg in tag for neg in ["negative space", "negative prompt"]
                    ):
                        tag = (
                            tag.replace("no ", "")
                            .replace("bad ", "")
                            .replace("worst ", "")
                        )
                        tag = f"lowquality {tag}"

                formatted_tags.append(tag)

            # Join tags with standardized separator
            return ", ".join(formatted_tags)

        except AttributeError as e:
            logger.error("String operation error in caption formatting: %s", str(e))
            return caption  # Return original if formatting fails
        except IndexError as e:
            logger.error("List operation error in caption formatting: %s", str(e))
            return caption  # Return original if formatting fails
        except TypeError as e:
            logger.error("Type error in caption formatting: %s", str(e))
            return caption  # Return original if formatting fails
