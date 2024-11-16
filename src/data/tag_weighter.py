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
        self.tag_classes = tag_classes or {
            "character": set(),
            "style": set(),
            "setting": set(),
            "action": set(),
            "object": set(),
            "quality": set(),
            "emphasis": set(),
            "meta": set(),
        }

        # Create tag to class mapping
        self.tag_to_class = {
            tag: class_name
            for class_name, tags in self.tag_classes.items()
            for tag in tags
        }

    def _init_statistics(self) -> None:
        """Initialize statistical tracking components."""
        self.stats = TagStats()
        self.cache = TagCache()

        # Configure caching behavior
        if not self.no_cache:
            self.calculate_tag_weights = lru_cache(maxsize=self.cache_size)(
                self._calculate_tag_weights
            )
            # Initialize cached version of _classify_tag
            self._classify_tag = lru_cache(maxsize=1024)(self._classify_tag.__wrapped__)
        else:
            self.calculate_tag_weights = self._calculate_tag_weights
            # Use uncached version
            self._classify_tag = self._classify_tag.__wrapped__

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

    def _process_weighted_tag(self, tag: str) -> Tuple[str, Optional[float]]:
        """Process a weighted tag format (tag::weight)."""
        parts = tag.split("::")
        try:
            return parts[0].strip(), float(parts[1])
        except (IndexError, ValueError):
            return parts[0].strip(), None

    def _process_mj_tag(
        self,
        tag: str, index: int, total_tags: int, special_tags: Dict[str, Any]
    ) -> Optional[str]:
        """Process Midjourney-specific tags.

        Args:
            tag (str): The tag to process
            index (int): Position of tag in the sequence
            total_tags (int): Total number of tags
            special_tags (Dict[str, Any]): Dictionary to store special tag parameters

        Returns:
            Optional[str]: Processed tag or None if tag was consumed

        Raises:
            ValueError: If numeric parameter parsing fails
            AttributeError: If regex pattern matching fails
            TypeError: If parameter type conversion fails
        """
        # Handle style/version tags
        if index == 0 and ("anime style" in tag or "niji" in tag):
            special_tags["niji"] = True
            return None
        if index == total_tags - 1 and tag in ["4", "5", "6"]:
            return "masterpiece"

        # Handle parameters
        for param in ["stylize", "chaos", "sw", "sv"]:
            if param in tag:
                try:
                    match = re.search(r"[\d.]+", tag)
                    if match is None:
                        logger.warning(
                            "No numeric value found in parameter tag: %s", tag
                        )
                        continue
                    value = float(match.group())
                    special_tags[param] = value
                    return None
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(
                        "Failed to parse parameter %s from tag %s: %s",
                        param,
                        tag,
                        str(e),
                    )
                    continue

        return tag

    def _clean_tag(self, tag: str) -> Optional[str]:
        """Clean and normalize a tag."""
        if tag.startswith(("a ", "an ", "the ")):
            tag = " ".join(tag.split()[1:])
        return tag.strip() if tag.strip() else None

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
            tags = [t.strip() for t in caption.split(",")]

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

    @lru_cache(maxsize=1024)
    def _classify_tag(self, tag: str) -> Optional[str]:
        """
        Classify a tag using NLP-based semantic analysis and pattern matching.
        
        Args:
            tag (str): The tag to classify
            
        Returns:
            Optional[str]: The most likely category for the tag, or None if uncertain
            
        Raises:
            ValueError: If tag is empty or invalid
        """
        if not isinstance(tag, str):
            raise ValueError(f"Tag must be a string, got {type(tag)}")
        
        tag = tag.strip().lower()
        if not tag:
            raise ValueError("Tag cannot be empty")

        # First check explicit mapping
        if tag in self.tag_to_class:
            return self.tag_to_class[tag]

        # Process tag with spaCy
        doc = self.nlp(tag)
        
        # Calculate scores for each category
        scores = {}
        for category, info in self.TAG_CATEGORIES.items():
            score = 0.0
            
            # Check exact keyword matches
            if any(keyword in tag for keyword in info["keywords"]):
                score += 1.0
                
            # Check regex patterns
            if any(re.search(pattern, tag) for pattern in info["patterns"]):
                score += 0.5
                
            # Check semantic similarity with keywords
            keyword_docs = [self.nlp(keyword) for keyword in info["keywords"]]
            semantic_scores = [doc.similarity(keyword_doc) for keyword_doc in keyword_docs]
            if semantic_scores:
                score += max(semantic_scores) * 0.8
                
            # Consider part of speech
            if doc[0].pos_ == "VERB" and category == "action":
                score += 0.3
            elif doc[0].pos_ == "NOUN" and category in ["object", "character", "setting"]:
                score += 0.3
            elif doc[0].pos_ == "ADJ" and category in ["quality", "emphasis"]:
                score += 0.3
                
            scores[category] = score
        
        # Get category with highest score if it exceeds threshold
        best_category = max(scores.items(), key=lambda x: x[1])
        if best_category[1] > 0.5:  # Confidence threshold
            return best_category[0]
            
        logger.debug("Could not confidently classify tag: %s (scores: %s)", tag, scores)
        return None

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
        if tag in self.tag_classes["emphasis"]:
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

    def calculate_weights(
        self, tags: List[str], special_tags: Dict[str, any] = None
    ) -> Dict[str, float]:
        """Calculate weights with proper no_cache handling"""
        if self.no_cache:
            # Calculate weights directly without caching
            return self._calculate_weights_no_cache(tags, special_tags)
        else:
            # Use cached calculation
            return self._calculate_weights_cached(tags, special_tags)

    def _calculate_weights_no_cache(
        self, tags: List[str], special_tags: Dict[str, any] = None
    ) -> Dict[str, float]:
        """Direct weight calculation without caching"""
        if special_tags is None:
            special_tags = {}

        weights = {}
        base_weight = 1.0

        # Apply modifiers directly without caching
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

        # Calculate individual weights
        for i, tag in enumerate(tags):
            # Get tag class importance
            class_name = self._classify_tag(tag)
            class_weight = self.class_base_weights.get(class_name, 1.0)

            # Apply position decay
            position_weight = 1.0 - (i * 0.05)

            # Apply rarity bonus
            rarity_score = self.cache.get_rarity(tag, 1.0)

            # Combine all factors
            final_weight = base_weight * class_weight * position_weight * rarity_score

            # Apply any explicit tag weights
            if f"{tag}_weight" in special_tags:
                final_weight *= special_tags[f"{tag}_weight"]

            weights[tag] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, final_weight))

        return weights

    def _calculate_weights_cached(
        self, tags: List[str], special_tags: Dict[str, any] = None
    ) -> Dict[str, float]:
        """Cached weight calculation"""
        # Convert inputs to hashable types for caching
        tags_tuple = tuple(sorted(tags))
        return self.calculate_tag_weights(
            tags_tuple, frozenset(special_tags.items()) if special_tags else None
        )

    def process_caption(self, caption: str) -> Tuple[List[str], Dict[str, float]]:
        """Process a caption to extract tags and weights.
        
        Args:
            caption (str): The caption text to process
            
        Returns:
            Tuple[List[str], Dict[str, float]]: A tuple containing:
                - List of regular tags
                - Dictionary mapping special tags to their weights
        """
        if not caption:
            return [], {}
            
        try:
            # Parse tags and special tags
            tags, special_tags = self.parse_tags(caption)
            
            # Convert special tags to a proper dictionary
            if isinstance(special_tags, dict):
                special_tags_dict = special_tags
            else:
                special_tags_dict = {}
                for tag in special_tags:
                    if isinstance(tag, str):
                        if "::" in tag:
                            tag_name, weight = tag.split("::", 1)
                            try:
                                special_tags_dict[tag_name] = float(weight)
                            except ValueError:
                                special_tags_dict[tag_name] = 1.0
                        else:
                            special_tags_dict[tag] = 1.0
                            
            return tags, special_tags_dict
            
        except Exception as e:
            logger.warning(f"Error processing caption: {str(e)}")
            return [], {}

    def calculate_caption_weight(self, caption: str) -> float:
        """Calculate the weight for a caption.
        
        Args:
            caption (str): The caption text
            
        Returns:
            float: The calculated weight, clamped between MIN_WEIGHT and MAX_WEIGHT
        """
        if not caption:
            return 1.0
            
        try:
            # Process caption to get tags and weights
            tags, special_tags = self.process_caption(caption)
            
            # Calculate weights
            weights = self.calculate_weights(tags, special_tags)
            
            if isinstance(weights, dict):
                # Average the weights if multiple were returned
                tag_weight = sum(weights.values()) / len(weights) if weights else 1.0
            else:
                # Single weight value
                tag_weight = float(weights) if weights else 1.0
                
            # Clamp weight to valid range
            return max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, tag_weight))
            
        except Exception as e:
            logger.warning(f"Error calculating caption weight: {str(e)}")
            return 1.0

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
            logger.error("Loss update failed: %s", str(e))
            raise

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
