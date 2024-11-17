import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from .base import CustomDatasetBase
from .bucket_manager import BucketManager
from .dataset_initializer import DatasetInitializer
from .image_grouper import ImageGrouper
from ..caption_processor import CaptionProcessor
from ..image_processor import ImageProcessor
from ..latent_cache import LatentCacheManager

logger = logging.getLogger(__name__)

class CustomDataset(CustomDatasetBase):
    """Custom dataset implementation for SDXL training with advanced features.
    
    This class provides a comprehensive dataset implementation that:
    1. Efficiently processes and caches image latents
    2. Handles dynamic image resolutions through bucket batching
    3. Processes captions with tag weighting
    4. Manages memory usage through disk caching
    
    Args:
        data_dir (str): Directory containing image-caption pairs
        vae (Optional[torch.nn.Module]): VAE model for latent computation
        tokenizer (Optional[Any]): Tokenizer for text processing
        tokenizer_2 (Optional[Any]): Secondary tokenizer for text processing
        text_encoder (Optional[Any]): Text encoder for text processing
        text_encoder_2 (Optional[Any]): Secondary text encoder for text processing
        cache_dir (str): Directory for caching latents
        no_caching_latents (bool): Disable latent caching
        all_ar (bool): Use aspect ratio for bucketing
        min_size (int): Minimum image size
        max_size (int): Maximum image size
        bucket_step_size (int): Resolution step size for buckets
        max_bucket_area (int): Maximum area for buckets
        token_dropout_rate (float): Token dropout probability
        caption_dropout_rate (float): Caption dropout probability
        min_tag_weight (float): Minimum weight for tags
        max_tag_weight (float): Maximum weight for tags
        use_tag_weighting (bool): Enable tag-based loss weighting
    """
    
    def __init__(self, 
                 data_dir: str, 
                 vae: Optional[torch.nn.Module] = None,
                 tokenizer: Optional[Any] = None,
                 tokenizer_2: Optional[Any] = None,
                 text_encoder: Optional[Any] = None,
                 text_encoder_2: Optional[Any] = None,
                 cache_dir: str = "latents_cache", 
                 no_caching_latents: bool = False,
                 all_ar: bool = False, 
                 min_size: int = 512, 
                 max_size: int = 4096,
                 bucket_step_size: int = 64, 
                 max_bucket_area: int = 1024*1024,
                 token_dropout_rate: float = 0.1, 
                 caption_dropout_rate: float = 0.1,
                 min_tag_weight: float = 0.1, 
                 max_tag_weight: float = 3.0,
                 use_tag_weighting: bool = True):
        super().__init__()
        
        # Store configuration
        self.config = {
            'data_dir': data_dir,
            'vae': vae,
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
            'text_encoder': text_encoder,
            'text_encoder_2': text_encoder_2,
            'cache_dir': cache_dir,
            'no_caching_latents': no_caching_latents,
            'all_ar': all_ar,
            'min_size': min_size,
            'max_size': max_size,
            'bucket_step_size': bucket_step_size,
            'max_bucket_area': max_bucket_area,
            'token_dropout_rate': token_dropout_rate,
            'caption_dropout_rate': caption_dropout_rate,
            'min_tag_weight': min_tag_weight,
            'max_tag_weight': max_tag_weight,
            'use_tag_weighting': use_tag_weighting
        }
        
        # Set device based on VAE or CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if vae is not None:
            self.device = next(vae.parameters()).device
            
        # Initialize components
        self._initialize_components()
        
        # Initialize dataset
        self._initialize_dataset()
        
        # Mark as initialized
        self._initialized = True
        
    def _initialize_components(self) -> None:
        """Initialize all dataset components with proper error handling."""
        try:
            # Initialize processors
            self.image_processor = ImageProcessor()
            self.caption_processor = CaptionProcessor(
                token_dropout_rate=self.config['token_dropout_rate'],
                caption_dropout_rate=self.config['caption_dropout_rate']
            )
            
            # Initialize cache managers
            self.latents_cache = LatentCacheManager(
                cache_dir=self.config['cache_dir'],
                vae=self.config['vae']
            )
            
            # Initialize bucket management
            self.bucket_manager = BucketManager(
                min_size=self.config['min_size'],
                max_size=self.config['max_size'],
                step_size=self.config['bucket_step_size'],
                max_area=self.config['max_bucket_area']
            )
            
            # Initialize image grouper
            self.image_grouper = ImageGrouper(self.bucket_manager)
            
            # Initialize dataset initializer
            self.dataset_initializer = DatasetInitializer()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _initialize_dataset(self) -> None:
        """Initialize dataset structure and caches."""
        try:
            # Find valid image-caption pairs
            self.image_paths = self.dataset_initializer.find_valid_pairs(self.config['data_dir'])
            if not self.image_paths:
                raise ValueError(f"No valid image-caption pairs found in {self.config['data_dir']}")
            
            # Group images into buckets
            self.bucket_data = self.image_grouper.group_images(
                image_paths=self.image_paths,
                use_ar=self.config['all_ar']
            )
            
            # Initialize latent caching if enabled
            if not self.config['no_caching_latents']:
                self._initialize_latent_cache()
            
            # Precompute tag weights if enabled
            if self.config['use_tag_weighting']:
                self._precompute_tag_weights()
            
            logger.info(f"Dataset initialized with {len(self.image_paths)} images in {len(self.bucket_data)} buckets")
            
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {str(e)}")
            raise

    def _initialize_latent_cache(self) -> None:
        """Initialize and process latent caching."""
        if self.config['vae'] is None:
            logger.warning("VAE not provided, skipping latent caching")
            return
            
        try:
            # Process images by bucket to ensure consistent tensor sizes
            batch_size = 4  # Adjust based on GPU memory
            
            # Iterate through each bucket group
            for bucket_dims, image_paths in self.bucket_data.items():
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i + batch_size]
                    
                    # Load and transform images using bucket dimensions
                    batch_tensors = []
                    valid_paths = []
                    for path in batch_paths:
                        image = self.image_processor.load_and_transform(
                            path,
                            target_height=bucket_dims[0],
                            target_width=bucket_dims[1]
                        )
                        if image is not None:
                            batch_tensors.append(image)
                            valid_paths.append(path)
                    
                    if not batch_tensors:
                        continue
                        
                    # Stack tensors and process batch
                    try:
                        batch_tensor = torch.stack(batch_tensors)
                        latents = self.latents_cache.process_latents_batch(batch_tensor)
                        
                        # Cache individual latents
                        if latents is not None:
                            for idx, path in enumerate(valid_paths):
                                self.latents_cache.latents_cache[path] = latents[idx]
                                
                    except Exception as e:
                        logger.error(f"Failed to process latent batch: {str(e)}")
                        continue
                        
            logger.info("Latent cache initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize latent cache: {str(e)}")
            raise

    def _precompute_tag_weights(self) -> None:
        """Pre-compute tag weights for all captions using thread pool."""
        try:
            # Process captions in parallel
            with ThreadPoolExecutor(max_workers=min(32, len(self.image_paths))) as executor:
                futures = []
                for path in self.image_paths:
                    futures.append(executor.submit(self.caption_processor.load_caption, path))
                
                # Process results
                for path, future in zip(self.image_paths, futures):
                    try:
                        caption = future.result()
                        if caption:
                            tags, weights = self.caption_processor.process_tags_with_weights(caption)
                            self.caption_processor.weight_cache[path] = (tags, weights)
                    except Exception as e:
                        logger.debug(f"Failed to process caption for {path}: {str(e)}")
                        continue
            
            logger.info(f"Precomputed weights for {len(self.caption_processor.weight_cache)} captions")
            
        except Exception as e:
            logger.error(f"Failed to precompute tag weights: {str(e)}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single item from the dataset.
        
        Args:
            index: Index of the item to get
            
        Returns:
            Dictionary containing:
                - image: Preprocessed image tensor
                - latents: VAE latent tensor (if caching enabled)
                - caption: Original caption string
                - tags: List of extracted tags
                - weights: Tag weights for loss computation
                - bucket_dims: (height, width) of the bucket
                - image_path: Path to the original image
        """
        try:
            # Get image path and bucket dimensions
            bucket_idx = index % len(self.bucket_data)
            bucket_dims = list(self.bucket_data.keys())[bucket_idx]
            image_paths = self.bucket_data[bucket_dims]
            image_idx = (index // len(self.bucket_data)) % len(image_paths)
            image_path = image_paths[image_idx]
            
            # Load and process image with target dimensions
            image = self.image_processor.load_and_transform(
                image_path,
                target_height=bucket_dims[0],
                target_width=bucket_dims[1]
            )
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Get latents if caching is enabled
            latents = None
            if not self.config['no_caching_latents']:
                latents = self.latents_cache.get_latents(image_path, image)
            
            # Load and process caption
            caption = self.caption_processor.load_caption(image_path)
            if caption is None:
                raise ValueError(f"Failed to load caption for {image_path}")
                
            # Get cached weights or compute them
            if image_path in self.caption_processor.weight_cache:
                tags, weights = self.caption_processor.weight_cache[image_path]
            else:
                tags, weights = self.caption_processor.process_tags_with_weights(caption)
            
            # Build return dictionary
            item = {
                'image': image,
                'caption': caption,
                'tags': tags,
                'weights': weights,
                'bucket_dims': bucket_dims,
                'image_path': image_path
            }
            
            if latents is not None:
                item['latents'] = latents
                
            return item
            
        except Exception as e:
            logger.error(f"Error getting item {index}: {str(e)}")
            raise

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        total_images = sum(len(paths) for paths in self.bucket_data.values())
        return total_images

    def shuffle_dataset(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset while maintaining bucket grouping.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Shuffle within each bucket
        for bucket_dims in self.bucket_data:
            paths = self.bucket_data[bucket_dims]
            indices = torch.randperm(len(paths)).tolist()
            self.bucket_data[bucket_dims] = [paths[i] for i in indices]
        
        logger.info("Dataset shuffled within buckets")

    def get_bucket_stats(self) -> Dict[Tuple[int, int], int]:
        """Get statistics about bucket usage.
        
        Returns:
            Dictionary mapping bucket dimensions to number of images
        """
        return {dims: len(paths) for dims, paths in self.bucket_data.items()}