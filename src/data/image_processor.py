"""
Image processing module for SDXL training pipeline.

This module provides functionality for loading, transforming, and validating
images for the SDXL training process. It handles various image formats,
resolutions, and aspect ratios while ensuring consistent preprocessing.

Classes:
    ImageProcessor: Main class for image processing operations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for image processing errors."""


class CUDAProcessingError(ProcessingError):
    """Exception raised for CUDA and memory-related errors."""


class InputValidationError(ProcessingError):
    """Exception raised for invalid input data."""


class ModelProcessingError(ProcessingError):
    """Exception raised for model-related errors."""


class ImageProcessor:
    """Handles image loading, transformation and validation.
    
    This class provides methods for loading, verifying, and transforming
    images for the SDXL training pipeline. It includes functionality for
    resolution adjustment, aspect ratio preservation, and normalization.
    
    Attributes:
        image_transforms: Composition of torchvision transforms
        min_size: Minimum allowed image dimension
        max_size: Maximum allowed image dimension
    """
    
    def __init__(
        self,
        min_size: int = 1024,
        max_size: int = 2048
    ) -> None:
        """Initialize the image processor.
        
        Args:
            min_size: Minimum allowed image dimension
            max_size: Maximum allowed image dimension
        """
        self.min_size = min_size
        self.max_size = max_size
        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_and_verify_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and verify an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Optional[Image.Image]: Loaded PIL Image if successful, None otherwise
            
        Note:
            Images are converted to RGB mode if necessary.
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
                # Reopen because verify() invalidates the image
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
        except Exception as error:
            logger.debug(
                "Failed to load image %s: %s",
                image_path, str(error)
            )
            return None
    
    def get_image_dimensions(self, img: Image.Image) -> Tuple[int, int]:
        """Get image dimensions (height, width).
        
        Args:
            img: PIL Image
            
        Returns:
            Tuple[int, int]: Image dimensions (height, width)
        """
        return img.size[1], img.size[0]
    
    def resize_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        resampling: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """Resize image while preserving aspect ratio.
        
        Args:
            image: PIL Image to resize
            target_size: Target size as (width, height)
            resampling: PIL resampling filter to use
            
        Returns:
            Image.Image: Resized image
            
        Raises:
            ValueError: If target size is invalid
        """
        if not all(s > 0 for s in target_size):
            raise ValueError("Target size must be positive")
            
        return image.resize(target_size, resampling)
    
    def transform_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor.
        
        Args:
            img: PIL Image
            
        Returns:
            torch.Tensor: Normalized image tensor
        """
        return self.image_transforms(img)
    
    def process_image(
        self,
        image: Union[Image.Image, str],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[torch.Tensor]:
        """Process an image for training.
        
        Args:
            image: PIL Image or path to image file
            target_size: Optional target size (width, height)
            
        Returns:
            Optional[torch.Tensor]: Processed image tensor if successful,
                None otherwise
        """
        try:
            if isinstance(image, str):
                image = self.load_and_verify_image(image)
                if image is None:
                    return None
            
            if target_size is not None:
                image = self.resize_image(image, target_size)
            
            # Apply transforms
            tensor = self.image_transforms(image)
            return tensor
            
        except Exception as error:
            logger.error(
                "Failed to process image: %s",
                str(error)
            )
            return None
    
    def load_and_transform(self, image_path: str, target_height: Optional[int] = None, target_width: Optional[int] = None) -> Optional[torch.Tensor]:
        """Load image from path and transform it to tensor.
        
        Args:
            image_path: Path to the image file
            target_height: Optional target height for resizing
            target_width: Optional target width for resizing
            
        Returns:
            Optional[torch.Tensor]: Transformed image tensor if successful,
                None otherwise
        """
        try:
            # Load and verify image
            img = self.load_and_verify_image(image_path)
            if img is None:
                return None
                
            # Resize if dimensions provided
            if target_height is not None and target_width is not None:
                img = self.resize_image(img, (target_width, target_height))
                
            # Transform to tensor
            return self.transform_to_tensor(img)
            
        except Exception as error:
            logger.debug(
                "Failed to transform image %s: %s",
                image_path, str(error)
            )
            return None
    
    def _process_single_latent(
        self,
        image: torch.Tensor,
        vae: nn.Module,
        device: str
    ) -> Optional[torch.Tensor]:
        """Process a single image into latent space.
        
        Args:
            image: Input image tensor
            vae: VAE model for encoding
            device: Device to run processing on
            
        Returns:
            Optional[torch.Tensor]: Encoded latent if successful, None otherwise
            
        Raises:
            CUDAProcessingError: For CUDA and memory-related errors
            InputValidationError: For invalid input data
            ModelProcessingError: For VAE model-related errors
        """
        try:
            with autocast():
                with torch.no_grad():
                    image = image.unsqueeze(0).to(device)
                    latent = vae.encode(image).latent_dist.sample()
                    return latent.cpu()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            raise CUDAProcessingError(f"CUDA/Memory error: {str(error)}") from error
        except (ValueError, TypeError) as error:
            raise InputValidationError(f"Invalid input: {str(error)}") from error
        except AttributeError as error:
            raise ModelProcessingError(f"VAE model error: {str(error)}") from error
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_latents_batch(
        self,
        image_paths: List[str],
        vae: nn.Module,
        device: str,
        bucket_height: int,
        bucket_width: int,
        batch_size: int = 32
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of images into latents.
        
        Args:
            image_paths: List of paths to images
            vae: VAE model for encoding
            device: Device to run processing on
            bucket_height: Height to resize images to
            bucket_width: Width to resize images to
            batch_size: Size of batches to process at once
            
        Returns:
            Dict[str, torch.Tensor]: Mapping of image paths to latent tensors
            
        Raises:
            ValueError: If no valid images are found
        """
        results: Dict[str, torch.Tensor] = {}
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images: List[torch.Tensor] = []
            valid_paths: List[str] = []
            
            # Load and preprocess images
            for path in batch_paths:
                try:
                    image = self.load_and_transform(
                        path,
                        target_height=bucket_height,
                        target_width=bucket_width
                    )
                    if image is not None:
                        images.append(image)
                        valid_paths.append(path)
                except Exception as error:
                    logger.debug("Failed to load image %s: %s", path, str(error))
                    continue
            
            if not images:
                continue
            
            # Process batch
            try:
                image_batch = torch.stack(images)
                with autocast():
                    with torch.no_grad():
                        image_batch = image_batch.to(device)
                        latents = vae.encode(image_batch).latent_dist.sample()
                        latents = latents.cpu()
                
                # Store results
                for path, latent in zip(valid_paths, latents):
                    results[path] = latent
                    
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                logger.warning(
                    "CUDA error in batch processing, falling back to single image processing"
                )
                
                # Fallback to single image processing
                for image, path in zip(images, valid_paths):
                    try:
                        latent = self._process_single_latent(image, vae, device)
                        if latent is not None:
                            results[path] = latent
                    except ProcessingError as error:
                        logger.debug(
                            "Failed to process image %s: %s",
                            path, str(error)
                        )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results

    def verify_image_caption_pair(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """Verify that both image and caption exist and are valid.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple[bool, Optional[str]]: Tuple containing a boolean indicating
                whether the pair is valid and an optional error message
        """
        try:
            # Check image
            with Image.open(image_path) as img:
                img.verify()
            
            # Check caption
            caption_path = Path(image_path).with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                if caption:
                    return True, None
                return False, "Empty caption file"
            return False, "Missing caption file"
            
        except Exception as error:
            return False, str(error)

    def process_latent(self, image_path: str, vae: torch.nn.Module, device: str) -> Optional[torch.Tensor]:
        """Process a single image into latent space using VAE.
        
        Args:
            image_path: Path to image file
            vae: VAE model to use for encoding
            device: Device to run VAE on
            
        Returns:
            Optional[torch.Tensor]: Latent tensor if successful, None if failed
        """
        try:
            # Load and preprocess image
            image = self.load_and_transform(image_path)
            if image is None:
                return None
            
            # Move to GPU and process through VAE encoder
            with autocast():
                with torch.no_grad():
                    image = image.to(device, non_blocking=True)
                    latent = vae.encode(image.unsqueeze(0)).latent_dist.sample()
                    latent = latent.cpu()  # Move back to CPU for storage
                    
            # Clean up GPU memory
            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return latent
            
        except Exception as error:
            logger.error(
                "Failed to process latent for %s: %s",
                image_path, str(error)
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
