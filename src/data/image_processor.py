import logging
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image loading, transformation and validation."""
    
    def __init__(self):
        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def load_and_verify_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and verify an image file."""
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
                # Reopen because verify() invalidates the image
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
        except Exception as e:
            logger.debug(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def get_image_dimensions(self, img: Image.Image) -> Tuple[int, int]:
        """Get image dimensions (height, width)."""
        return img.size[1], img.size[0]
    
    def resize_image(self, img: Image.Image, target_height: int, target_width: int) -> Image.Image:
        """Resize image to target dimensions."""
        if img.size != (target_width, target_height):
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return img
    
    def transform_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        return self.image_transforms(img)
    
    def load_and_transform(self, image_path: str, target_height: Optional[int] = None, target_width: Optional[int] = None) -> Optional[torch.Tensor]:
        """Load image from path and transform it to tensor.
        
        Args:
            image_path: Path to the image file
            target_height: Optional target height for resizing
            target_width: Optional target width for resizing
            
        Returns:
            Transformed image tensor or None if loading fails
        """
        try:
            # Load and verify image
            img = self.load_and_verify_image(image_path)
            if img is None:
                return None
                
            # Resize if dimensions provided
            if target_height is not None and target_width is not None:
                img = self.resize_image(img, target_height, target_width)
                
            # Transform to tensor
            return self.transform_to_tensor(img)
            
        except Exception as e:
            logger.debug(f"Failed to transform image {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def verify_image_caption_pair(image_path: str) -> Tuple[bool, Optional[str]]:
        """Verify that both image and caption exist and are valid."""
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
            
        except Exception as e:
            return False, str(e)

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
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image = image.to(device, non_blocking=True)
                    latent = vae.encode(image.unsqueeze(0)).latent_dist.sample()
                    latent = latent.cpu()  # Move back to CPU for storage
                    
            # Clean up GPU memory
            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return latent
            
        except Exception as e:
            logger.error(f"Failed to process latent for {image_path}: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def process_latents_batch(self, image_paths: List[str], vae: torch.nn.Module, 
                            device: str, bucket_height: int, bucket_width: int,
                            batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Process a batch of images into latents, with proper bucketing and error handling.
        
        Args:
            image_paths: List of paths to images
            vae: VAE model to use for encoding
            device: Device to run VAE on
            bucket_height: Height to resize images to
            bucket_width: Width to resize images to
            batch_size: Size of batches to process at once
            
        Returns:
            Dict[str, torch.Tensor]: Mapping of image paths to their latent tensors
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    # Load and transform image
                    img = self.load_and_verify_image(path)
                    if img is None:
                        continue
                        
                    # Resize to bucket dimensions
                    img = self.resize_image(img, bucket_height, bucket_width)
                    
                    # Convert to tensor
                    tensor = self.transform_to_tensor(img)
                    images.append(tensor)
                    valid_paths.append(path)
                    
                except Exception as e:
                    logger.debug(f"Failed to process image {path}: {str(e)}")
                    continue
            
            if not images:
                continue
                
            # Stack images into a batch
            try:
                image_batch = torch.stack(images)
                
                # Process through VAE
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        image_batch = image_batch.to(device)
                        latents = vae.encode(image_batch).latent_dist.sample()
                        latents = latents.cpu()  # Move to CPU to save GPU memory
                
                # Store results
                for path, latent in zip(valid_paths, latents):
                    results[path] = latent
                    
            except RuntimeError as e:  # Handle CUDA out of memory
                logger.warning(f"CUDA error while processing batch: {str(e)}")
                logger.warning("Trying to process one image at a time...")
                
                # Fallback to processing one at a time
                for image, path in zip(images, valid_paths):
                    try:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                image = image.unsqueeze(0).to(device)
                                latent = vae.encode(image).latent_dist.sample()
                                latent = latent.cpu()
                        results[path] = latent
                    except Exception as e:
                        logger.debug(f"Failed to process single image {path}: {str(e)}")
                    finally:
                        # Aggressively clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
