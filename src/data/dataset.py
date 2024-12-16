from typing import List, Optional, Tuple, Callable, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import os
from PIL import Image
import logging
from tqdm.auto import tqdm
from torchvision import transforms
from dataclasses import dataclass

from src.data.text_embedder import TextEmbedder
from src.data.tag_weighter import TagWeighter, parse_tags
from src.data.bucket import AspectRatioBucket, ImageBucket

logger = logging.getLogger(__name__)

@dataclass
class NovelAIDatasetConfig:
    """Configuration for NovelAI dataset."""
    image_size: Tuple[int, int] = (1024, 1024)
    min_size: Tuple[int, int] = (256, 256)
    max_dim: int = 1024
    bucket_step: int = 64
    min_bucket_size: int = 1
    bucket_tolerance: float = 0.2
    max_aspect_ratio: float = 3.0
    cache_dir: str = "cache"
    use_caching: bool = True
    proportion_empty_prompts: float = 0.0

class NovelAIDataset(Dataset):
    def __init__(
        self,
        image_dirs: List[str],
        text_embedder: TextEmbedder,
        tag_weighter: TagWeighter,
        vae,  # AutoencoderKL
        config: NovelAIDatasetConfig,
        transform: Optional[Callable] = None,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize NovelAI dataset.
        
        Args:
            image_dirs: List of directories containing images and captions
            text_embedder: Text embedder instance
            tag_weighter: Tag weighter instance
            vae: VAE model instance
            config: Dataset configuration
            transform: Optional transform for images
            device: Torch device
        """
        self.config = config
        self.transform = transform or self._get_default_transforms()
        self.device = device
        self.text_embedder = text_embedder
        self.tag_weighter = tag_weighter
        self.vae = vae
        
        # Get model dtype from VAE
        self.dtype = next(vae.parameters()).dtype

        # Setup cache directories
        self.cache_dir = Path(config.cache_dir)
        self.latent_cache = self.cache_dir / "latents"
        self.text_cache = self.cache_dir / "text"
        
        for cache_dir in [self.latent_cache, self.text_cache]:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize bucketing
        self.bucket_manager = AspectRatioBucket(
            max_image_size=config.image_size,
            min_image_size=config.min_size,
            max_dim=config.max_dim,
            bucket_step=config.bucket_step
        )

        # Process and cache data
        self.items = []
        self._process_data(image_dirs)
        
        logger.info(f"Initialized dataset with {len(self)} samples in {len(self.bucket_manager.buckets)} buckets")

    def _get_default_transforms(self) -> Callable:
        """Get default image transforms."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: x.to(self.dtype))  # Convert to model dtype
        ])

    def _process_data(self, image_dirs: List[str]) -> None:
        """Process and cache dataset with optimized GPU batching."""
        total_found = 0
        total_processed = 0
        batch_size = 32  # Adjust based on GPU memory
        
        # Collect all valid image paths first
        image_files = []
        for image_dir in image_dirs:
            logger.info(f"Processing directory: {image_dir}")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        # Pre-filter valid images with text files
        valid_images = []
        for img_path in image_files:
            if Path(img_path).with_suffix('.txt').exists():
                valid_images.append(img_path)
        
        total_found = len(valid_images)
        
        # Process in optimized batches
        for i in tqdm(range(0, len(valid_images), batch_size), desc="Processing batches"):
            batch_paths = valid_images[i:i + batch_size]
            batch_images = []
            batch_buckets = []
            batch_latent_paths = []
            batch_text_paths = []
            
            # Prepare batch data
            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        width, height = img.size
                        
                        bucket = self.bucket_manager.find_bucket(width, height)
                        if bucket is None:
                            continue
                            
                        latent_cache_path = self.latent_cache / f"{Path(img_path).stem}.pt"
                        text_cache_path = self.text_cache / f"{Path(img_path).stem}.pt"
                        
                        # Only process if caching is enabled and file doesn't exist
                        if self.config.use_caching and not latent_cache_path.exists():
                            processed_img = self._process_image(img, bucket)
                            batch_images.append(processed_img)
                            batch_buckets.append(bucket)
                            batch_latent_paths.append(latent_cache_path)
                            batch_text_paths.append(text_cache_path)
                        
                        # Store item info
                        self.items.append({
                            'image_path': img_path,
                            'bucket': bucket,
                            'latent_cache': latent_cache_path,
                            'text_cache': text_cache_path,
                            'original_size': (height, width),
                            'crop_top_left': (0, 0)
                        })
                        total_processed += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
            
            # Process VAE in batches if we have images to process
            if batch_images:
                # Stack images into single tensor
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Process through VAE efficiently
                with torch.no_grad(), torch.cuda.amp.autocast():
                    latents = self.vae.encode(batch_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Save latents efficiently
                for latent, path in zip(latents, batch_latent_paths):
                    torch.save(latent.cpu(), path)
                
                # Process text embeddings in same batch
                text_batch = []
                for img_path, text_path in zip(batch_paths, batch_text_paths):
                    if not text_path.exists():
                        with open(Path(img_path).with_suffix('.txt'), 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        text_batch.append(caption)
                
                if text_batch:
                    text_embeds = self.text_embedder.encode_prompt_list(
                        text_batch,
                        proportion_empty_prompts=self.config.proportion_empty_prompts
                    )
                    tags = [parse_tags(txt) for txt in text_batch]
                    
                    # Save text embeddings
                    for i, path in enumerate(batch_text_paths):
                        if not path.exists():
                            torch.save({
                                'embeds': text_embeds['prompt_embeds'][i],
                                'tags': tags[i],
                            }, path)
                
                # Clear GPU memory
                del batch_tensor, latents
                torch.cuda.empty_cache()

        logger.info(f"Total images found: {total_found}")
        logger.info(f"Total images processed: {total_processed}")

    def _process_image(self, img: Image.Image, bucket: ImageBucket) -> torch.Tensor:
        """Process image to match bucket size."""
        # Calculate scaling
        width, height = img.size
        scale = min(bucket.width / width, bucket.height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize and center crop
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        left = (new_width - bucket.width) // 2
        top = (new_height - bucket.height) // 2
        right = left + bucket.width
        bottom = top + bucket.height
        
        img = img.crop((left, top, right, bottom))
        
        # Apply transforms and ensure correct dtype
        img = self.transform(img)
        
        return img.to(self.dtype)

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item with cached data."""
        item = self.items[idx]
        
        # Load cached latent
        latent = torch.load(item['latent_cache'])
        if len(latent.shape) == 4:
            latent = latent.squeeze(0)
        
        # Load cached text data
        text_data = torch.load(item['text_cache'])
        
        # Get tag weight
        tag_weight = torch.tensor(self.tag_weighter.get_weight(text_data['tags']))
        
        return {
            'model_input': latent,
            'text_embeds': text_data['embeds'],
            'tag_weights': tag_weight,
            'original_sizes': item['original_size'],
            'crop_top_lefts': item['crop_top_left'],
            'target_sizes': (item['bucket'].height, item['bucket'].width)
        }

    def __len__(self) -> int:
        return len(self.items)