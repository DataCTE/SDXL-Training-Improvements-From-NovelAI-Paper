import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Callable
import hashlib
import torch.nn as nn

from .buckets import AspectRatioBucket, ImageBucket
from models.embedder import TextEmbedder
from models.tag_weighter import TagWeighter, parse_tags

class NovelAIDataset(Dataset):
    def __init__(
        self,
        image_dirs: List[str],
        transform: Optional[Callable] = None,
        device: torch.device = torch.device('cpu'),
        max_image_size: Tuple[int, int] = (768, 1024),
        max_dim: int = 1024,
        bucket_step: int = 64,
        min_bucket_size: int = 1,
        cache_dir: str = "latent_cache",
        text_cache_dir: str = "text_cache",
        vae = None,
        local_rank: int = -1,
        world_size: int = 1
    ):
        self.transform = transform
        self.device = device
        self.text_embedder = TextEmbedder(device=device)
        self.tag_weighter = TagWeighter()
        self.vae = vae
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Single cache directory for all processes
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.text_cache_dir = Path(text_cache_dir)
        self.text_cache_dir.mkdir(exist_ok=True)

    def _cleanup_embedder(self):
        """Clean up text embedder to free memory"""
        if self.text_embedder is not None:
            for encoder in self.text_embedder.text_encoders.values():
                del encoder
            self.text_embedder.text_encoders.clear()
            
            for tokenizer in self.text_embedder.tokenizers.values():
                del tokenizer
            self.text_embedder.tokenizers.clear()
            
            del self.text_embedder
            self.text_embedder = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _process_and_cache_data(self, image_dirs, max_image_size, max_dim, bucket_step, min_bucket_size):
        """Process all images and cache latents/embeddings"""
        # Collect all image files first
        all_image_files = []
        for image_dir in image_dirs:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                all_image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

        # Partition work across GPUs
        if self.world_size > 1:
            # Calculate partition for this GPU
            partition_size = len(all_image_files) // self.world_size
            start_idx = self.local_rank * partition_size
            end_idx = start_idx + partition_size if self.local_rank < self.world_size - 1 else len(all_image_files)
            # This GPU only processes its partition
            image_files = all_image_files[start_idx:end_idx]
            print(f"Rank {self.local_rank} processing {len(image_files)} images")
        else:
            image_files = all_image_files

        # Process assigned partition
        for img_path in tqdm(image_files, desc=f"Rank {self.local_rank} processing"):
            img_cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
            if not img_cache_path.exists():
                # Process and cache image
                self._cache_single_image(img_path, img_cache_path)

        # Wait for all processes to finish caching
        if self.world_size > 1:
            torch.distributed.barrier()

        # After caching, all processes load the full dataset
        self.items = []
        for img_path in all_image_files:
            img_cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
            if img_cache_path.exists():
                self.items.append((img_path, img_cache_path))

    def _process_image(self, image_path: str, bucket: ImageBucket) -> torch.Tensor:
        """Process a single image to fit in its bucket"""
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            width, height = img.size
            scale = min(bucket.width / width, bucket.height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            left = (new_width - bucket.width) // 2
            top = (new_height - bucket.height) // 2
            right = left + bucket.width
            bottom = top + bucket.height
            
            left = max(0, left)
            top = max(0, top)
            right = min(new_width, right)
            bottom = min(new_height, bottom)
            
            img = img.crop((left, top, right, bottom))
            
            if img.size != (bucket.width, bucket.height):
                new_img = Image.new('RGB', (bucket.width, bucket.height))
                paste_left = (bucket.width - img.size[0]) // 2
                paste_top = (bucket.height - img.size[1]) // 2
                new_img.paste(img, (paste_left, paste_top))
                img = new_img
            
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
                img = img[:3]
                img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
                img = img.to(torch.bfloat16)
            
            return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Get a single item from the dataset"""
        img_path, bucket, img_cache_path, text_cache_path = self.items[idx]
        
        try:
            img = torch.load(img_cache_path, weights_only=True)
            if len(img.shape) == 4:
                img = img.squeeze(0)
        except Exception as e:
            print(f"Error loading cached latent {img_cache_path}: {e}")
            raise e
        
        try:
            cached_text = torch.load(text_cache_path, weights_only=True)
            text_embeds = cached_text['embeds']
            tags = cached_text['tags']
        except Exception as e:
            print(f"Error loading cached text {text_cache_path}: {e}")
            raise e
        
        tag_weight = torch.tensor(self.tag_weighter.get_weight(tags))
        
        return img, text_embeds, tag_weight

    def __len__(self) -> int:
        return len(self.items) 