import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Callable

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
        vae = None
    ):
        self.transform = transform
        self.device = device
        self.text_embedder = TextEmbedder(device=device)
        self.tag_weighter = TagWeighter()
        self.vae = vae
        
        # Setup cache directories
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.text_cache_dir = Path(text_cache_dir)
        self.text_cache_dir.mkdir(exist_ok=True)
        
        # Process directories and cache latents and text embeddings
        self._process_and_cache_data(
            image_dirs=image_dirs,
            max_image_size=max_image_size,
            max_dim=max_dim,
            bucket_step=bucket_step,
            min_bucket_size=min_bucket_size
        )
        
        # Cleanup embedder to free memory
        self._cleanup_embedder()

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
        bucket_manager = AspectRatioBucket(
            max_image_size=max_image_size,
            max_dim=max_dim,
            bucket_step=bucket_step
        )
        
        self.items = []
        total_found = 0
        total_processed = 0
        total_cached_latents = 0
        total_cached_text = 0
        total_skipped = 0
        
        for image_dir in image_dirs:
            print(f"\nProcessing directory: {image_dir}")
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff', '*.tif', '*.gif']:
                image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
                image_files.extend(glob.glob(os.path.join(image_dir, '**', ext.upper()), recursive=True))
            
            dir_found = len(image_files)
            total_found += dir_found
            print(f"Found {dir_found} images")
            
            dir_processed = 0
            dir_cached_latents = 0
            dir_cached_text = 0
            dir_skipped = 0
            
            for img_path in tqdm(image_files, desc="Loading images"):
                txt_path = img_path.replace(os.path.splitext(img_path)[1], '.txt')
                if not os.path.exists(txt_path):
                    continue
                
                img_cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
                text_cache_path = self.text_cache_dir / f"{Path(img_path).stem}.pt"
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        best_bucket = bucket_manager.find_bucket(width, height)
                        
                        if best_bucket is not None:
                            if not img_cache_path.exists() and self.vae is not None:
                                processed_img = self._process_image(img_path, best_bucket)
                                with torch.no_grad():
                                    latent = self.vae.encode(
                                        processed_img.unsqueeze(0).to(self.device)
                                    ).latent_dist.sample()
                                    latent = latent * 0.13025
                                    torch.save(latent.cpu(), img_cache_path)
                                dir_cached_latents += 1
                            
                            if not text_cache_path.exists():
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    caption = f.read().strip()
                                text_embeds = self.text_embedder(caption)
                                torch.save({
                                    'embeds': text_embeds,
                                    'tags': parse_tags(caption)
                                }, text_cache_path)
                                dir_cached_text += 1
                            
                            self.items.append((img_path, best_bucket, img_cache_path, text_cache_path))
                            dir_processed += 1
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            total_processed += dir_processed
            total_cached_latents += dir_cached_latents
            total_cached_text += dir_cached_text
            total_skipped += dir_skipped
            
            print(f"Successfully processed {dir_processed} images")
            print(f"  - {dir_cached_latents} new latents cached")
            print(f"  - {dir_cached_text} new text embeddings cached")
            print(f"  - {dir_skipped} existing items skipped")
        
        print(f"\nFinal Summary:")
        print(f"Total images found: {total_found}")
        print(f"Total images processed: {total_processed}")
        print(f"  - {total_cached_latents} new latents cached")
        print(f"  - {total_cached_text} new text embeddings cached")
        print(f"  - {total_skipped} existing items skipped")

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