from typing import List, Optional, Tuple, Callable, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import os
from PIL import Image
from tqdm import tqdm
from .bucket import AspectRatioBucket, ImageBucket
from .text_embedder import TextEmbedder
from .tag_weighter import TagWeighter, parse_tags
from diffusers import AutoencoderKL
from torchvision import transforms


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
        vae: Optional[AutoencoderKL] = None,
        config = None
    ):
        self.transform = transform
        self.device = device
        self.text_embedder = TextEmbedder(device=device)
        self.tag_weighter = TagWeighter(config)
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
        
        # Cleanup text embedder to free memory
        if self.text_embedder is not None:
            del self.text_embedder
            self.text_embedder = None
            torch.cuda.empty_cache()

    def _process_and_cache_data(self, image_dirs, max_image_size, max_dim, bucket_step, min_bucket_size):
        """Process and cache dataset with distributed awareness"""
        bucket_manager = AspectRatioBucket(
            max_image_size=max_image_size,
            max_dim=max_dim,
            bucket_step=bucket_step
        )
        
        self.items = []
        total_found = 0
        total_processed = 0
        
        # Get rank and world size for distributed processing
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        for image_dir in image_dirs:
            if rank == 0:
                print(f"\nProcessing directory: {image_dir}")
                
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
            
            # Sort for deterministic ordering across processes
            image_files.sort()
            
            # Distribute files across processes
            per_rank = len(image_files) // world_size
            start_idx = rank * per_rank
            end_idx = start_idx + per_rank if rank < world_size - 1 else len(image_files)
            rank_files = image_files[start_idx:end_idx]
            
            dir_found = len(rank_files)
            total_found += dir_found
            if rank == 0:
                print(f"Found {len(image_files)} total images, processing {dir_found} on rank {rank}")
            
            for img_path in tqdm(rank_files, desc=f"Processing images (rank {rank})", disable=rank != 0):
                txt_path = str(Path(img_path).with_suffix('.txt'))
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
                            
                            if not text_cache_path.exists():
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    caption = f.read().strip()
                                text_embeds = self.text_embedder(caption)
                                torch.save({
                                    'embeds': text_embeds,
                                    'tags': parse_tags(caption)
                                }, text_cache_path)
                            
                            self.items.append((img_path, best_bucket, img_cache_path, text_cache_path))
                            total_processed += 1
                
                except Exception as e:
                    if rank == 0:
                        print(f"Error processing {img_path}: {e}")
        
        # Gather statistics across processes
        if world_size > 1:
            total_found = torch.tensor(total_found, device=self.device)
            total_processed = torch.tensor(total_processed, device=self.device)
            
            torch.distributed.all_reduce(total_found, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_processed, op=torch.distributed.ReduceOp.SUM)
            
            total_found = total_found.item()
            total_processed = total_processed.item()
        
        if rank == 0:
            print(f"\nTotal images found across all processes: {total_found}")
            print(f"Total images processed across all processes: {total_processed}")
            
        # Sync processes before returning
        if world_size > 1:
            torch.distributed.barrier()

    def _process_image(self, image_path: str, bucket: ImageBucket) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            # Calculate scaling to fit bucket
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
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
                img = img[:3]
                img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
                img = img.to(torch.bfloat16)
            
            return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        img_path, bucket, img_cache_path, text_cache_path = self.items[idx]
        
        # Load cached latent
        img = torch.load(img_cache_path)
        if len(img.shape) == 4:
            img = img.squeeze(0)
        
        # Load cached text embeddings
        cached_text = torch.load(text_cache_path)
        text_embeds = cached_text['embeds']
        tags = cached_text['tags']
        
        # Get tag weights
        tag_weight = torch.tensor(self.tag_weighter.get_weight(tags))
        
        return img, text_embeds, tag_weight

    def __len__(self) -> int:
        return len(self.items) 