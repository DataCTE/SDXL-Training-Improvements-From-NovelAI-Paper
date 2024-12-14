import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Callable
from filelock import FileLock
import queue
import threading

from .buckets import AspectRatioBucket, ImageBucket
from models.embedder import TextEmbedder
from models.tag_weighter import FastTagWeighter, parse_tags
from utils.error_handling import error_handler
from diffusers import AutoencoderKL
from typing import Dict

class MultiDirectoryImageDataset(Dataset):
    """Dataset that handles multiple image directories with efficient loading"""
    
    def __init__(
        self,
        image_dirs: List[str],
        transform: Optional[Callable] = None,
        cache_dir: str = "latent_cache",
        text_cache_dir: str = "text_cache",
        min_size: int = 512,
        max_size: int = 2048,
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    ):
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.text_cache_dir = Path(text_cache_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratio_range = aspect_ratio_range
        
        # Create cache directories
        self.cache_dir.mkdir(exist_ok=True)
        self.text_cache_dir.mkdir(exist_ok=True)
        
        # Process all image directories
        self.image_files = []
        self.text_files = []
        self._process_directories(image_dirs)
        
    def _process_directories(self, image_dirs: List[str]):
        """Process multiple image directories efficiently"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for dir_path in image_dirs:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"Warning: Directory {dir_path} does not exist, skipping")
                continue
                
            print(f"Processing directory: {dir_path}")
            
            # Find all image files with valid extensions
            for ext in image_extensions:
                image_files = list(dir_path.rglob(f"*{ext}"))
                
                # Filter images by size and aspect ratio
                for img_path in tqdm(image_files, desc=f"Validating {ext} files"):
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            
                            # Check only size constraints if aspect_ratio_range is None
                            if self.aspect_ratio_range is None:
                                size_ok = (min(width, height) >= self.min_size and
                                         max(width, height) <= self.max_size)
                            else:
                                aspect_ratio = width / height
                                size_ok = (min(width, height) >= self.min_size and
                                         max(width, height) <= self.max_size and
                                         self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1])
                            
                            if size_ok:
                                # Look for corresponding text file
                                txt_path = img_path.with_suffix('.txt')
                                if txt_path.exists():
                                    self.image_files.append(img_path)
                                    self.text_files.append(txt_path)
                                    
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            print(f"Found {len(self.image_files)} valid images in {dir_path}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        img_path = self.image_files[idx]
        txt_path = self.text_files[idx]
        
        # Generate cache paths
        img_cache_path = self.cache_dir / f"{img_path.stem}.pt"
        text_cache_path = self.text_cache_dir / f"{img_path.stem}.pt"
        
        # Load or generate image tensor
        try:
            if img_cache_path.exists():
                img = torch.load(img_cache_path, map_location='cpu')
            else:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                torch.save(img, img_cache_path)
        except Exception:
            # Return a default/empty tensor silently
            img = torch.zeros((3, 256, 256))
            
        # Load or generate text embeddings
        try:
            if text_cache_path.exists():
                text_data = torch.load(text_cache_path, map_location='cpu')
                text_embeds = text_data.get('embeds', {})
                tag_weight = text_data.get('tag_weight', torch.tensor(1.0))
            else:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                text_embeds = self.text_embedder(caption) if hasattr(self, 'text_embedder') else {}
                tag_weight = torch.tensor(1.0)
                torch.save({'embeds': text_embeds, 'tag_weight': tag_weight}, text_cache_path)
        except Exception:
            # Return empty embeddings silently
            text_embeds = {}
            tag_weight = torch.tensor(1.0)
            
        return img, text_embeds, tag_weight
    
    def __len__(self) -> int:
        return len(self.image_files)

class NovelAIDataset(MultiDirectoryImageDataset):
    def __init__(
        self,
        image_dirs: List[str],
        transform: Optional[Callable] = None,
        device: torch.device = None,
        vae: Optional[AutoencoderKL] = None,
        cache_dir: str = "latent_cache",
        text_cache_dir: str = "text_cache"
    ):
        super().__init__(
            image_dirs=image_dirs,
            transform=transform,
            cache_dir=cache_dir,
            text_cache_dir=text_cache_dir
        )
        self.device = device
        self.vae = vae
        self.text_embedder = TextEmbedder(device=device)
        
        # Initialize optimized tag weighter
        self.tag_weighter = FastTagWeighter(
            min_weight=0.1,
            max_weight=2.0,
            default_weight=1.0,
            smoothing_factor=0.1,
            rarity_scale=0.5,
            cache_size=10000
        )
        
        # Pre-compute tag weights with optimized system
        print("Computing tag weights with optimized system...")
        self._precompute_tag_weights()
        
        # Print statistics after initialization
        self.tag_weighter.print_stats()
        
        # Add prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=3)
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, 
            daemon=True
        )
        self.prefetch_thread.start()
        
    def _precompute_tag_weights(self):
        """Pre-compute weights for all tags with optimized parsing"""
        for txt_path in tqdm(self.text_files, desc="Processing tags"):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                tags = parse_tags(caption)
                self.tag_weighter.update_frequencies(tags)
            except Exception:
                # Skip silently
                continue

    def _get_default_text_embeds(self) -> Dict[str, torch.Tensor]:
        """Create default text embeddings with correct structure"""
        return {
            'base_text_embeds': torch.zeros((1, 77, 768)),  # [batch, seq_len, hidden_dim]
            'base_pooled_embeds': torch.zeros((1, 768)),    # [batch, hidden_dim]
        }

    def _prefetch_worker(self):
        """Background worker to prefetch next batches"""
        while True:
            idx = self.prefetch_queue.get()
            if idx is None:
                break
            # Prefetch and cache next batch
            self._prefetch_item(idx)