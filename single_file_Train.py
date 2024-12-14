import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from PIL import Image
from pathlib import Path
import random
from dataclasses import dataclass
from typing import List, Optional
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Dict, List, Optional, Tuple, Callable
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
from accelerate import Accelerator
from adamw_bf16 import AdamWBF16
import glob
import os
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
import shutil
import argparse
from safetensors.torch import load_file
import signal
import sys


@dataclass
class ImageBucket:
    width: int
    height: int
    items: List = None
    
    def __post_init__(self):
        self.aspect_ratio = self.width / self.height
        if self.items is None:
            self.items = []

class AspectRatioBucket:
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (768, 1024),
        max_dim: int = 1024,
        bucket_step: int = 64
    ):
        self.max_width, self.max_height = max_image_size
        self.max_dim = max_dim
        self.bucket_step = bucket_step
        self.buckets: List[ImageBucket] = []
        self._generate_buckets()
        
    def _generate_buckets(self):
        """Generate bucket resolutions following section 4.1.2"""
        # Generate width-first buckets
        width = 256
        while width <= self.max_dim:
            # Find largest height that satisfies constraints
            height = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / width)
            )
            self.buckets.append(ImageBucket(
                width=width,
                height=height
            ))
            width += self.bucket_step
            
        # Generate height-first buckets
        height = 256
        while height <= self.max_dim:
            width = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / height)
            )
            # Skip if bucket already exists
            if not any(b.width == width and b.height == height for b in self.buckets):
                self.buckets.append(ImageBucket(
                    width=width,
                    height=height
                ))
            height += self.bucket_step
            
        # Add standard square bucket
        if not any(b.width == 1024 and b.height == 1024 for b in self.buckets):
            self.buckets.append(ImageBucket(
                width=1024,
                height=1024
            ))

    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best fitting bucket for given image dimensions"""
        image_aspect = width / height
        log_aspects = np.log([b.aspect_ratio for b in self.buckets])
        log_image_aspect = np.log(image_aspect)
        
        # Find closest bucket in log-space
        idx = np.argmin(np.abs(log_aspects - log_image_aspect))
        return self.buckets[idx]

class TextEmbedder:
    def __init__(
        self,
        device: torch.device,
        tokenizer_paths: Dict[str, str] = {
            "base": "openai/clip-vit-large-patch14",
            "large": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        }
    ):
        self.device = device
        self.max_length = 77
        
        # Load tokenizers (remove subfolder paths)
        self.tokenizers = {
            "base": CLIPTokenizer.from_pretrained(tokenizer_paths["base"]),
            "large": CLIPTokenizer.from_pretrained(tokenizer_paths["large"])
        }
        
        # Load text encoders with bfloat16 (remove subfolder paths)
        self.text_encoders = {
            "base": CLIPTextModel.from_pretrained(tokenizer_paths["base"]).to(device).to(torch.bfloat16),
            "large": CLIPTextModel.from_pretrained(tokenizer_paths["large"]).to(device).to(torch.bfloat16)
        }
        
        for encoder in self.text_encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def __call__(self, prompt: str) -> Dict[str, torch.Tensor]:
        # Tokenize
        tokens = {
            k: tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            for k, tokenizer in self.tokenizers.items()
        }
        
        # Generate embeddings
        embeds = {}
        for k, encoder in self.text_encoders.items():
            # Move tokens to GPU but keep as int64 for embedding layer
            tokens_gpu = {key: val.to(self.device) for key, val in tokens[k].items()}
            
            # Generate embeddings
            output = encoder(**tokens_gpu)
            
            # Ensure text embeddings have shape [batch_size, seq_len, hidden_dim]
            last_hidden_state = output.last_hidden_state
            if last_hidden_state.dim() == 2:
                last_hidden_state = last_hidden_state.unsqueeze(0)
            
            # Ensure pooled embeddings have shape [batch_size, hidden_dim]
            pooled_output = output.pooler_output
            if pooled_output.dim() == 1:
                pooled_output = pooled_output.unsqueeze(0)
            elif pooled_output.dim() == 3:
                pooled_output = pooled_output.squeeze(1)
            
            # Keep embeddings on CPU with consistent dimensions
            embeds[f"{k}_text_embeds"] = last_hidden_state.cpu()  # [batch_size, seq_len, hidden_dim]
            embeds[f"{k}_pooled_embeds"] = pooled_output.cpu()   # [batch_size, hidden_dim]
            
        return embeds

class TagWeighter:
    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        default_weight: float = 1.0
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.default_weight = default_weight
        
        # Initialize empty dictionaries for dynamic tag tracking
        self.tag_counts = {}
        self.total_count = 0
        self.tag_weights = {}

    def update_frequencies(self, tags: List[str]):
        """Update tag frequency counters"""
        for tag in tags:
            if tag not in self.tag_counts:
                self.tag_counts[tag] = 0
            self.tag_counts[tag] += 1
            self.total_count += 1
            
    def compute_weights(self):
        """Compute weights for all seen tags"""
        if not self.total_count:
            return
            
        # Calculate average frequency
        avg_freq = self.total_count / len(self.tag_counts) if self.tag_counts else 1.0
        
        # Compute weights for each tag
        for tag, count in self.tag_counts.items():
            # More common tags get lower weights
            raw_weight = avg_freq / count
            
            # Clamp weight to allowed range
            weight = min(self.max_weight, max(self.min_weight, raw_weight))
            self.tag_weights[tag] = weight
                
    def get_weight(self, tags: List[str]) -> float:
        """Get combined weight for a set of tags"""
        if not tags:
            return self.default_weight
            
        weights = [self.tag_weights.get(tag, self.default_weight) for tag in tags]
        # Use geometric mean to combine weights
        return torch.tensor(weights).mean().item()

def parse_tags(caption: str) -> List[str]:
    """Extract tags from caption"""
    # Convert to lowercase and split on commas
    parts = caption.lower().split(',')
    # Clean each tag
    tags = [tag.strip() for tag in parts]
    return tags

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
        vae: Optional[AutoencoderKL] = None
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
        
        # Unload text encoders and clear CUDA cache
        if self.text_embedder is not None:
            # Delete text encoders
            for encoder in self.text_embedder.text_encoders.values():
                del encoder
            self.text_embedder.text_encoders.clear()
            
            # Delete tokenizers
            for tokenizer in self.text_embedder.tokenizers.values():
                del tokenizer
            self.text_embedder.tokenizers.clear()
            
            # Delete the text embedder itself
            del self.text_embedder
            self.text_embedder = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _process_and_cache_data(self, image_dirs, max_image_size, max_dim, bucket_step, min_bucket_size):
        # Initialize buckets - keep the AspectRatioBucket instance, not just its buckets list
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
                
                # Generate cache paths
                img_cache_path = self.cache_dir / f"{Path(img_path).stem}.pt"
                text_cache_path = self.text_cache_dir / f"{Path(img_path).stem}.pt"
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        # Use the bucket_manager instance to find the best bucket
                        best_bucket = bucket_manager.find_bucket(width, height)
                        
                        if best_bucket is not None:
                            # Cache image latents if needed
                            if not img_cache_path.exists():
                                if self.vae is not None:
                                    processed_img = self._process_image(img_path, best_bucket)
                                    with torch.no_grad():
                                        latent = self.vae.encode(
                                            processed_img.unsqueeze(0).to(self.device)
                                        ).latent_dist.sample()
                                        latent = latent * 0.13025
                                        torch.save(latent.cpu(), img_cache_path)
                                    dir_cached_latents += 1
                            
                            # Cache text embeddings if needed
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
        """Load and process image to fit bucket"""
        with Image.open(image_path) as img:
            # Convert to RGB first
            img = img.convert('RGB')
            
            # Calculate scaling to fit bucket while preserving aspect ratio
            width, height = img.size
            scale = min(
                bucket.width / width,
                bucket.height / height
            )
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center crop to bucket size
            left = (new_width - bucket.width) // 2
            top = (new_height - bucket.height) // 2
            right = left + bucket.width
            bottom = top + bucket.height
            
            # Ensure we don't exceed image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(new_width, right)
            bottom = min(new_height, bottom)
            
            img = img.crop((left, top, right, bottom))
            
            # Pad if necessary to reach exact bucket dimensions
            if img.size != (bucket.width, bucket.height):
                new_img = Image.new('RGB', (bucket.width, bucket.height))
                paste_left = (bucket.width - img.size[0]) // 2
                paste_top = (bucket.height - img.size[1]) // 2
                new_img.paste(img, (paste_left, paste_top))
                img = new_img
            
            # Convert to tensor and apply transforms
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform if none provided
                img = transforms.ToTensor()(img)
                img = img[:3]  # Ensure 3 channels
                img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
                img = img.to(torch.bfloat16)
            
            return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single item from the dataset"""
        img_path, bucket, img_cache_path, text_cache_path = self.items[idx]
        
        # Load cached latent
        try:
            img = torch.load(img_cache_path, weights_only=True)
            if len(img.shape) == 4:
                img = img.squeeze(0)
        except Exception as e:
            print(f"Error loading cached latent {img_cache_path}: {e}")
            raise e
        
        # Load cached text embeddings
        try:
            cached_text = torch.load(text_cache_path, weights_only=True)
            text_embeds = cached_text['embeds']
            tags = cached_text['tags']
        except Exception as e:
            print(f"Error loading cached text {text_cache_path}: {e}")
            raise e
        
        # Get tag weights
        tag_weight = torch.tensor(self.tag_weighter.get_weight(tags))
        
        return img, text_embeds, tag_weight

    def __len__(self) -> int:
        return len(self.items)

class AspectBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by exact bucket dimensions
        self.groups = {}
        for idx, (_, bucket, img_cache_path, _) in enumerate(dataset.items):
            key = (bucket.width, bucket.height)
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)
        
        # Create batches of exactly matching dimensions
        self.batches = []
        for indices in self.groups.values():
            # Sort indices by actual image dimensions to ensure exact matches
            sorted_indices = []
            for idx in indices:
                img_path, _, _, _ = dataset.items[idx]
                with Image.open(img_path) as img:
                    width, height = img.size
                    sorted_indices.append((idx, (width, height)))
            
            # Group by exact dimensions
            exact_groups = {}
            for idx, dims in sorted_indices:
                if dims not in exact_groups:
                    exact_groups[dims] = []
                exact_groups[dims].append(idx)
            
            # Create batches from each exact group
            for exact_indices in exact_groups.values():
                for i in range(0, len(exact_indices), self.batch_size):
                    batch = exact_indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size:  # Only use full batches
                        self.batches.append(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)



class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        gradient_accumulation_steps: int = 4,
    ):
        super().__init__()
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        
        # Add gradient accumulation steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Add projection layer for CLIP embeddings
        self.hidden_proj = nn.Linear(768, model.config.cross_attention_dim).to(
            device=device, 
            dtype=torch.float32
        )
        
        # Pre-allocate tensors for time embeddings
        self.register_buffer('base_area', torch.tensor(1024 * 1024, dtype=torch.float32))
        self.register_buffer('aesthetic_score', torch.tensor(6.0, dtype=torch.bfloat16))
        self.register_buffer('crop_score', torch.tensor(3.0, dtype=torch.bfloat16))
        self.register_buffer('zero_score', torch.tensor(0.0, dtype=torch.bfloat16))
        
        # Initialize ZTSNR parameters
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = 20000.0
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Enable memory efficient attention
        model.enable_xformers_memory_efficient_attention()
        
        # Enable channels last memory format
        model = model.to(memory_format=torch.channels_last)
        
        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Disable debug APIs
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            print("No training state found, starting from scratch with pretrained weights")

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized time_ids computation for H100"""
        batch_size = images.shape[0]
        orig_height = images.shape[2] * 8
        orig_width = images.shape[3] * 8
        
        add_time_ids = torch.empty((batch_size, 2, 4), device=self.device, dtype=torch.bfloat16)
        add_time_ids[:, 0, 0] = orig_height
        add_time_ids[:, 0, 1] = orig_width
        add_time_ids[:, 0, 2] = self.aesthetic_score
        add_time_ids[:, 0, 3] = self.zero_score
        add_time_ids[:, 1, 0] = orig_height
        add_time_ids[:, 1, 1] = orig_width
        add_time_ids[:, 1, 2] = self.crop_score
        add_time_ids[:, 1, 3] = self.zero_score
        
        return add_time_ids.reshape(batch_size, -1)

    def get_karras_scalings(self, sigma, sigma_data=1.0):
        # sigma: [batch_size] tensor
        sigma_sq = sigma * sigma
        sigma_data_sq = sigma_data * sigma_data
        denominator = sigma_data_sq + sigma_sq
        c_skip = sigma_data_sq / denominator
        c_out = -sigma_data * sigma / torch.sqrt(denominator)
        c_in = 1.0 / torch.sqrt(denominator)
        return c_skip, c_out, c_in

    @torch.no_grad()
    def get_velocity(scheduler: DDPMScheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        # Computes the v_prediction target as described in the paper.
        # v = (x - x_0) / sqrt(sigma^2 + sigma_data^2)
        return scheduler.get_velocity(latents, noise, timesteps)

    def get_sigmas(self) -> torch.Tensor:
            """Generate noise schedule for ZTSNR with optimized scaling.
            
            Uses a modified ramp function to ensure:
            1. First step has σ = σ_min (0.002)
            2. Last step has σ = σ_max (20000) as practical infinity
            3. Intermediate steps follow power-law scaling with ρ=7
            
            Returns:
                torch.Tensor: Noise schedule sigmas of shape [num_timesteps]
            """
            # Generate ramp on device directly
            ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
            
            # Compute inverse rho values
            min_inv_rho = self.sigma_min ** (1/self.rho)
            max_inv_rho = self.sigma_max ** (1/self.rho)
            
            # Generate full schedule with vectorized operations
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            
            # Ensure exact values at endpoints
            sigmas[0] = self.sigma_min
            sigmas[-1] = self.sigma_max
            
            return sigmas

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        # SNR(σ) = (σ_data / σ)²
        return (self.sigma_data / sigma).square()

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Uses the scheduler's alpha values to compute SNR(t) = α/(1-α)
        # w(t) = min(SNR(t), γ)
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]  # [batch_size]
        
        # SNR in terms of alpha: SNR = α/(1-α)
        snr = alpha_t / (1 - alpha_t)
        
        # Clamp to min_snr_gamma
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr, min_snr).float()
        return weights

    def training_step(
        self,
        images: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        images = images.to(self.device)  # [B,4,H,W]
        text_embeds = {k: v.to(self.device) for k,v in text_embeds.items()}
        tag_weights = tag_weights.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        total_loss = 0.0
        total_v_pred = None
        running_loss = 0.0

        for i in range(self.gradient_accumulation_steps):
            torch.cuda.empty_cache()

            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size

            # Extract micro-batch
            batch_latents = images[start_idx:end_idx]  # [mB,4,H,W]

            # Apply area-based noise scaling
            height = batch_latents.shape[2]
            width = batch_latents.shape[3]
            area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
            noise_scale = torch.sqrt(area / self.base_area)
            batch_latents = batch_latents * noise_scale

            batch_tag_weights = tag_weights[start_idx:end_idx].view(-1,1,1,1)  # [mB,1,1,1]

            # Prepare text embeddings - clone and detach to ensure proper gradient flow
            base_hidden = text_embeds["base_text_embeds"][start_idx:end_idx].squeeze(1).clone()  # [mB, seq_len, 768]
            base_pooled = text_embeds["base_pooled_embeds"][start_idx:end_idx].squeeze(1).clone()  # [mB, 768]

            # Project text embeddings
            base_hidden_float32 = base_hidden.to(dtype=torch.float32)
            batch_size, seq_len, _ = base_hidden_float32.shape
            encoder_hidden_states = self.hidden_proj(
                base_hidden_float32.view(-1, 768)
            ).view(batch_size, seq_len, -1)

            # Sample noise and timesteps
            noise = torch.randn_like(batch_latents)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_latents.size(0),), 
                device=self.device
            ).long()  # Ensure long dtype

            sigmas = self.sigmas[timesteps]

            # Add noise
            noisy_latents = batch_latents + sigmas.view(-1,1,1,1)*noise

            # Compute v_target
            v_target = self.scheduler.get_velocity(batch_latents, noise, timesteps)  # [mB,4,H,W]

            # Generate time_ids and ensure they're cloned for gradient checkpointing
            time_ids = self._get_add_time_ids(batch_latents).clone() # [mB,8]

            # Karras scaling
            c_skip, c_out, c_in = self.get_karras_scalings(sigmas)

            # Ensure input tensors are cloned and have gradients enabled
            scaled_input = (c_in.view(-1,1,1,1)*noisy_latents).clone().requires_grad_(True)
            encoder_hidden_states = encoder_hidden_states.clone().requires_grad_(True)
            timesteps = timesteps.clone()

            # Forward pass with autocast
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                F_out = self.model(
                    scaled_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": base_pooled.clone(),
                        "time_ids": time_ids
                    }
                ).sample

                # D_out = c_skip*x + c_out*F_out
                D_out = c_skip.view(-1,1,1,1)*noisy_latents + c_out.view(-1,1,1,1)*F_out

                # Compute MSE loss per-sample
                loss_per_sample = F.mse_loss(D_out.float(), v_target.float(), reduction='none')  # [mB,4,H,W]
                loss_per_sample = loss_per_sample.mean(dim=[1,2,3])  # [mB]

                # Get SNR weights
                snr_weights = self.get_minsnr_weights(timesteps)  # [mB]

                # Apply tag weights and SNR weights
                loss_per_sample = loss_per_sample * batch_tag_weights.squeeze() * snr_weights

                # Average over micro-batch
                loss = loss_per_sample.mean() / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            
            # Convert to float for tracking
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value

            total_v_pred = D_out.detach() if total_v_pred is None else torch.cat([total_v_pred, D_out.detach()], dim=0)

        avg_loss = running_loss / self.gradient_accumulation_steps
        return total_loss, batch_latents, total_v_pred, timesteps, avg_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> float:
        self.current_epoch = epoch
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Clear cache at start of batch
            torch.cuda.empty_cache()
            
            # Unpack and move batch to device
            images, text_embeds, tag_weights = batch
            images = images.to(self.device, dtype=torch.float32)
            text_embeds = {k: v.to(self.device, dtype=torch.float32) 
                          for k, v in text_embeds.items()}
            tag_weights = tag_weights.to(self.device, dtype=torch.float32)
            
            # Training step
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, pred_images, v_pred, timesteps, avg_batch_loss = self.training_step(
                    images, text_embeds, tag_weights
                )
            
            # Compute gradient norm and update optimizer
            grad_norm = self.compute_grad_norm()
            self.optimizer.step()
            
            # Update metrics
            if self.accelerator.is_main_process:
                wandb.log({
                    'grad/norm': grad_norm,
                    'loss/batch': avg_batch_loss,
                    'loss/running_avg': total_loss / (batch_idx + 1),
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                })
            
            # Update total loss
            total_loss += avg_batch_loss
            
            # Console logging
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch {epoch} [{batch_idx}/{num_batches}]:')
                print(f'  Batch Loss = {avg_batch_loss:.4f}')
                print(f'  Avg Loss = {avg_loss:.4f}')
                print(f'  Grad norm = {grad_norm:.4f}')
                print(f'  Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB')
            
            self.global_step += 1
        
        # Return average loss for the epoch
        return total_loss / num_batches

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ) -> DataLoader:
        sampler = AspectBatchSampler(dataset, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

    def save_checkpoint(self, save_path: str):
        """Save model checkpoint with only modified components in fp16 format"""
        if self.accelerator is not None:
            # Unwrap model if using accelerator
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        # Create checkpoint directory
        os.makedirs(save_path, exist_ok=True)

        # Temporarily convert model to float16 for saving
        original_dtype = unwrapped_model.dtype
        unwrapped_model = unwrapped_model.to(torch.float16)

        try:
            # Save UNet weights in fp16
            unwrapped_model.save_pretrained(
                os.path.join(save_path, "unet"),
                safe_serialization=True  # Use safetensors format
            )
        finally:
            # Convert back to original dtype
            unwrapped_model = unwrapped_model.to(original_dtype)

        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

        print(f"Saved checkpoint to {save_path} in fp16 format")

    def compute_grad_norm(self):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def log_detailed_metrics(self,
                            loss: torch.Tensor,
                            v_pred: torch.Tensor,
                            grad_norm: float,
                            timesteps: torch.Tensor):
        """Log detailed training metrics to W&B"""
        if not self.accelerator.is_main_process:
            return
        
        # Compute v-prediction statistics
        v_pred_mean = v_pred.mean().item()
        v_pred_std = v_pred.std().item()
        v_pred_min = v_pred.min().item()
        v_pred_max = v_pred.max().item()
        
        # Log detailed metrics
        wandb.log({
            'loss/total': loss.item(),
            'v_pred/mean': v_pred_mean,
            'v_pred/std': v_pred_std,
            'v_pred/min': v_pred_min,
            'v_pred/max': v_pred_max,
            'grad/norm': grad_norm,
            'timesteps/mean': timesteps.float().mean().item(),
            'timesteps/std': timesteps.float().std().item()
        })

def ensure_three_channels(x):
    return x[:3]

def convert_to_bfloat16(x):
    return x.to(torch.bfloat16)


def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint directory to resume from")
    parser.add_argument("--unet_path", type=str, help="Path to UNet safetensors file to start from")
    args = parser.parse_args()

    # Create a variable to hold our trainer instance
    trainer = None

    def signal_handler(signum, frame):
        """Handle interrupt signals by saving checkpoint before exit"""
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name} signal. Attempting to save checkpoint...")
        
        if trainer is not None:
            try:
                emergency_save_path = os.path.join("checkpoints", "emergency_checkpoint")
                trainer.save_checkpoint(emergency_save_path)
                print("Emergency checkpoint saved successfully.")
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")
        
        print("Exiting...")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # termination request

    # Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="sdxl-finetune",
        config={
            "batch_size": 32,
            "grad_accum_steps": 4,
            "effective_batch": 128,
            "learning_rate": 4e-7,
            "num_epochs": 10,
            "model": "SDXL-base-1.0",
            "optimizer": "AdamW-BF16",
            "scheduler": "DDPM",
            "min_snr_gamma": 0.1,
        }
    )
    
    # Dataset paths
    image_dirs = [
        r"/workspace/collage",
        r"/workspace/upscaled",
        r"/workspace/High-quality-photo10k",
        r"/workspace/LAION_220k_GPT4Vision_captions",
        r"/workspace/photo-concept-bucket/train"
    ]
    
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("Loading UNet...")
    if args.resume_from_checkpoint:
        print(f"Loading UNet from checkpoint directory: {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            args.resume_from_checkpoint,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    elif args.unet_path:
        print(f"Loading UNet from safetensors file: {args.unet_path}")
        # Get the directory containing the safetensors file
        unet_dir = os.path.dirname(args.unet_path)
        config_path = os.path.join(unet_dir, "config.json")
        
        if os.path.exists(config_path):
            # If config exists in same directory as safetensors, load from directory
            print(f"Found config.json in same directory, loading from: {unet_dir}")
            unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(device)
        else:
            # If no config found, load from base model first then load weights
            print("No config.json found, loading architecture from base model")
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            ).to(device)
            # Load weights from safetensors
            state_dict = load_file(args.unet_path)
            unet.load_state_dict(state_dict)
    else:
        print("Loading fresh UNet from pretrained model")
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)

    # Enable gradient checkpointing for more memory savings
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()
    # Setup transform without lambdas
    transform = transforms.Compose([
        transforms.ToTensor(),
        ensure_three_channels,
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        convert_to_bfloat16
    ])
    
    print("Creating dataset...")
    dataset = NovelAIDataset(
        image_dirs=image_dirs,
        transform=transform,
        device=device,
        vae=vae,
        cache_dir="latent_cache",
        text_cache_dir="text_cache"
    )
    
    print("Creating dataloader...")
    dataloader = NovelAIDiffusionV3Trainer.create_dataloader(
        dataset=dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print("Setting up optimizer...")
    optimizer = AdamWBF16(
        unet.parameters(),
        lr=4e-7,  # For effective batch size of 128 (32 * 4)
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    print("Setting up noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name,
        subfolder="scheduler",
        torch_dtype=torch.bfloat16
    )
    
    print("Setting up accelerator...")
    accelerator = Accelerator(
        gradient_accumulation_steps=4,  # Accumulate over 4 steps for effective batch size of 128
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="logs",
        device_placement=True,
        cpu=False,
    )

    # Initialize tracking for TensorBoard
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="sdxl_finetune",
            config={
                "train_batch_size": 32,
                "gradient_accumulation_steps": 4,  # Updated to 4 steps
                "effective_batch_size": 128,  # Updated effective batch size (32 * 4)
                "learning_rate": 4e-7,  # Updated learning rate
                "num_epochs": 10,
            }
        )
    
    # Move everything to device and prepare for accelerated training
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )
    
    print("Creating trainer...")
    trainer = NovelAIDiffusionV3Trainer(
        model=unet,
        vae=vae,
        optimizer=optimizer,
        scheduler=noise_scheduler,
        device=device,
        accelerator=accelerator,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=4
    )
    
    # Adjust starting epoch if resuming
    start_epoch = trainer.current_epoch
    print(f"Starting training from epoch {start_epoch + 1}")
    
    # Training configuration
    num_epochs = 10  # Adjust as needed
    save_interval = 1000  # Save every 1000 steps
    log_interval = 10  # Log every 10 steps
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nStarting training...")
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            try:
                epoch_loss = trainer.train_epoch(
                    dataloader=dataloader,
                    epoch=epoch,
                    log_interval=log_interval
                )
                
                # Log metrics to TensorBoard
                if accelerator.is_main_process:
                    try:
                        accelerator.log(
                            {
                                "train/loss": epoch_loss,
                                "train/epoch": epoch,
                                "train/step": trainer.global_step,
                            },
                            step=trainer.global_step,
                        )
                    except Exception as e:
                        print(f"Error in TensorBoard logging: {e}")
                
                # Save checkpoint at the end of each epoch and periodically by steps
                try:
                    # Save epoch checkpoint
                    epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                    print(f"Saving epoch checkpoint to {epoch_checkpoint_path}")
                    trainer.save_checkpoint(epoch_checkpoint_path)
                    
                    # Save step checkpoint if interval is reached
                    if trainer.global_step % save_interval == 0:
                        step_checkpoint_path = os.path.join(
                            save_dir, 
                            f"checkpoint_step_{trainer.global_step}"
                        )
                        print(f"Saving step checkpoint to {step_checkpoint_path}")
                        trainer.save_checkpoint(step_checkpoint_path)
                        
                        # Cleanup old step checkpoints (keep last 3)
                        try:
                            step_checkpoints = sorted([
                                f for f in os.listdir(save_dir) 
                                if f.startswith("checkpoint_step_")
                            ])
                            if len(step_checkpoints) > 3:
                                for old_ckpt in step_checkpoints[:-3]:
                                    old_path = os.path.join(save_dir, old_ckpt)
                                    print(f"Removing old checkpoint: {old_path}")
                                    shutil.rmtree(old_path)
                        except Exception as e:
                            print(f"Error cleaning up old checkpoints: {e}")
                            
                except Exception as e:
                    print(f"Error saving checkpoints: {e}")
                    print(f"Current GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                
                print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
                
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                print(f"Memory stats:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                raise
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        final_checkpoint_path = os.path.join(save_dir, "final_checkpoint")
        trainer.save_checkpoint(final_checkpoint_path)
        print("Final checkpoint saved.")
    else:
        # Training completed normally, save final checkpoint
        print("\nTraining completed. Saving final checkpoint...")
        final_checkpoint_path = os.path.join(save_dir, "final_checkpoint")
        trainer.save_checkpoint(final_checkpoint_path)
        print("Final checkpoint saved.")
    
    # Close wandb run
    wandb.finish()
    
    # Close accelerator
    accelerator.end_training()
    print("Training completed!")

if __name__ == "__main__":
    main()
