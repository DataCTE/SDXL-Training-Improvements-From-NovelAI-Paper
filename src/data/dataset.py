from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
from tqdm import tqdm
import logging
from transformers import CLIPModel, CLIPProcessor
from utils.device import to_device
import traceback
import os
import time
import glob

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache", batch_size=1):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size 

        # Store model references as instance variables
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        # Validate dataset before processing
        is_valid, stats = validate_dataset(data_dir)
        if not is_valid:
            raise ValueError("Dataset validation failed - no valid images found")
        
        logger.info("Dataset validation passed. Starting cache generation...")

        # Find all image files and their corresponding caption files
        self.image_paths = []
        self.caption_paths = []
        for img_path in self.data_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                caption_path = img_path.with_suffix('.txt')
                if caption_path.exists():
                    self.image_paths.append(img_path)
                    self.caption_paths.append(caption_path)

        # Initialize aspect buckets
        self.aspect_buckets = self.create_aspect_buckets()

        # Add tag processing
        self.tag_list = self._build_tag_list()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # Cache latents and embeddings
        self._cache_latents_and_embeddings_optimized()

    def create_aspect_buckets(self):
        aspect_buckets = {
            (1, 1): [],    # Square
            (4, 3): [],    # Landscape
            (3, 4): [],    # Portrait
            (16, 9): [],   # Widescreen
            (9, 16): [],   # Tall
        }
        # Sort images into aspect buckets
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                w, h = img.size
                ratio = w / h
                bucket = min(aspect_buckets.keys(),
                             key=lambda x: abs(x[0]/x[1] - ratio))
                aspect_buckets[bucket].append(img_path)
        return aspect_buckets

    def get_target_size_for_bucket(self, ratio):
        """Calculate target size maintaining aspect ratio and ~1024x1024 total pixels"""
        w, h = ratio
        # Calculate scale to maintain ~1024x1024 pixels
        scale = math.sqrt(1024 * 1024 / (w * h))
        
        # Calculate dimensions and ensure multiple of 64 (VAE requirement)
        target_w = int(round(w * scale / 64)) * 64
        target_h = int(round(h * scale / 64)) * 64
        
        # Ensure minimum dimension of 256 (NovelAI V3 requirement)
        target_w = max(256, min(2048, target_w))
        target_h = max(256, min(2048, target_h))
        
        return (target_h, target_w)

    def _build_tag_list(self):
        """Build a list of all unique tags from captions"""
        tags = set()
        for caption_path in self.caption_paths:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
                # Assuming tags are comma-separated
                caption_tags = [t.strip() for t in caption.split(',')]
                tags.update(caption_tags)
        return list(tags)

    def _get_clip_embeddings(self, image, tags):
        """Get CLIP embeddings for image and tags"""
        with torch.no_grad():
            # Get image embeddings
            inputs = self.clip_processor(images=image, return_tensors="pt").to("cuda")
            image_features = self.clip_model.get_image_features(**inputs)

            # Get tag embeddings with explicit max length and truncation
            text_inputs = self.clip_processor(
                text=tags,
                return_tensors="pt",
                padding=True,
                max_length=77,
                truncation=True
            ).to("cuda")
            text_features = self.clip_model.get_text_features(**text_inputs)

            return image_features, text_features

    def transform_image(self, image):
        """Transform image maintaining aspect ratio within SDXL requirements"""
        # Get original aspect ratio
        w, h = image.size
        
        # Log original dimensions
        logger.debug(f"Original dimensions: {w}x{h}")
        
        # Calculate target dimensions maintaining aspect ratio
        aspect_ratio = w / h
        
        # Enforce minimum dimension of 256 pixels
        min_size = 256
        max_size = 2048
        
        if w < h:
            target_w = max(min_size, min(max_size, w))
            target_h = int(target_w / aspect_ratio)
        else:
            target_h = max(min_size, min(max_size, h))
            target_w = int(target_h * aspect_ratio)
            
        # Ensure both dimensions meet minimum size
        target_w = max(min_size, target_w)
        target_h = max(min_size, target_h)
        
        # Ensure dimensions don't exceed maximum
        if target_w > max_size or target_h > max_size:
            scale = max_size / max(target_w, target_h)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        
        # Round to nearest multiple of 8 for VAE
        target_w = (target_w // 8) * 8
        target_h = (target_h // 8) * 8
        
        # Log target dimensions
        logger.debug(f"Target dimensions: {target_w}x{target_h}")
        
        # Validate final dimensions
        if not (min_size <= target_w <= max_size and min_size <= target_h <= max_size):
            raise ValueError(
                f"Invalid target dimensions {target_w}x{target_h}. "
                f"Must be between {min_size} and {max_size} pixels."
            )
        
        transform = transforms.Compose([
            transforms.Resize((target_h, target_w), 
                           interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return transform(image)

    def _cache_latents_and_embeddings_optimized(self):
        """Optimized version of caching that uses bfloat16 and batch processing"""
        try:
            # Group files by aspect ratio bucket
            bucket_files = {}
            for img_path, caption_path in zip(self.image_paths, self.caption_paths):
                with Image.open(img_path) as img:
                    w, h = img.size
                    # Calculate minimum size that maintains aspect ratio
                    if w < h:
                        scale = 256 / w
                    else:
                        scale = 256 / h
                        
                    target_w = max(256, int(w * scale))
                    target_h = max(256, int(h * scale))
                    
                    bucket = (target_w // 64, target_h // 64)  # Round to nearest multiple of 64
                    
                    if bucket not in bucket_files:
                        bucket_files[bucket] = []
                    bucket_files[bucket].append((img_path, caption_path))
            
            if not bucket_files:
                logger.info("All latents and embeddings already cached, skipping...")
                return

            # Process each bucket separately
            BATCH_SIZE = 8
            
            with to_device(self.vae, "cuda") as vae, \
                 to_device(self.text_encoder, "cuda") as text_encoder, \
                 to_device(self.text_encoder_2, "cuda") as text_encoder_2, \
                 to_device(self.clip_model, "cuda") as clip_model:
                
                vae.eval()
                text_encoder.eval()
                text_encoder_2.eval()
                clip_model.eval()

                # Add validation before caching
                def validate_dimensions(w, h):
                    min_size = 256
                    max_size = 2048
                    if w < min_size or h < min_size:
                        return False
                    if w > max_size or h > max_size:
                        return False
                    return True

                # Process each aspect ratio bucket
                for bucket, files in bucket_files.items():
                    w, h = bucket
                    target_h, target_w = self.get_target_size_for_bucket((w, h))
                    
                    # Validate dimensions before processing
                    if not validate_dimensions(target_w, target_h):
                        logger.warning(
                            f"Skipping bucket {bucket} - target size {target_w}x{target_h} "
                            f"outside valid range (256-2048)"
                        )
                        continue

                    logger.info(f"Processing bucket {bucket} with {len(files)} files...")
                    logger.info(f"Bucket {bucket} target size: {target_w}x{target_h}")
                    
                    bucket_start = time.time()
                    cache_stats = {
                        'processed': 0,
                        'skipped': 0,
                        'failed': 0
                    }
                    
                    # Create transform for this bucket
                    bucket_transform = transforms.Compose([
                        transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    
                    # Process files in batches within each bucket
                    for i in tqdm(range(0, len(files), BATCH_SIZE), 
                                desc=f"Caching bucket {bucket}"):
                        batch_files = files[i:i + BATCH_SIZE]
                        
                        # Prepare batch data
                        vae_images = []
                        clip_images = []
                        captions = []
                        
                        for img_path, caption_path in batch_files:
                            try:
                                # Check if already cached
                                latents_path = self.cache_dir / f"{Path(img_path).stem}_latents.pt"
                                embeddings_path = self.cache_dir / f"{Path(img_path).stem}_embeddings.pt"
                                
                                if os.path.exists(latents_path) and os.path.exists(embeddings_path):
                                    cache_stats['skipped'] += 1
                                    continue
                                
                                image = Image.open(img_path).convert("RGB")
                                vae_image = bucket_transform(image)
                                
                                # Validate image size meets minimum requirements
                                if vae_image.shape[1] < 256 or vae_image.shape[2] < 256:
                                    raise ValueError(
                                        f"Image {img_path} too small after resizing: "
                                        f"{vae_image.shape[1]}x{vae_image.shape[2]}. "
                                        f"Minimum size is 256x256."
                                    )
                                
                                logger.debug(f"Processed {img_path}")
                                logger.debug(f"Tensor dimensions: {vae_image.shape}")
                                
                                vae_images.append(vae_image)
                                clip_images.append(image)
                                
                                with open(caption_path, 'r', encoding='utf-8') as f:
                                    captions.append(f.read().strip())
                                    
                                cache_stats['processed'] += 1
                                
                            except Exception as e:
                                logger.error(f"Failed to process {img_path}: {str(e)}")
                                cache_stats['failed'] += 1
                                continue
                        
                        if not vae_images:  # Skip if no valid images in batch
                            continue
                            
                        # Stack images for VAE (now guaranteed to be same size within bucket)
                        image_batch = torch.stack(vae_images).to("cuda", dtype=torch.bfloat16)
                        
                        # Validate batch dimensions
                        if image_batch.shape[-2:] != (target_h, target_w):
                            raise ValueError(
                                f"Batch shape mismatch. Expected {(target_h, target_w)}, "
                                f"got {image_batch.shape[-2:]}"
                            )

                        with torch.no_grad():
                            # VAE encoding
                            latents_batch = self.vae.encode(image_batch).latent_dist.sample() * 0.18215

                            # Process text embeddings in batch
                            text_inputs = self.tokenizer(
                                captions,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to("cuda")
                            
                            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

                            text_inputs_2 = self.tokenizer_2(
                                captions,
                                padding="max_length",
                                max_length=self.tokenizer_2.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to("cuda")
                            
                            text_outputs_2 = self.text_encoder_2(text_inputs_2.input_ids)
                            text_embeddings_2 = text_outputs_2.last_hidden_state
                            pooled_text_embeddings_2 = text_outputs_2.pooler_output

                            # CLIP processing with original images
                            clip_inputs = self.clip_processor(
                                images=clip_images,  # Use original images
                                return_tensors="pt"
                            ).to("cuda")
                            clip_image_embeds = self.clip_model.get_image_features(**clip_inputs)
                            
                            # Process tags
                            all_tags = [t.strip() for caption in captions for t in caption.split(',')]
                            clip_text_inputs = self.clip_processor(
                                text=all_tags,
                                return_tensors="pt",
                                padding=True,
                                max_length=77,
                                truncation=True
                            ).to("cuda")
                            clip_tag_embeds = self.clip_model.get_text_features(**clip_text_inputs)

                        # Save individual files from batch
                        for idx, (img_path, _) in enumerate(batch_files):
                            cache_latents_path = self.cache_dir / f"{Path(img_path).stem}_latents.pt"
                            cache_embeddings_path = self.cache_dir / f"{Path(img_path).stem}_embeddings.pt"
                            
                            # Save embeddings
                            torch.save({
                                "text_embeddings": text_embeddings[idx].cpu(),
                                "text_embeddings_2": text_embeddings_2[idx].cpu(),
                                "pooled_text_embeddings_2": pooled_text_embeddings_2[idx].cpu(),
                                "clip_image_embed": clip_image_embeds[idx].cpu(),
                                "clip_tag_embeds": clip_tag_embeds[idx].cpu(),
                                "tags": all_tags[idx] if idx < len(all_tags) else []
                            }, cache_embeddings_path)
                            
                            # Save latents
                            torch.save(latents_batch[idx].cpu(), cache_latents_path)

                        # Clear memory
                        del image_batch, latents_batch, text_embeddings, text_embeddings_2
                        del pooled_text_embeddings_2, clip_image_embeds, clip_tag_embeds
                        torch.cuda.empty_cache()
                    
                    bucket_time = time.time() - bucket_start
                    logger.info(f"Bucket processing time: {bucket_time:.2f}s")
                    logger.info(f"Files processed: {cache_stats['processed']}")
                    logger.info(f"Files skipped: {cache_stats['skipped']}")
                    logger.info(f"Files failed: {cache_stats['failed']}")

            # Move models back to CPU
            self.vae.to("cpu")
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
            self.clip_model.to("cpu")

        except Exception as e:
            logger.error(f"Caching failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get item with proper tensor handling"""
        img_path = self.image_paths[idx]
        latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
        embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"

        # Load cached data
        latents = torch.load(latents_path, map_location='cpu')
        embeddings = torch.load(embeddings_path, map_location='cpu')

        # Load original image
        image = Image.open(img_path).convert("RGB")
        original_images = self.transform_image(image)  # [3, H, W]

        # Ensure latents is a tensor and has correct shape
        if isinstance(latents, list):
            latents = torch.stack(latents)

        if self.batch_size == 1:
            # Single sample processing
            return {
                "latents": latents.unsqueeze(0),  # [1, 4, H/8, W/8]
                "text_embeddings": embeddings["text_embeddings"].unsqueeze(0),  # [1, 77, 768]
                "text_embeddings_2": embeddings["text_embeddings_2"].unsqueeze(0),  # [1, 77, 1280]
                "pooled_text_embeddings_2": embeddings["pooled_text_embeddings_2"].unsqueeze(0),  # [1, 1280]
                "target_size": torch.tensor([original_images.shape[1], original_images.shape[2]], dtype=torch.long),
                "clip_image_embed": embeddings["clip_image_embed"].unsqueeze(0),
                "clip_tag_embeds": embeddings["clip_tag_embeds"].unsqueeze(0),
                "tags": embeddings["tags"],
                "original_images": original_images.unsqueeze(0)  # [1, 3, H, W]
            }
        else:
            # Batch processing
            return {
                "latents": latents,  # [4, H/8, W/8]
                "text_embeddings": embeddings["text_embeddings"],  # [77, 768]
                "text_embeddings_2": embeddings["text_embeddings_2"],  # [77, 1280]
                "pooled_text_embeddings_2": embeddings["pooled_text_embeddings_2"],  # [1280]
                "target_size": torch.tensor([original_images.shape[1], original_images.shape[2]], dtype=torch.long),
                "clip_image_embed": embeddings["clip_image_embed"],
                "clip_tag_embeds": embeddings["clip_tag_embeds"],
                "tags": embeddings["tags"],
                "original_images": original_images  # [3, H, W]
            }

def validate_dataset(data_dir):
    """Pre-process validation of all images in dataset"""
    logger.info("Starting dataset validation...")
    stats = {'valid': 0, 'invalid': 0, 'errors': {}}
    
    for path in glob.glob(os.path.join(data_dir, "*.png")):
        try:
            # Try to open and verify the image
            with Image.open(path) as img:
                # Check if image can be loaded
                img.verify()
                
                # Additional checks
                if img.mode != 'RGB':
                    raise ValueError(f"Invalid mode: {img.mode}")
                
                # Get image size
                width, height = img.size
                if width < 512 or height < 512:
                    raise ValueError(f"Image too small: {width}x{height}")
                
                stats['valid'] += 1
                
        except Exception as e:
            stats['invalid'] += 1
            error_type = type(e).__name__
            error_msg = str(e)
            
            if error_type not in stats['errors']:
                stats['errors'][error_type] = []
            stats['errors'][error_type].append((path, error_msg))
            
            logger.warning(f"- {path}: {error_type}: {error_msg}")

    # Log error statistics
    if stats['errors']:
        logger.warning("\nError Summary:")
        for error_type, errors in stats['errors'].items():
            logger.warning(f"\n{error_type} ({len(errors)} occurrences):")
            # Show first 5 examples of each error type
            for path, msg in errors[:5]:
                logger.warning(f"  - {os.path.basename(path)}: {msg}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors)-5} more")

    logger.info(f"\nValidation complete: {stats['valid']} valid, {stats['invalid']} invalid images")
    
    return stats['valid'] > 0, stats

def validate_image_dimensions(width, height):
    """Validate image dimensions for SDXL training"""
    # Convert to latent dimensions
    latent_width = width // 8
    latent_height = height // 8
    
    # Check minimum dimensions
    if latent_width < 32 or latent_height < 32:  # 256 pixels in image space
        return False, f"Latent dimensions too small: {latent_width}x{latent_height}"
        
    # Check maximum dimensions
    if latent_width > 256 or latent_height > 256:  # 2048 pixels in image space
        return False, f"Latent dimensions too large: {latent_width}x{latent_height}"
        
    # Check aspect ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        return False, f"Aspect ratio ({aspect_ratio:.2f}) outside supported range"
        
    return True, None
