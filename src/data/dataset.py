from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
from tqdm import tqdm
import logging
import traceback
import random
import re
from utils.validation import validate_dataset
import cv2
import numpy as np
from .ultimate_upscaler import UltimateUpscaler, USDUMode, USDUSFMode
from utils.validation import validate_image_dimensions
import os
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache", no_caching_latents=False, all_ar=False):
        """Initialize dataset with image preprocessing"""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.no_caching_latents = no_caching_latents
        self.all_ar = all_ar
        
        if not no_caching_latents:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all valid image paths (support multiple formats)
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            self.image_paths.extend(self.data_dir.glob(ext))
        self.image_paths = sorted(self.image_paths)
        
        self.latent_paths = [self.cache_dir / f"{path.stem}_latents.pt" for path in self.image_paths]
        
        # Log dataset size
        logger.info(f"Found {len(self.image_paths)} images in dataset")
        
        # Store model references
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # Initialize processors only if needed
        if not all_ar:
            logger.info("Initializing image processors...")
            self.upscaler = UltimateUpscaler()
        
        # Process images and captions
        self._preprocess_images()
        self.tag_stats = self._build_tag_statistics()
        logger.info(f"Processed {len(self.image_paths)} images with tag statistics")

    def _preprocess_images(self):
        """Preprocess all images to correct SDXL sizes and remove corrupted images"""
        valid_images = []
        corrupted_images = []
        
        # Use ThreadPoolExecutor for parallel processing
        def process_single_image(img_path):
            try:
                # Quick validation without full load
                with Image.open(img_path) as image:
                    # Only load header initially
                    width, height = image.size
                    
                    # If all_ar is True, accept image as-is
                    if self.all_ar:
                        return (img_path, None, True)
                    
                    # Get target size
                    target_width, target_height = self._get_target_size(width, height)
                    
                    # Skip if already correct size
                    if width == target_width and height == target_height:
                        return (img_path, None, True)
                    
                    # Only fully load and process if resize needed
                    image.load()
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Process image size
                    processed = self.process_image_size(image, target_width, target_height)
                    
                    # Save processed image
                    output_path = img_path.with_suffix('.png')
                    processed.save(output_path, format='PNG', quality=95)
                    
                    if output_path != img_path:
                        os.remove(img_path)
                    
                    return (output_path, None, True)
                    
            except Exception as e:
                return (img_path, str(e), False)

        # Use multiple threads for IO-bound operations
        max_workers = min(32, os.cpu_count() * 4)  # Limit max threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_image, self.image_paths),
                total=len(self.image_paths),
                desc="Preprocessing images"
            ))
        
        # Process results
        for img_path, error, is_valid in results:
            if is_valid:
                valid_images.append(img_path)
            else:
                corrupted_images.append(img_path)
                if error:
                    logger.error(f"Error preprocessing {img_path}: {error}")

        # Update image paths
        self.image_paths = valid_images
        
        # Log statistics
        total = len(valid_images) + len(corrupted_images)
        logger.info(f"Preprocessing complete:")
        logger.info(f"Total images processed: {total}")
        logger.info(f"Valid images: {len(valid_images)}")
        logger.info(f"Corrupted/removed images: {len(corrupted_images)}")

    def _parse_tags(self, caption):
        """Parse Midjourney-specific tags and general tags"""
        tags = []
        special_tags = {}
        
        # Split and process tags
        raw_tags = [t.strip() for t in caption.split(',')]
        
        # Only handle MJ-specific tags if they exist
        has_mj_tags = any('niji' in t.lower() or t.strip() in ['4', '5', '6'] for t in raw_tags)
        
        if has_mj_tags:
            # Handle anime style/niji at start
            if raw_tags and ('anime style' in raw_tags[0].lower() or 'niji' in raw_tags[0].lower()):
                special_tags['niji'] = True
                raw_tags = raw_tags[1:]
                
            # Handle version number - add masterpiece tag for any version
            if raw_tags and raw_tags[-1].strip() in ['4', '5', '6']:
                raw_tags = raw_tags[:-1]
                tags.append('masterpiece')  # Add masterpiece instead of version number
        
        for tag in raw_tags:
            tag = tag.lower().strip()
            
            # Skip empty tags
            if not tag:
                continue
            
            # Handle compound tags with weights (format: "tag::weight")
            if '::' in tag:
                parts = tag.split('::')
                tag = parts[0].strip()
                try:
                    weight = float(parts[1])
                    special_tags[f'{tag}_weight'] = weight
                except: pass

            # Handle style references
            if 'sref' in tag:
                refs = re.findall(r'[a-f0-9]{8}|https?://[^\s>]+', tag)
                if refs:
                    special_tags['sref'] = refs
                    continue  # Skip adding sref to regular tags

            # Handle MJ style parameters only if MJ tags exist
            if has_mj_tags:
                is_param = False
                for param in ['stylize', 'chaos', 'sw', 'sv']:
                    if param in tag:
                        try:
                            value = float(re.search(r'[\d.]+', tag).group())
                            special_tags[param] = value
                            is_param = True
                        except: continue
                if is_param:
                    continue  # Skip adding style parameters to regular tags

            # Handle general tags
            if tag.startswith(('a ', 'an ', 'the ')):  # Remove articles
                tag = ' '.join(tag.split()[1:])
            
            # Add to regular tags if it's not a special parameter
            if tag:
                tags.append(tag)
        
        return tags, special_tags

    def _calculate_tag_weights(self, tags, special_tags):
        """Calculate weights for tags"""
        weights = {}
        base_weight = 1.0
        
        # Apply special tag modifiers
        if 'masterpiece' in tags:  # Changed from version/quality check to masterpiece
            base_weight *= 1.3
        if special_tags.get('niji', False):
            base_weight *= 1.2
        if 'stylize' in special_tags:
            stylize_value = special_tags['stylize']
            stylize_weight = 1.0 + (math.log10(stylize_value) / 3.0)
            base_weight *= stylize_weight
        if 'chaos' in special_tags:
            chaos_value = special_tags['chaos']
            chaos_factor = 1.0 + (chaos_value / 200.0)
            base_weight *= chaos_factor
            
        # Calculate individual tag weights
        for i, tag in enumerate(tags):
            position_weight = 1.0 - (i * 0.05)
            weights[tag] = base_weight * max(0.5, position_weight)
            
        return weights

    def _format_caption(self, caption):
        """Format caption by cleaning and standardizing tags"""
        # Remove extra whitespace and normalize separators
        caption = re.sub(r'\s+', ' ', caption.strip())
        caption = re.sub(r'\s*,\s*', ', ', caption)
        
        # Split into tags
        tags = [t.strip().lower() for t in caption.split(',')]
        
        # Process tags
        formatted_tags = []
        special_params = []
        
        # Check if it has MJ-specific tags
        has_mj_tags = any('niji' in t.lower() or t.strip() in ['4', '5', '6'] for t in tags)
        
        for tag in tags:
            # Skip empty tags
            if not tag:
                continue
                
            # Handle special parameters (like --ar, etc)
            if tag.startswith('--'):
                special_params.append(tag)
                continue
                
            # Handle version numbers - convert to masterpiece (only if MJ tags present)
            if has_mj_tags and tag in ['4', '5', '6']:
                if 'masterpiece' not in formatted_tags:
                    formatted_tags.append('masterpiece')
                continue
                
            # Handle style parameters (only if MJ tags present)
            if has_mj_tags and any(param in tag for param in ['stylize', 'chaos', 'quality', 'niji']):
                special_params.append(tag)
                continue
                
            # Clean up regular tags
            tag = tag.strip()
            if tag.startswith(('a ', 'an ', 'the ')):  # Remove articles
                tag = ' '.join(tag.split()[1:])
            
            formatted_tags.append(tag)
        
        # Combine tags and parameters
        formatted_caption = ', '.join(formatted_tags)
        if special_params:
            formatted_caption += ', ' + ', '.join(special_params)
            
        return formatted_caption

    def _build_tag_statistics(self):
        """Build dataset-wide tag statistics and format captions"""
        stats = {
            'niji_count': 0,
            'quality_6_count': 0,
            'stylize_values': [],
            'chaos_values': [],
            'total_images': len(self.image_paths),
            'formatted_count': 0
        }
        
        for img_path in tqdm(self.image_paths, desc="Processing captions"):
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()
                    
                    # Format caption
                    formatted_caption = self._format_caption(original_caption)
                    
                    # Save if changed
                    if formatted_caption != original_caption:
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(formatted_caption)
                        stats['formatted_count'] += 1
                    
                    # Update tag statistics
                    _, special_tags = self._parse_tags(formatted_caption)
                    if special_tags.get('niji', False):
                        stats['niji_count'] += 1
                    if special_tags.get('version', 0) == 6:
                        stats['quality_6_count'] += 1
                    if 'stylize' in special_tags:
                        stats['stylize_values'].append(special_tags['stylize'])
                    if 'chaos' in special_tags:
                        stats['chaos_values'].append(special_tags['chaos'])
                        
                except Exception as e:
                    logger.warning(f"Error processing caption for {img_path}: {str(e)}")
        
        logger.info(f"Formatted {stats['formatted_count']} captions")
        return stats

    def _get_target_size(self, width, height):
        """
        Get target size based on more permissive SDXL validation rules.
        Only intervenes for extreme cases.
        """
        try:
            # Check if dimensions need adjustment
            is_valid, _ = validate_image_dimensions(width, height)
            
            if not is_valid:
                min_dim = min(width, height)
                max_dim = max(width, height)
                aspect_ratio = width / height
                
                # Scale up if both dimensions are too small
                if width < 512 and height < 512:
                    scale = 512 / min_dim
                    width = int(width * scale)
                    height = int(height * scale)
                
                # Scale down if any dimension is too large
                elif max_dim > 2560:
                    scale = 2560 / max_dim
                    width = int(width * scale)
                    height = int(height * scale)
                
                # Fix extreme aspect ratios
                elif aspect_ratio > 4.0:
                    width = int(height * 4.0)
                elif aspect_ratio < 0.25:
                    height = int(width * 4.0)
                
                # Fix tiny dimension with normal/large other dimension
                elif min_dim < 384 and max_dim > 768:
                    scale = 384 / min_dim
                    width = int(width * scale)
                    height = int(height * scale)
            
            # Round dimensions to multiples of 8
            width = ((width + 7) // 8) * 8
            height = ((height + 7) // 8) * 8
            
            return (width, height)

        except Exception as e:
            logger.error(f"Error calculating target size: {str(e)}")
            # Return rounded dimensions even on error
            width = ((width + 7) // 8) * 8
            height = ((height + 7) // 8) * 8
            return (width, height)

    def process_image_size(self, image, target_width, target_height):
        """Process image size with advanced upscaling"""
        width, height = image.size
        
        if width == target_width and height == target_height:
            return image

        logger.info(f"Processing image from {width}x{height} to {target_width}x{target_height}")
        
        scale_factor = max(target_width / width, target_height / height)
        
        try:
            if scale_factor < 1:
                # For downscaling, use high-quality downscaling
                return self._downscale_image(image, target_width, target_height)
            else:
                # For upscaling, use AI upscaling only if the image is too small
                min_size = 768  # Minimum size before using AI upscaling
                if width < min_size or height < min_size:
                    return self._upscale_image(image, target_width, target_height)
                else:
                    # Use regular Lanczos upscaling for larger images
                    return image.resize((target_width, target_height), Image.LANCZOS)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def _upscale_image(self, image, target_width, target_height):
        """Upscale image using Ultimate SD Upscaler"""
        try:
            # Calculate scale factor
            scale_factor = max(target_width / image.width, target_height / image.height)
            
            # Get caption for the image
            caption_path = Path(str(image.filename)).with_suffix('.txt')
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            except:
                prompt = "high quality, detailed image"  # fallback prompt
            
            # Process with upscaler
            upscaled = self.upscaler.upscale(
                image=image,
                prompt=prompt,
                upscale_factor=scale_factor,
                mode=USDUMode.LINEAR,
                tile_width=512,
                tile_height=512,
                padding=32,
                num_steps=20,
                guidance_scale=7.5,
                strength=0.4,
                seam_fix_mode=USDUSFMode.HALF_TILE,
                seam_fix_denoise=0.35,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16
            )
            
            # Final resize to exact target size if needed
            if upscaled.size != (target_width, target_height):
                upscaled = upscaled.resize((target_width, target_height), Image.LANCZOS)
                
            return upscaled
            
        except Exception as e:
            logger.error(f"AI upscaling failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def _downscale_image(self, image, target_width, target_height):
        """High-quality downscaling optimized for performance while maintaining quality"""
        try:
            # Convert to numpy once at the start
            img_np = np.array(image)
            
            # Calculate scale factor
            scale_factor = max(image.width / target_width, image.height / target_height)
            
            # Calculate number of steps - fewer steps for smaller scale factors
            num_steps = max(1, min(3, int(scale_factor // 1.5)))
            
            # Apply initial gaussian blur to prevent aliasing
            # Adjust blur based on scale factor
            blur_radius = min(2.0, 0.6 * scale_factor)
            kernel_size = int(blur_radius * 3) | 1  # ensure odd number
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), blur_radius)
            
            # Apply edge-preserving bilateral filter for large downscaling
            if scale_factor > 2.0:
                d = 9  # diameter of pixel neighborhood
                sigma_color = 75
                sigma_space = 75
                img_np = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
            
            # Progressive downscaling
            for step in range(num_steps):
                # Calculate intermediate size
                progress = (step + 1) / num_steps
                intermediate_width = int(target_width + (image.width - target_width) * (1 - progress))
                intermediate_height = int(target_height + (image.height - target_height) * (1 - progress))
                
                # Use INTER_AREA for downscaling
                img_np = cv2.resize(img_np, (intermediate_width, intermediate_height), 
                                  interpolation=cv2.INTER_AREA)
                
                # Apply light sharpening on final step
                if step == num_steps - 1:
                    kernel = np.array([[-0.5,-0.5,-0.5], 
                                     [-0.5, 5.0,-0.5],
                                     [-0.5,-0.5,-0.5]]) / 2.0
                    img_np = cv2.filter2D(img_np, -1, kernel)
            
            # Convert back to PIL
            return Image.fromarray(img_np)
            
        except Exception as e:
            logger.error(f"Advanced downscaling failed: {str(e)}, falling back to basic resize")
            return image.resize((target_width, target_height), Image.LANCZOS)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        latent_path = self.latent_paths[idx]
        caption_path = image_path.with_suffix('.txt')

        # Load caption and process tags
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            tags, special_tags = self._parse_tags(caption)
            tag_weights = self._calculate_tag_weights(tags, special_tags)

        # Load cached latents if available and caching is enabled
        if not self.no_caching_latents and latent_path.exists():
            cached_data = torch.load(latent_path)
            return {
                "latents": cached_data["latents"],
                "text_embeddings": cached_data["text_embeddings"],
                "text_embeddings_2": cached_data["text_embeddings_2"],
                "added_cond_kwargs": cached_data["added_cond_kwargs"],
                "tags": tags,
                "special_tags": special_tags,
                "tag_weights": tag_weights
            }
        
        # Generate VAE latents
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Get target size from SDXL buckets
            width, height = image.size
            target_width, target_height = self._get_target_size(width, height)
            
            # Process image size with AI upscaling when needed
            processed_image = self.process_image_size(image, target_width, target_height)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = transform(processed_image).unsqueeze(0)
            
            # Generate VAE latents
            with torch.no_grad():
                image = image.to(self.vae.device, dtype=self.vae.dtype)
                latents = self.vae.encode(image).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.squeeze(0)
            
            # Process text embeddings
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate text embeddings
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.text_encoder.device))[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs_2.input_ids.to(self.text_encoder_2.device),
                    output_hidden_states=True
                )
                
                # Get pooled and hidden states
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
                
                # Create added conditions
                added_cond_kwargs = {
                    "text_embeds": pooled_output,
                    "time_ids": self._get_add_time_ids(image),
                }
            
            # Save to cache only if caching is enabled
            if not self.no_caching_latents:
                cache_data = {
                    "latents": latents,
                    "text_embeddings": text_embeddings,
                    "text_embeddings_2": hidden_states,
                    "added_cond_kwargs": added_cond_kwargs
                }
                torch.save(cache_data, latent_path)
            
            return {
                "latents": latents,
                "text_embeddings": text_embeddings,
                "text_embeddings_2": hidden_states,
                "added_cond_kwargs": added_cond_kwargs,
                "tags": tags,
                "special_tags": special_tags,
                "tag_weights": tag_weights
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _get_add_time_ids(self, image):
        """Generate time ids for SDXL conditioning"""
        original_size = image.shape[-2:]  # (height, width)
        target_size = (1024, 1024)  # SDXL default
        crop_coords_top_left = (0, 0)
        crop_coords_bottom_right = (original_size[0], original_size[1])
        
        add_time_ids = torch.tensor([
            original_size[1],  # original width
            original_size[0],  # original height
            target_size[1],    # target width
            target_size[0],    # target height
            crop_coords_top_left[1],      # crop top
            crop_coords_top_left[0],      # crop left
            crop_coords_bottom_right[1],  # crop bottom
            crop_coords_bottom_right[0],  # crop right
        ])
        
        return add_time_ids.unsqueeze(0)

    def __len__(self):
        return len(self.image_paths)

    def shuffle_samples(self):
        """Shuffle the dataset samples"""
        combined = list(zip(self.image_paths, self.latent_paths))
        random.shuffle(combined)
        self.image_paths, self.latent_paths = zip(*combined)


def custom_collate(batch):
    """Custom collate function that validates dimensions and handles varying tensor sizes"""
    def validate_latents(latents):
        h, w = latents.shape[-2:]
        min_size = 256 // 8  # Minimum 256 pixels in image space
        max_size = 2048 // 8  # Maximum 2048 pixels in image space
        return (h >= min_size and w >= min_size and
                h <= max_size and w <= max_size)

    def resize_latents(latents, target_height, target_width):
        return transforms.functional.interpolate(
            latents,
            size=(target_height, target_width),
            mode='nearest'
        )

    # Filter out invalid samples
    valid_batch = [item for item in batch if validate_latents(item['latents'])]
    
    if not valid_batch:
        raise ValueError("No valid samples in batch (all below minimum size)")

    # Find the largest dimensions to pad to
    max_height = max(latent.shape[-2] for latent in [x["latents"] for x in valid_batch])
    max_width = max(latent.shape[-1] for latent in [x["latents"] for x in valid_batch])

    # Resize or pad latents to the largest dimensions
    resized_latents = [
        resize_latents(x["latents"], max_height, max_width)
        for x in valid_batch
    ]

    # Stack resized tensors
    batch_dict = {
        "latents": torch.stack(resized_latents),
        "text_embeddings": torch.stack([x["text_embeddings"] for x in valid_batch]),
        "text_embeddings_2": torch.stack([x["text_embeddings_2"] for x in valid_batch]),
        "pooled_text_embeddings_2": torch.stack([x["pooled_text_embeddings_2"] for x in valid_batch]),
        "tags": [x["tags"] for x in valid_batch]
    }

    return batch_dict
