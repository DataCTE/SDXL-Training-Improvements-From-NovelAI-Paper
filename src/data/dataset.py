from pathlib import Path
from PIL import Image, ImageFilter
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
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.realesrgan_utils import RealESRGANer
import numpy as np

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache", no_caching_latents=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.no_caching_latents = no_caching_latents
        
        if not no_caching_latents:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset first
        valid = validate_dataset(data_dir)[0]
        if not valid:
            raise ValueError("Dataset validation failed")
            
        # Get all valid image paths
        self.image_paths = sorted(list(self.data_dir.glob("*.png")))
        self.latent_paths = [self.cache_dir / f"{path.stem}_latents.pt" for path in self.image_paths]
        
        # Store model references
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # Process captions and build statistics
        self.tag_stats = self._build_tag_statistics()
        logger.info(f"Processed {len(self.image_paths)} images with tag statistics")
        
        # Initialize upscaler with standard model instead of anime
        model_path = load_file_from_url(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth',
            model_dir='weights'
        )
        self.upscaler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model='RealESRGAN_x4plus',
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=True
        )

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
        """Get closest SDXL aspect ratio bucket size"""
        # SDXL standard aspect ratios (larger sizes)
        sdxl_sizes = [
            (1024, 1024),  # 1:1
            (1152, 896),   # ~1.29:1
            (896, 1152),   # ~1:1.29
            (1216, 832),   # ~1.46:1
            (832, 1216),   # ~1:1.46
            (1344, 768),   # ~1.75:1
            (768, 1344),   # ~1:1.75
            (1536, 640),   # ~2.4:1
            (640, 1536),   # ~1:2.4
        ]
        
        aspect_ratio = width / height
        closest_size = min(sdxl_sizes, 
            key=lambda size: abs((size[0] / size[1]) - aspect_ratio))
        return closest_size

    def _process_image_size(self, image, target_width, target_height):
        """Process image size with AI upscaling for small images and high-quality downscaling for large images"""
        width, height = image.size
        
        # Calculate scaling factors
        width_scale = width / target_width
        height_scale = height / target_height
        
        # If image is larger than target, use progressive downscaling
        if width_scale > 1 or height_scale > 1:
            # Calculate number of steps for progressive downscaling
            scale_factor = max(width_scale, height_scale)
            num_steps = max(1, min(4, int(scale_factor // 2)))
            
            current_image = image
            for step in range(num_steps):
                # Calculate intermediate size
                intermediate_scale = 1.0 + (scale_factor - 1.0) * (num_steps - step - 1) / num_steps
                intermediate_width = int(target_width * intermediate_scale)
                intermediate_height = int(target_height * intermediate_scale)
                
                # Apply gaussian blur before resizing to reduce aliasing
                if step == 0:
                    blur_radius = min(2.0, 0.5 * scale_factor)
                    current_image = current_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                # Resize with high-quality Lanczos
                current_image = current_image.resize(
                    (intermediate_width, intermediate_height),
                    Image.Resampling.LANCZOS
                )
            
            return current_image
        
        # If image is smaller, use AI upscaling
        if width_scale < 1 or height_scale < 1:
            # Convert PIL to numpy array
            img_np = np.array(image)
            
            # Upscale with RealESRGAN
            output, _ = self.upscaler.enhance(img_np)
            
            # Convert back to PIL and resize to target
            upscaled = Image.fromarray(output)
            return upscaled.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return image

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
            processed_image = self._process_image_size(image, target_width, target_height)
            
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
