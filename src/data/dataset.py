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
import random
import re

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
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

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        latent_path = self.latent_paths[idx]
        caption_path = image_path.with_suffix('.txt')

        # Load caption and process tags
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            tags, special_tags = self._parse_tags(caption)
            tag_weights = self._calculate_tag_weights(tags, special_tags)

        # Load cached latents if available
        if latent_path.exists():
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
        
        # Generate latents if not cached
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = transform(image).unsqueeze(0)
            
            # Generate VAE latents
            with torch.no_grad():
                image = image.to(self.vae.device, dtype=self.vae.dtype)
                latents = self.vae.encode(image).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
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
            
            # Save to cache
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
