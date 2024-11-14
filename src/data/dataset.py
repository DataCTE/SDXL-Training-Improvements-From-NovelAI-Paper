from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
from tqdm import tqdm
import logging
import traceback
import random
import re
import cv2
import numpy as np
from .ultimate_upscaler import UltimateUpscaler, USDUMode, USDUSFMode
from utils.validation import validate_image_dimensions
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache", no_caching_latents=False, all_ar=False, 
                 max_dimension=2048, num_workers=None, prefetch_factor=2, 
                 min_size=512, max_size=2048, resolution_type="square",
                 enable_bucket_sampler=True, bucket_reso_steps=64,
                 min_bucket_reso=256, max_bucket_reso=2048,
                 token_dropout_rate=0.1, caption_dropout_rate=0.1):
        """
        Enhanced dataset initialization with NovelAI improvements
        Args:
            resolution_type: One of ["square", "portrait", "landscape"] for aspect ratio handling
            enable_bucket_sampler: Use bucketing for efficient batching
            bucket_reso_steps: Resolution steps for bucketing (64 recommended)
            token_dropout_rate: Rate for random token dropout during training
            caption_dropout_rate: Rate for entire caption dropout
        """
        super().__init__()
        
        # Basic initialization
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.no_caching_latents = no_caching_latents
        self.all_ar = all_ar
        
        # Resolution and bucketing parameters
        self.min_size = min_size
        self.max_size = max_size
        self.resolution_type = resolution_type
        self.enable_bucket_sampler = enable_bucket_sampler
        self.bucket_reso_steps = bucket_reso_steps
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        
        # Text augmentation parameters
        self.token_dropout_rate = token_dropout_rate
        self.caption_dropout_rate = caption_dropout_rate
        
        # Model components
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # Initialize upscaler
        self.upscaler = UltimateUpscaler(
            model_path="Lykon/DreamShaper",
            device=self.vae.device,
            dtype=self.vae.dtype
        )
        
        # Performance optimization
        self.num_workers = num_workers or min(os.cpu_count(), 32)
        self.prefetch_factor = prefetch_factor
        
        # Create cache directory
        if not no_caching_latents:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize image paths
        self._initialize_dataset()
        
        # Initialize bucketing
        if self.enable_bucket_sampler:
            self._initialize_buckets()
            
        # Process latents and build tag statistics
        if not no_caching_latents:
            self._batch_process_latents_efficient()
        self.tag_stats = self._build_tag_statistics()
        
    def _initialize_dataset(self):
        """Initialize dataset with improved image validation"""
        logger.info("Initializing dataset...")
        
        # Collect image paths
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            image_paths.extend(self.data_dir.glob(ext))
        
        # Validate images in parallel
        valid_paths = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for path in image_paths:
                futures.append(executor.submit(self._validate_image, path))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating images"):
                result = future.result()
                if result is not None:
                    valid_paths.append(result)
        
        self.image_paths = sorted(valid_paths)
        self.latent_paths = [self.cache_dir / f"{path.stem}_latents.pt" for path in self.image_paths]
        logger.info(f"Found {len(self.image_paths)} valid images")
        
    def _validate_image(self, img_path):
        """Validate single image with improved error handling"""
        try:
            # Check caption file
            caption_path = img_path.with_suffix('.txt')
            if not caption_path.exists():
                logger.warning(f"Skipping {img_path}: No caption file")
                return None
                
            # Validate image
            with Image.open(img_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                # Get dimensions
                width, height = img.size
                
                # If all_ar is True, accept all sizes
                if self.all_ar:
                    return img_path
                
                # Basic validation
                if width < self.min_size or height < self.min_size:
                    # Instead of skipping, try to upscale
                    target_width = max(width, self.min_size)
                    target_height = max(height, self.min_size)
                    img = self._upscale_image(img, target_width, target_height)
                    width, height = img.size
                    
                # Aspect ratio validation
                if not self.all_ar:
                    aspect_ratio = width / height
                    if self.resolution_type == "square" and not 0.95 <= aspect_ratio <= 1.05:
                        logger.warning(f"Skipping {img_path}: Not square ({aspect_ratio:.2f})")
                        return None
                    elif self.resolution_type == "portrait" and aspect_ratio >= 1:
                        logger.warning(f"Skipping {img_path}: Not portrait ({aspect_ratio:.2f})")
                        return None
                    elif self.resolution_type == "landscape" and aspect_ratio <= 1:
                        logger.warning(f"Skipping {img_path}: Not landscape ({aspect_ratio:.2f})")
                        return None
                
                return img_path
                
        except Exception as e:
            logger.error(f"Error validating {img_path}: {str(e)}")
            return None

    def _preprocess_images_parallel(self):
        """
        Parallel image preprocessing with more robust error handling
        Uses ThreadPoolExecutor for I/O-bound tasks
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def safe_process_image(img_path):
            try:
                with Image.open(img_path) as image:
                    width, height = image.size
                    
                    # Skip processing if all_ar is True
                    if self.all_ar:
                        return img_path, None
                    
                    # Get target size
                    target_width, target_height = self._get_target_size(width, height)
                    
                    # Only process if resize is needed
                    if width != target_width or height != target_height:
                        processed = self.process_image_size(image, target_width, target_height)
                        output_path = img_path.with_suffix('.png')
                        processed.save(output_path, format='PNG', quality=95)
                        
                        # Remove original if processed
                        if output_path != img_path:
                            os.remove(img_path)
                        
                        return output_path, None
                    
                    return img_path, None
                
            except Exception as e:
                return img_path, str(e)

        # Efficient parallel processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(safe_process_image, path): path for path in self.image_paths}
            
            processed_images = []
            errors = []
            
            for future in as_completed(future_to_path):
                processed_path, error = future.result()
                
                if error:
                    errors.append((processed_path, error))
                else:
                    processed_images.append(processed_path)
            
            # Update image paths
            self.image_paths = processed_images
            
            # Log errors if any
            if errors:
                for path, error in errors:
                    logger.error(f"Error processing {path}: {error}")
        
        logger.info(f"Preprocessing complete: {len(processed_images)} images processed")

    def _batch_process_latents_efficient(self, batch_size=4):
        """Process and cache latents in batches using multiple workers"""
        uncached_images = [
            img_path for img_path, lat_path in zip(self.image_paths, self.latent_paths)
            if not lat_path.exists()
        ]
        
        if not uncached_images:
            logger.info("All latents are already cached")
            return

        logger.info(f"Caching latents for {len(uncached_images)} images in batches")
        
        # Group images by size using parallel processing
        def group_image_by_size(img_path):
            try:
                # First check if caption file exists
                caption_path = Path(img_path).with_suffix('.txt')
                if not caption_path.exists():
                    logger.warning(f"Skipping {img_path}: No caption file found")
                    return None
                    
                with Image.open(img_path) as img:
                    # Get dimensions
                    width, height = img.size
                    
                    # If all_ar is True, just round to multiples of 8
                    if self.all_ar:
                        width = ((width + 7) // 8) * 8
                        height = ((height + 7) // 8) * 8
                    # Otherwise limit maximum dimension while preserving aspect ratio
                    elif max(width, height) > 2048:
                        scale = 2048 / max(width, height)
                        width = int(width * scale)
                        height = int(height * scale)
                        
                    return (img_path, f"{width}x{height}")
            except Exception as e:
                logger.error(f"Error reading image {img_path}: {str(e)}")
                return None

        # Use ThreadPoolExecutor for I/O-bound size checking
        size_groups = {}
        valid_images = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for result in tqdm(
                executor.map(group_image_by_size, uncached_images),
                total=len(uncached_images),
                desc="Grouping images by size"
            ):
                if result:
                    img_path, size_key = result
                    if size_key not in size_groups:
                        size_groups[size_key] = []
                    size_groups[size_key].append(img_path)
                    valid_images.append(img_path)

        def process_batch_with_vae(batch_paths):
            """Process a batch of images through VAE and generate embeddings"""
            if not batch_paths:
                return []
            
            batch_results = []
            try:
                # Process images in batch
                images = []
                valid_paths = []
                for path in batch_paths:
                    try:
                        with Image.open(path) as img:
                            width, height = img.size
                            
                            # If all_ar is True, just round to multiples of 8
                            if self.all_ar:
                                width = ((width + 7) // 8) * 8
                                height = ((height + 7) // 8) * 8
                                if width != img.size[0] or height != img.size[1]:
                                    img = img.resize((width, height), Image.LANCZOS)
                            # Otherwise scale down if needed while preserving aspect ratio
                            elif max(width, height) > 2048:
                                scale = 2048 / max(width, height)
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Transform image
                            image_tensor = transforms.ToTensor()(img)
                            image_tensor = transforms.Normalize([0.5], [0.5])(image_tensor)
                            images.append(image_tensor)
                            valid_paths.append(path)
                    except Exception as e:
                        logger.error(f"Error processing image {path}: {str(e)}")
                        continue

                if not images:
                    return []

                # Stack images and process through VAE
                with torch.no_grad():
                    image_tensor = torch.stack(images).to(self.vae.device, dtype=self.vae.dtype)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # Process each result
                for idx, img_path in enumerate(valid_paths):
                    try:
                        caption_path = Path(img_path).with_suffix('.txt')
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        # Generate text embeddings
                        text_inputs = self.tokenizer(
                            caption,
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.text_encoder.device)
                        
                        text_inputs_2 = self.tokenizer_2(
                            caption,
                            padding="max_length",
                            max_length=self.tokenizer_2.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.text_encoder_2.device)
                        
                        with torch.no_grad():
                            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                            text_embeddings_2 = self.text_encoder_2(
                                text_inputs_2.input_ids,
                                output_hidden_states=True
                            )
                            pooled_output = text_embeddings_2[0]
                            hidden_states = text_embeddings_2.hidden_states[-2]
                        
                        # Reshape pooled output to match hidden states dimensions
                        pooled_output = pooled_output.unsqueeze(1).unsqueeze(2).expand(-1, hidden_states.size(1), hidden_states.size(2), -1)
                        
                        # Save cache data
                        cache_data = {
                            "latents": latents[idx],
                            "text_embeddings": text_embeddings,
                            "text_embeddings_2": hidden_states,
                            "added_cond_kwargs": {
                                "text_embeds": pooled_output,
                                "time_ids": self._get_add_time_ids(image_tensor[idx:idx+1])
                            }
                        }
                        
                        latent_path = self.cache_dir / f"{Path(img_path).stem}_latents.pt"
                        torch.save(cache_data, latent_path)
                        batch_results.append(latent_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
                        continue

                # Clear memory
                del image_tensor, latents
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
            
            return batch_results

        # Process each size group
        processed_paths = []
        total_images = sum(len(group) for group in size_groups.values())
        
        with tqdm(total=total_images, desc="Caching latents") as pbar:
            for size, group_paths in size_groups.items():
                logger.info(f"Processing size group {size} ({len(group_paths)} images)")
                
                # Process images in chunks
                for chunk_start in range(0, len(group_paths), batch_size):
                    chunk_paths = group_paths[chunk_start:chunk_start + batch_size]
                    results = process_batch_with_vae(chunk_paths)
                    processed_paths.extend(results)
                    pbar.update(len(chunk_paths))
                    
                    # Cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
        
        logger.info(f"Successfully cached {len(processed_paths)} latents")

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
            # If all_ar is True, keep original size
            if self.all_ar:
                # Just round to multiples of 8 for VAE
                width = ((width + 7) // 8) * 8
                height = ((height + 7) // 8) * 8
                return (width, height)
                
            # Basic validation - only check minimum size
            if width < 256 or height < 256:
                min_dim = min(width, height)
                max_dim = max(width, height)
                aspect_ratio = width / height
                
                # Scale up if both dimensions are too small
                if width < 512 and height < 512:
                    scale = 512 / min_dim
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
        # If all_ar is True, keep original size
        if self.all_ar:
            width, height = image.size
            # Just round to multiples of 8 for VAE
            width = ((width + 7) // 8) * 8
            height = ((height + 7) // 8) * 8
            if width == image.size[0] and height == image.size[1]:
                return image
            return image.resize((width, height), Image.LANCZOS)
            
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

    def _initialize_buckets(self):
        """Initialize aspect ratio buckets according to NovelAI paper section 4.1.2"""
        logger.info("Initializing aspect ratio buckets...")
        buckets = []
        
        # Generate buckets based on max dimensions
        width = 256
        while width <= self.max_bucket_reso:
            # Find largest height that satisfies constraints
            height = self.max_bucket_reso
            while height * width > 512 * 768 or height > self.max_bucket_reso:
                height -= 64
            if height >= self.min_bucket_reso:
                buckets.append((width, height))
            width += 64
            
        # Add portrait buckets
        width = 256
        while width <= self.max_bucket_reso:
            height = self.max_bucket_reso
            while height * width > 512 * 768 or height > self.max_bucket_reso:
                height -= 64
            if height >= self.min_bucket_reso and (height, width) not in buckets:
                buckets.append((height, width))
            width += 64
            
        # Add square bucket
        buckets.append((512, 512))
        
        # Store bucket info
        self.buckets = buckets
        self.bucket_data = {bucket: [] for bucket in buckets}
        logger.info(f"Created {len(buckets)} aspect ratio buckets")

    def _assign_to_bucket(self, img_path):
        """Assign an image to the most appropriate bucket"""
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Find the best fitting bucket
                best_bucket = None
                min_area_diff = float('inf')
                
                for bucket_h, bucket_w in self.buckets:
                    # Skip if bucket is too small
                    if bucket_h < height or bucket_w < width:
                        continue
                    
                    # Calculate area difference
                    area_diff = (bucket_h * bucket_w) - (height * width)
                    if area_diff < min_area_diff:
                        min_area_diff = area_diff
                        best_bucket = (bucket_h, bucket_w)
                
                return img_path, best_bucket
                
        except Exception as e:
            logger.error(f"Error assigning {img_path} to bucket: {str(e)}")
            return img_path, None

    def _apply_text_transforms(self, caption):
        """Apply text augmentation transforms"""
        if random.random() < self.caption_dropout_rate:
            return ""
        
        if self.token_dropout_rate > 0:
            tokens = caption.split(",")
            tokens = [token.strip() for token in tokens if token.strip()]
            
            # Keep tokens with probability (1 - token_dropout_rate)
            tokens = [token for token in tokens if random.random() > self.token_dropout_rate]
            
            # Ensure at least one token remains
            if not tokens:
                tokens = [random.choice(caption.split(",")).strip()]
            
            return ", ".join(tokens)
        
        return caption

    def _process_uncached_item(self, img_path, caption_path):
        """Process an item that hasn't been cached yet"""
        # Load and process image
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            
            # Get bucket dimensions
            bucket_h, bucket_w = self._get_bucket_size(image.size)
            
            # Process image with advanced resizing
            processed_image = self._advanced_resize(image, bucket_w, bucket_h)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image_tensor = transform(processed_image).unsqueeze(0)
            
            # Generate VAE latents
            with torch.no_grad():
                image_tensor = image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.squeeze(0)
            
            # Load and process caption
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Apply text augmentation
            caption = self._apply_text_transforms(caption)
            
            # Generate text embeddings
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder.device)
            
            text_inputs_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder_2.device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                text_embeddings_2 = self.text_encoder_2(
                    text_inputs_2.input_ids,
                    output_hidden_states=True
                )
                pooled_output = text_embeddings_2[0]
                hidden_states = text_embeddings_2.hidden_states[-2]
            
            # Generate time embeddings
            original_size = image.size
            target_size = (bucket_w, bucket_h)
            crop_coords_top_left = (0, 0)
            crop_coords_bottom_right = original_size
            
            add_time_ids = torch.tensor([
                original_size[0],          # original width
                original_size[1],          # original height
                target_size[0],            # target width
                target_size[1],            # target height
                crop_coords_top_left[0],     # crop left
                crop_coords_top_left[1],     # crop top
                crop_coords_bottom_right[0], # crop right
                crop_coords_bottom_right[1]  # crop bottom
            ], dtype=torch.float32, device=self.vae.device)
            
            # Add batch dimension: [1, 6]
            add_time_ids = add_time_ids.unsqueeze(0)
            
            # Prepare cache data
            cache_data = {
                'latents': latents,
                'text_embeddings': text_embeddings,
                'text_embeddings_2': hidden_states,
                'added_cond_kwargs': {
                    'text_embeds': pooled_output.unsqueeze(1).unsqueeze(2),
                    'time_ids': add_time_ids
                },
                'original_caption': caption,
                'bucket_size': (bucket_h, bucket_w)
            }
            
            return cache_data
    
    def _get_bucket_size(self, image_size):
        """Get the most appropriate bucket size for an image"""
        width, height = image_size
        
        # Find best fitting bucket
        best_bucket = None
        min_area_diff = float('inf')
        
        for bucket_h, bucket_w in self.buckets:
            if bucket_h < height or bucket_w < width:
                continue
            
            area_diff = (bucket_h * bucket_w) - (height * width)
            if area_diff < min_area_diff:
                min_area_diff = area_diff
                best_bucket = (bucket_h, bucket_w)
        
        if best_bucket is None:
            # If no bucket fits, use the largest bucket
            best_bucket = max(self.buckets, key=lambda x: x[0] * x[1])
        
        return best_bucket
    
    def _advanced_resize(self, image, target_width, target_height):
        """Advanced image resizing with aspect ratio preservation"""
        width, height = image.size
        
        # Calculate target dimensions preserving aspect ratio
        aspect_ratio = width / height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # High-quality downscaling
        if width > new_width or height > new_height:
            # Use Lanczos for downscaling
            image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            # Use bicubic for upscaling
            image = image.resize((new_width, new_height), Image.BICUBIC)
        
        # Create new image with padding
        result = Image.new('RGB', (target_width, target_height))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        result.paste(image, (left, top))
        
        return result

    def custom_collate(self, batch):
        """
        Custom collate function for DataLoader with advanced batch handling.
        Handles variable size inputs and applies proper padding.
        
        Args:
            batch: List of tuples (latent, text_ids, text_ids_2)
            
        Returns:
            dict: Collated batch with proper padding
        """
        latents = torch.stack([item["latent"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
        
        # Get original image sizes for proper scaling
        original_sizes = torch.tensor([[item["original_size"][0], item["original_size"][1]] for item in batch])
        target_sizes = torch.tensor([[item["target_size"][0], item["target_size"][1]] for item in batch])
        
        # Collate attention masks if present
        attention_mask = None
        if "attention_mask" in batch[0]:
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        attention_mask_2 = None
        if "attention_mask_2" in batch[0]:
            attention_mask_2 = torch.stack([item["attention_mask_2"] for item in batch])
        
        # Handle tag weights if present
        tag_weights = None
        if "tag_weights" in batch[0]:
            tag_weights = torch.stack([item["tag_weights"] for item in batch])
        
        return {
            "latent": latents,
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "original_size": original_sizes,
            "target_size": target_sizes,
            "attention_mask": attention_mask,
            "attention_mask_2": attention_mask_2,
            "tag_weights": tag_weights
        }

    def __getitem__(self, idx):
        """Get a single item with improved text augmentation and bucketing"""
        try:
            img_path = self.image_paths[idx]
            caption_path = Path(img_path).with_suffix('.txt')
            
            # Load cached latents if available
            if not self.no_caching_latents and self.latent_paths[idx].exists():
                cache_data = torch.load(self.latent_paths[idx])
                
                # Apply text augmentation
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                caption = self._apply_text_transforms(caption)
                
                # Generate new text embeddings if caption changed
                if caption != cache_data.get('original_caption', ''):
                    text_inputs = self.tokenizer(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.text_encoder.device)
                    
                    text_inputs_2 = self.tokenizer_2(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.text_encoder_2.device)
                    
                    with torch.no_grad():
                        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                        text_embeddings_2 = self.text_encoder_2(
                            text_inputs_2.input_ids,
                            output_hidden_states=True
                        )
                        pooled_output = text_embeddings_2[0]
                        hidden_states = text_embeddings_2.hidden_states[-2]
                        
                    cache_data.update({
                        'text_embeddings': text_embeddings,
                        'text_embeddings_2': hidden_states,
                        'added_cond_kwargs': {
                            'text_embeds': pooled_output.unsqueeze(1).unsqueeze(2),
                            'time_ids': cache_data['added_cond_kwargs']['time_ids']
                        },
                        'original_caption': caption
                    })
                
                return cache_data
            
            # Process uncached item
            return self._process_uncached_item(img_path, caption_path)
        
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            return None

def custom_collate(batch):
    """
    Custom collate function for DataLoader that handles batches with same-sized images
    and supports aspect ratio bucketing.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Collated batch dictionary with:
        - pixel_values: Stacked image tensors
        - input_ids: Stacked token tensors
        - attention_mask: Stacked attention masks
        - bucket_size: Tuple of (height, width) for the current bucket
    """
    # Filter out None values that might come from failed loading
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
        
    # All images in a batch should have same dimensions due to bucketing
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch]) if "input_ids" in batch[0] else None
    attention_mask = torch.stack([item["attention_mask"] for item in batch]) if "attention_mask" in batch[0] else None
    
    # Get bucket size from first item
    bucket_size = batch[0]["pixel_values"].shape[-2:]  # (H, W)
    
    collated_batch = {
        "pixel_values": pixel_values,
        "bucket_size": bucket_size
    }
    
    if input_ids is not None:
        collated_batch["input_ids"] = input_ids
    if attention_mask is not None:
        collated_batch["attention_mask"] = attention_mask
        
    # Add any additional fields that are present in all batch items
    extra_keys = [k for k in batch[0].keys() if k not in ["pixel_values", "input_ids", "attention_mask", "bucket_size"]]
    for k in extra_keys:
        if isinstance(batch[0][k], torch.Tensor):
            collated_batch[k] = torch.stack([item[k] for item in batch])
        else:
            collated_batch[k] = [item[k] for item in batch]
            
    return collated_batch

def create_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = None,
    cache_dir: str = "latents_cache",
    all_ar: bool = False,
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
):
    """
    Create a DataLoader with the CustomDataset.
    
    Args:
        data_dir (str): Directory containing the training images
        batch_size (int): Batch size for training
        num_workers (int, optional): Number of worker processes
        cache_dir (str, optional): Directory to cache latents
        all_ar (bool, optional): Accept all aspect ratios without resizing
        shuffle (bool, optional): Whether to shuffle the dataset
        pin_memory (bool, optional): Use pinned memory for faster GPU transfer
        prefetch_factor (int, optional): Number of batches to prefetch per worker
        persistent_workers (bool, optional): Keep worker processes alive between epochs
    
    Returns:
        torch.utils.data.DataLoader: Configured data loader
    """
    try:
        # Initialize dataset
        dataset = CustomDataset(
            data_dir=data_dir,
            cache_dir=cache_dir,
            all_ar=all_ar,
            num_workers=num_workers
        )
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers is not None else min(8, os.cpu_count() or 1),
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=custom_collate
        )
        
        logger.info(f"Created DataLoader with {len(dataset)} samples")
        logger.info(f"Batch size: {batch_size}, Num workers: {num_workers}")
        
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {str(e)}")
        logger.error(traceback.format_exc())
        raise
