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

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                 cache_dir="latents_cache", batch_size=1):
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
        """Calculate target size maintaining aspect ratio and ~1024x1024 total pixels
        Args:
            ratio: (width, height) tuple of aspect ratio
        Returns:
            (height, width) tuple of target size
        """
        # Calculate scale to maintain ~1024x1024 pixels
        scale = math.sqrt(1024 * 1024 / (ratio[0] * ratio[1]))

        # Calculate dimensions and ensure multiple of 64
        w = int(round(ratio[0] * scale / 64)) * 64
        h = int(round(ratio[1] * scale / 64)) * 64

        # Clamp to max dimension of 2048
        w = min(2048, w)
        h = min(2048, h)

        return (h, w)

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
        """Transform image to tensor without extra batch dimension"""
        transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return transform(image)  # Remove unsqueeze(0)

    def _cache_latents_and_embeddings_optimized(self):
        """Optimized version of caching that uses bfloat16 and batch processing"""
        try:
            # First check which files need processing
            files_to_process = []
            for img_path, caption_path in zip(self.image_paths, self.caption_paths):
                cache_latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
                cache_embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"
                
                if not cache_latents_path.exists() or not cache_embeddings_path.exists():
                    files_to_process.append((img_path, caption_path))
            
            if not files_to_process:
                logger.info("All latents and embeddings already cached, skipping...")
                return

            # Process in batches of 8 (or adjust based on your GPU memory)
            BATCH_SIZE = 8
            
            with to_device(self.vae, "cuda") as vae, \
                 to_device(self.text_encoder, "cuda") as text_encoder, \
                 to_device(self.text_encoder_2, "cuda") as text_encoder_2, \
                 to_device(self.clip_model, "cuda") as clip_model:
                
                vae.eval()
                text_encoder.eval()
                text_encoder_2.eval()
                clip_model.eval()

                # Process files in batches
                logger.info(f"Caching {len(files_to_process)} new files...")
                for i in tqdm(range(0, len(files_to_process), BATCH_SIZE), desc="Caching"):
                    batch_files = files_to_process[i:i + BATCH_SIZE]
                    
                    # Prepare batch data
                    vae_images = []  # For VAE
                    clip_images = []  # For CLIP
                    captions = []
                    for img_path, caption_path in batch_files:
                        image = Image.open(img_path).convert("RGB")
                        # Store original image for CLIP
                        clip_images.append(image)
                        # Transform image for VAE
                        vae_images.append(self.transform_image(image))
                        
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            captions.append(f.read().strip())
                    
                    # Stack images for VAE
                    image_batch = torch.stack(vae_images).to("cuda", dtype=torch.bfloat16)
                    
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
                        cache_latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
                        cache_embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"
                        
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
        """Get item with separate handling for batch_size=1 and batch_size>1"""
        img_path = self.image_paths[idx]
        latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
        embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"

        # Load cached data
        latents = torch.load(latents_path, map_location='cpu', weights_only=True)
        embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=True)

        # Load original image
        image = Image.open(img_path).convert("RGB")
        original_images = self.transform_image(image)  # [3, H, W]

        if self.batch_size == 1:
            # Original single sample processing
            return {
                "latents": latents,  # [1, 4, 128, 128]
                "text_embeddings": embeddings["text_embeddings"],  # [1, 77, 768]
                "text_embeddings_2": embeddings["text_embeddings_2"],  # [1, 77, 1280]
                "pooled_text_embeddings_2": embeddings["pooled_text_embeddings_2"],  # [1, 1280]
                "target_size": (original_images.shape[1], original_images.shape[2]),  # tuple of (H, W)
                "clip_image_embed": embeddings["clip_image_embed"],
                "clip_tag_embeds": embeddings["clip_tag_embeds"],
                "tags": embeddings["tags"],
                "original_images": original_images  # [3, H, W]
            }
        else:
            # Batch processing
            return {
                "latents": latents.squeeze(0),  # [4, 128, 128]
                "text_embeddings": embeddings["text_embeddings"].squeeze(0),  # [77, 768]
                "text_embeddings_2": embeddings["text_embeddings_2"].squeeze(0),  # [77, 1280]
                "pooled_text_embeddings_2": embeddings["pooled_text_embeddings_2"].squeeze(0),  # [1280]
                "target_size": torch.tensor([original_images.shape[1], original_images.shape[2]], dtype=torch.long),  # [2]
                "clip_image_embed": embeddings["clip_image_embed"].squeeze(0),
                "clip_tag_embeds": embeddings["clip_tag_embeds"].squeeze(0),
                "tags": embeddings["tags"],
                "original_images": original_images  # [3, H, W]
            }