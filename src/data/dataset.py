from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
from tqdm import tqdm
import logging
from diffusers import CLIPModel, CLIPProcessor
from utils.device import to_device

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
        """Optimized version of caching that uses bfloat16"""
        try:
            with to_device(self.vae, "cuda") as vae, \
                 to_device(self.text_encoder, "cuda") as text_encoder, \
                 to_device(self.text_encoder_2, "cuda") as text_encoder_2, \
                 to_device(self.clip_model, "cuda") as clip_model:
                
                vae.eval()
                text_encoder.eval()
                text_encoder_2.eval()
                clip_model.eval()
                
                # Add progress bar
                print("Caching latents and embeddings...")
                for img_path, caption_path in tqdm(zip(self.image_paths, self.caption_paths),
                                                   total=len(self.image_paths),
                                                   desc="Caching"):
                    cache_latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
                    cache_embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"

                    if not cache_latents_path.exists() or not cache_embeddings_path.exists():
                        image = Image.open(img_path).convert("RGB")
                        image_tensor = self.transform_image(image).unsqueeze(0)  # Add batch dim only for VAE

                        with torch.no_grad():
                            # Get tags from caption
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                                tags = [t.strip() for t in caption.split(',')]

                            # Get CLIP embeddings
                            clip_image_embed, clip_tag_embeds = self._get_clip_embeddings(image, tags)

                            # Regular processing
                            image_tensor = image_tensor.to("cuda", dtype=torch.bfloat16)
                            latents = self.vae.encode(image_tensor).latent_dist.sample()  # [1, 4, H/8, W/8]
                            latents = latents * 0.18215  # Scaling factor

                            # Process text embeddings
                            text_input = self.tokenizer(
                                caption,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to("cuda")

                            text_embeddings = self.text_encoder(text_input.input_ids)[0]  # [1, 77, 768]

                            # SDXL text encoder 2
                            text_input_2 = self.tokenizer_2(
                                caption,
                                padding="max_length",
                                max_length=self.tokenizer_2.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to("cuda")

                            text_encoder_output_2 = self.text_encoder_2(text_input_2.input_ids)
                            text_embeddings_2 = text_encoder_output_2.last_hidden_state  # [1, 77, 1280]
                            pooled_text_embeddings_2 = text_encoder_output_2.pooler_output  # [1, 1280]

                            # Save all embeddings
                            torch.save({
                                "text_embeddings": text_embeddings.cpu(),
                                "text_embeddings_2": text_embeddings_2.cpu(),
                                "pooled_text_embeddings_2": pooled_text_embeddings_2.cpu(),
                                "clip_image_embed": clip_image_embed.cpu(),
                                "clip_tag_embeds": clip_tag_embeds.cpu(),
                                "tags": tags
                            }, cache_embeddings_path)

                            torch.save(latents.cpu(), cache_latents_path)

                            # Clear memory
                            del image_tensor, latents, text_embeddings, text_embeddings_2, pooled_text_embeddings_2
                            torch.cuda.empty_cache()

            # Move models back to CPU after the loop
            self.vae.to("cpu")
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
            self.clip_model.to("cpu")

        except Exception as e:
            logger.error(f"Caching failed: {str(e)}")
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