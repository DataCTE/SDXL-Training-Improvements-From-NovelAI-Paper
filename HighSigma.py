import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import logging
import argparse
import math
from PIL import Image
from bitsandbytes.optim import AdamW8bit
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from transformers.optimization import Adafactor
import torchvision.models as models
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import os
import wandb
import json
import traceback

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# Global bfloat16 settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_sigmas(num_inference_steps=28, sigma_min=0.0292, sigma_max=20000.0):
    """
    Generate sigmas using a schedule that supports Zero Terminal SNR (ZTSNR)
    Args:
        num_inference_steps: Number of inference steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value (set to a large number for ZTSNR)
    Returns:
        Tensor of sigma values
    """
    t = torch.linspace(1, 0, num_inference_steps)
    rho = 7.0  # Hyperparameter, can be adjusted
    sigmas = (sigma_max**(1 / rho) + t * (sigma_min**(1 / rho) - sigma_max**(1 / rho))) ** rho
    return sigmas

def compute_snr(sigma):
    """Compute Signal-to-Noise Ratio (SNR) from sigma."""
    return 1 / (sigma**2)


def training_loss(model, x_0, sigma, text_embeddings, text_embeddings_2, pooled_text_embeds_2, timestep, target_size):
    #print("\n=== Starting training_loss ===")
    #print(f"Initial shapes:")
    #print(f"x_0: {x_0.shape}")
    #print(f"text_embeddings: {text_embeddings.shape}")
    #print(f"text_embeddings_2: {text_embeddings_2.shape}")
    #print(f"pooled_text_embeds_2: {pooled_text_embeds_2.shape}")
    #print(f"target_size: {target_size}")
    #print(f"sigma: {sigma.shape}")
    #print(f"timestep: {timestep.shape}")

    # Remove extra dimensions to get to [B, 4, H, W]
    while x_0.dim() > 4:
        x_0 = x_0.squeeze(1)
    #print(f"x_0 after squeeze: {x_0.shape}")

    batch_size = x_0.shape[0]
    #print(f"batch_size: {batch_size}")

    # Fix tensor shapes
    # Remove extra dimension from pooled_text_embeds_2
    # Handle only the specific error case where dim=1 doesn't exist
    if pooled_text_embeds_2.dim() == 1:  # Handle 1D case
        pooled_text_embeds_2 = pooled_text_embeds_2.unsqueeze(0)  # Add batch dimension
    
    # Safe squeeze operation
    if pooled_text_embeds_2.dim() > 2:
        pooled_text_embeds_2 = pooled_text_embeds_2.squeeze()
        # Ensure we have at least 2 dimensions
        if pooled_text_embeds_2.dim() == 1:
            pooled_text_embeds_2 = pooled_text_embeds_2.unsqueeze(0)
    
    # Validate final shape
    assert pooled_text_embeds_2.dim() == 2, f"Expected 2D tensor [B, 1280], got shape {pooled_text_embeds_2.shape}"
    
    # Fix text_embeddings shape - need one more squeeze for batch processing
    text_embeddings = text_embeddings.squeeze(1)  # [B, 77, 768]
    text_embeddings_2 = text_embeddings_2.squeeze(1)  # [B, 77, 1280]
    #print(f"text_embeddings after squeeze: {text_embeddings.shape}")
    #print(f"text_embeddings_2 after squeeze: {text_embeddings_2.shape}")
    
    # Create micro-conditioning tensors
    if isinstance(target_size, list):
        # Convert list to tuple for hashing
        target_size = tuple(target_size[0])  # Convert first target size to tuple
        
        time_ids = torch.tensor([
            target_size[0],  # height
            target_size[1],  # width
            target_size[0],  # target height
            target_size[1],  # target width
            0,  # crop top
            0,  # crop left
        ], device=x_0.device, dtype=torch.float32)
        
        # Repeat for batch size
        time_ids = time_ids.unsqueeze(0).repeat(batch_size, 1)  # [B, 6]
    else:
        # Handle single target_size case
        target_size = tuple(target_size)  # Convert to tuple for consistency
        
        time_ids = torch.tensor([
            target_size[0],
            target_size[1],
            target_size[0],
            target_size[1],
            0,
            0,
        ], device=x_0.device, dtype=torch.float32)
        
        time_ids = time_ids.unsqueeze(0).repeat(batch_size, 1)  # [B, 6]
    
    #print(f"time_ids final shape: {time_ids.shape}")

    # Prepare SDXL conditioning kwargs
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds_2,  # [B, 1280]
        "time_ids": time_ids  # [B, 6]
    }

    # Combine text embeddings
    combined_text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)  # [B, 77, 2048]
    #print(f"combined_text_embeddings shape: {combined_text_embeddings.shape}")

    # Generate noise and add to input
    noise = torch.randn_like(x_0)  # [B, 4, H/8, W/8]
    sigma = sigma.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    x_t = x_0 + sigma * noise  # [B, 4, H/8, W/8]
    #print(f"noise shape: {noise.shape}")
    #print(f"sigma shape after view: {sigma.shape}")
    #print(f"x_t shape: {x_t.shape}")

    # V-prediction target
    v = (x_t - x_0) / (sigma**2 + 1).sqrt()
    target = v  # [B, 4, H/8, W/8]
    #print(f"target shape: {target.shape}")

    #print("\nStarting UNet forward pass...")
    # UNet forward pass
    model_output = model(
        sample=x_t,
        timestep=timestep,
        encoder_hidden_states=combined_text_embeddings,
        added_cond_kwargs=added_cond_kwargs
    ).sample
    #print(f"model_output shape: {model_output.shape}")

    # MinSNR loss weighting
    snr = compute_snr(sigma.squeeze())  # [B]
    gamma = 1.0  # Hyperparameter, can be tuned
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    mse_loss = F.mse_loss(model_output, target, reduction='none')
    loss = (min_snr_gamma.view(-1, 1, 1, 1) * mse_loss).mean()
    #print(f"final loss: {loss.item()}")
    #print("=== Finished training_loss ===\n")

    return loss


class TagBasedLossWeighter:
    def __init__(self, tag_classes=None, min_weight=0.1, max_weight=3.0):
        """
        Initialize the tag-based loss weighting system.
        
        Args:
            tag_classes (dict): Dictionary mapping tag class names to lists of tags
                              e.g., {'character': ['girl', 'boy'], 'style': ['anime', 'sketch']}
            min_weight (float): Minimum weight multiplier for any image
            max_weight (float): Maximum weight multiplier for any image
        """
        self.tag_classes = tag_classes or {
            'character': set(),
            'style': set(),
            'setting': set(),
            'action': set(),
            'object': set(),
            'quality': set()
        }
        
        # Track tag frequencies per class
        self.tag_frequencies = {class_name: {} for class_name in self.tag_classes}
        self.class_total_counts = {class_name: 0 for class_name in self.tag_classes}
        
        # Parameters for weight calculation
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Cache for tag classifications
        self.tag_to_class = {}
        
    def add_tags_to_frequency(self, tags, count=1):
        """
        Update tag frequencies with new tags.
        
        Args:
            tags (list): List of tags from an image
            count (int): How many times to count these tags (default: 1)
        """
        for tag in tags:
            # Find which class this tag belongs to (if not already cached)
            if tag not in self.tag_to_class:
                for class_name, tag_set in self.tag_classes.items():
                    if tag in tag_set:
                        self.tag_to_class[tag] = class_name
                        break
                else:
                    continue  # Skip tags that don't belong to any class
            
            # Update frequencies
            class_name = self.tag_to_class.get(tag)
            if class_name:
                self.tag_frequencies[class_name][tag] = (
                    self.tag_frequencies[class_name].get(tag, 0) + count
                )
                self.class_total_counts[class_name] += count

    def calculate_tag_weights(self, tags):
        try:
            # Convert lists to sets for membership testing
            if isinstance(tags, list):
                tags = set(tags)
            
            weights = []
            for class_name in self.tag_classes:
                try:
                    # Convert class tags to set if it's a list
                    if isinstance(self.tag_classes[class_name], list):
                        self.tag_classes[class_name] = set(self.tag_classes[class_name])
                    
                    class_tags = tags.intersection(self.tag_classes[class_name])
                    weight = self.class_weights.get(class_name, 1.0)
                    weights.append(weight if class_tags else 1.0)
                    
                except Exception as class_error:
                    logger.error(f"Error processing class {class_name}: {str(class_error)}")
                    logger.error(f"Class traceback: {traceback.format_exc()}")
                    weights.append(1.0)
            
            return torch.tensor(weights).mean()
            
        except Exception as e:
            logger.error(f"Tag weight calculation failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return torch.tensor(1.0)

    def update_training_loss(self, loss, tags):
        """
        Apply tag-based weighting to the training loss.
        
        Args:
            loss (torch.Tensor): Original loss value
            tags (list): List of tags for the current image/batch
            
        Returns:
            torch.Tensor: Weighted loss value
        """
        try:
            weight = self.calculate_tag_weights(tags)
            return loss * weight
        except Exception as e:
            logger.error(f"Loss update failed: {str(e)}")
            logger.error(f"Loss update traceback: {traceback.format_exc()}")
            return loss  # Return original loss on error
    
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
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.clip_model.eval()

        self.vae.to("cuda")
        self.text_encoder.to("cuda")
        self.text_encoder_2.to("cuda")
        self.clip_model.to("cuda")

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

class PerceptualLoss:
    def __init__(self):
        # Use pre-trained VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to("cuda")
        # Convert VGG to bfloat16
        self.vgg = self.vgg.to(dtype=torch.bfloat16)
        self.vgg.requires_grad_(False)
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def get_features(self, x):
        # Ensure input is in bfloat16
        x = x.to(dtype=torch.bfloat16)
        x = self.normalize(x)
        features = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def __call__(self, pred, target):
        # Ensure inputs are in bfloat16
        pred = pred.to(dtype=torch.bfloat16)
        target = target.to(dtype=torch.bfloat16)
        
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)

        loss = 0.0
        for key in pred_features:
            loss += F.mse_loss(pred_features[key], target_features[key])
        return loss

def custom_collate(batch):
    """
    Custom collate function for DataLoader that handles both single and batched samples.
    
    Args:
        batch: List of dictionaries containing dataset items
            For batch_size=1: List with single dictionary
            For batch_size>1: List of multiple dictionaries
    
    Returns:
        For batch_size=1: Original dictionary without any stacking
        For batch_size>1: Dictionary with properly stacked tensors and lists
    """
    # Get batch size from first item
    batch_size = len(batch)
    
    if batch_size == 1:
        # For single samples, return the dictionary directly without any stacking
        return batch[0]
    else:
        # For batched samples, we need to stack the tensors properly
        elem = batch[0]  # Get first item to determine dictionary structure
        collated = {}
        
        for key in elem:
            if key == "tags":
                # Tags are lists of strings, so we keep them as a list of lists
                collated[key] = [d[key] for d in batch]
            elif key == "target_size":
                # target_size needs to remain as separate tensors for each batch item
                collated[key] = [d[key] for d in batch]
            else:
                try:
                    # Try to stack tensors along a new batch dimension
                    collated[key] = torch.stack([d[key] for d in batch])
                except:
                    # If stacking fails, keep as list (for non-tensor data)
                    collated[key] = [d[key] for d in batch]
        
        return collated

class VAEFineTuner:
    def __init__(self, vae, learning_rate=1e-6):
        try:
            self.vae = vae
            self.optimizer = AdamW8bit(vae.parameters(), lr=learning_rate)
            
            # Initialize Welford's online statistics
            self.latent_count = 0
            self.latent_means = None
            self.latent_m2 = None
            
            logger.info("VAEFineTuner initialized successfully")
            
        except Exception as e:
            logger.error(f"VAEFineTuner initialization failed: {str(e)}")
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise
            
    def update_statistics(self, latents):
        """Update running mean and variance using Welford's online algorithm"""
        try:
            if self.latent_means is None:
                self.latent_means = torch.zeros(latents.size(1), device=latents.device)
                self.latent_m2 = torch.zeros(latents.size(1), device=latents.device)
                
            flat_latents = latents.view(latents.size(0), latents.size(1), -1)
            
            for i in range(latents.size(0)):
                self.latent_count += 1
                delta = flat_latents[i].mean(dim=1) - self.latent_means
                self.latent_means += delta / self.latent_count
                delta2 = flat_latents[i].mean(dim=1) - self.latent_means
                self.latent_m2 += delta * delta2
                
        except Exception as e:
            logger.error(f"Statistics update failed: {str(e)}")
            logger.error(f"Update traceback: {traceback.format_exc()}")
            logger.error(f"Latents shape: {latents.shape}")
            logger.error(f"Current means shape: {self.latent_means.shape if self.latent_means is not None else None}")
            
    def get_statistics(self):
        try:
            if self.latent_count < 2:
                return None, None
                
            variance = self.latent_m2 / (self.latent_count - 1)
            std = torch.sqrt(variance + 1e-8)
            return self.latent_means, std
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            logger.error(f"Calculation traceback: {traceback.format_exc()}")
            return None, None
            
    def training_step(self, batch):
        try:
            self.optimizer.zero_grad()
            
            # Log batch info
            logger.debug(f"Processing batch with keys: {batch.keys()}")
            
            # Get input images
            images = batch["pixel_values"].to(self.vae.device)
            logger.debug(f"Input images shape: {images.shape}")
            
            # Encode
            latents = self.vae.encode(images).latent_dist.sample()
            logger.debug(f"Encoded latents shape: {latents.shape}")
            
            # Update statistics
            self.update_statistics(latents.detach())
            
            # Get current statistics
            means, stds = self.get_statistics()
            if means is not None and stds is not None:
                latents = (latents - means[None,:,None,None]) / stds[None,:,None,None]
                decode_latents = latents * stds[None,:,None,None] + means[None,:,None,None]
            else:
                decode_latents = latents
                
            # Decode
            decoded = self.vae.decode(decode_latents).sample
            logger.debug(f"Decoded images shape: {decoded.shape}")
            
            # Calculate loss
            loss = F.mse_loss(decoded, images, reduction="mean")
            
            loss.backward()
            self.optimizer.step()
            
            return {
                "loss": loss.item(),
                "latent_means": self.latent_means.detach().cpu() if self.latent_means is not None else None,
                "latent_stds": stds.detach().cpu() if stds is not None else None
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            logger.error(f"Training traceback: {traceback.format_exc()}")
            logger.error(f"Batch keys: {batch.keys()}")
            logger.error(f"Device info - VAE: {self.vae.device}, Images: {images.device if 'images' in locals() else 'N/A'}")
            return {"loss": float('inf')}  # Return infinite loss on error
            
    def save_pretrained(self, path):
        """Custom save method"""
        try:
            os.makedirs(path, exist_ok=True)
            self.vae.save_pretrained(path)
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
            
            # Save statistics if available
            if self.latent_means is not None and self.latent_m2 is not None:
                stats = {
                    "means": self.latent_means.cpu(),
                    "m2": self.latent_m2.cpu(),
                    "count": self.latent_count
                }
                torch.save(stats, os.path.join(path, "latent_stats.pt"))
                
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
            logger.error(f"Save traceback: {traceback.format_exc()}")
            raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--finetune_vae", action="store_true", help="Enable VAE finetuning")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6, help="VAE learning rate")
    parser.add_argument("--vae_train_freq", type=int, default=10, help="VAE training frequency")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--use_adafactor", action="store_true", help="Use Adafactor instead of AdamW8bit")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients"
    )
    # Removed compile_mode argument
    parser.add_argument("--enable_compile", action="store_true", help="Enable model compilation")
    parser.add_argument("--compile_mode", type=str, choices=['default', 'reduce-overhead', 'max-autotune'], default='default', help="Torch compile mode")
    
    parser.add_argument("--save_checkpoints", action="store_true", help="Save checkpoints after each epoch")
    parser.add_argument("--min_tag_weight", type=float, default=0.1, help="Minimum tag-based loss weight")
    parser.add_argument("--max_tag_weight", type=float, default=3.0, help="Maximum tag-based loss weight")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sdxl-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HuggingFace Hub model ID")
    parser.add_argument("--hub_private", action="store_true", help="Make the HuggingFace repo private")
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=1,  # Run validation every epoch by default
        help="Number of epochs between validation runs"
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=[
            "a detailed portrait of a girl",
            "completely black",
            "a red ball on top of a blue cube, both infront of a green triangle"
        ],
        help="Prompts to use for validation"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip validation entirely"
    )

    args = parser.parse_args()
    return args



    
def main(args):
    try:
        # Initialize wandb if enabled
        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                settings=wandb.Settings(code_dir="."),
            )
            
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Set up dtype
        dtype = torch.bfloat16

        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load models with bfloat16
        unet = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=torch.bfloat16
        ).to(device)

        vae = AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        )
        text_encoder_2 = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16
        )
        

        # Keep other models on CPU
        vae.to("cpu")
        text_encoder.to("cpu")
        text_encoder_2.to("cpu")

        

        # Enable gradient checkpointing
        unet.enable_gradient_checkpointing()

        # Setup optimizer with per-device batch size
        dataset = CustomDataset(
            args.data_dir,
            vae,
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )

        # Setup optimizer
        if args.use_adafactor:
            optimizer = Adafactor(
                unet.parameters(),
                lr=args.learning_rate * args.batch_size,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
                clip_threshold=1.0,
                beta1=0.9,
                weight_decay=1e-2,
            )
        else:
            optimizer = AdamW8bit(
                unet.parameters(),
                lr=args.learning_rate * args.batch_size,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-8
            )

        # Enable memory efficient attention only for models that support it
        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

        # Setup EMA
        ema_model = AveragedModel(unet, avg_fn=lambda avg, new, _: args.ema_decay * avg + (1 - args.ema_decay) * new)

        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_training_steps = args.num_epochs * num_update_steps_per_epoch

        # Setup scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Initialize training components
        if args.finetune_vae:
            vae = vae.to("cuda").to(dtype=torch.bfloat16)
            vae_finetuner = VAEFineTuner(
                vae=vae,
                learning_rate=args.vae_learning_rate
            )

        # Initialize validator after model setup
        validator = ModelValidator(
            model=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device
        )   

        # Move models to device and set dtype
        unet = unet.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)

        # Initialize tag weighter with dataset's tag list
        tag_weighter = TagBasedLossWeighter(
            min_weight=args.min_tag_weight,
            max_weight=args.max_tag_weight
        )

        # Populate tag frequencies from dataset
        print("Initializing tag frequencies...")
        for img_path in tqdm(dataset.image_paths):
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                    tags = [t.strip() for t in caption.split(',')]
                    tag_weighter.add_tags_to_frequency(tags)

        # Training loop
        logger.info("Starting training...")
        total_steps = args.num_epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

        for epoch in range(args.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")

            # Run validation if enabled
            if not args.skip_validation and hasattr(args, 'validation_frequency') and \
               epoch % args.validation_frequency == 0:
                try:
                    logger.info(f"\nRunning validation for epoch {epoch+1}")
                    test_prompts = [
                        "a detailed portrait of a girl",
                        "completely black",
                        "a red ball on top of a blue cube, both infront of a green triangle"
                    ]
                    
                    validation_results = validator.run_paper_validation(test_prompts)
                    
                    # Save validation images
                    validation_dir = Path(args.output_dir) / "validation_results" / f"epoch_{epoch+1}"
                    validator.save_validation_images(validation_results, validation_dir)
                    
                    if args.use_wandb:
                        wandb.log({
                            'ztsnr/black_generation_brightness': validation_results['ztsnr'][1]['ztsnr_mean_brightness'],
                            'epoch': epoch + 1
                        })
                except Exception as val_error:
                    logger.error(f"Validation failed: {val_error}")
                    logger.info("Continuing training despite validation error")

            for step, batch in enumerate(train_dataloader):
                # Convert inputs to correct dtype (bfloat16)
                latents = batch["latents"].to(device).to(dtype=torch.bfloat16)
                text_embeddings = batch["text_embeddings"].to(device).to(dtype=torch.bfloat16)
                text_embeddings_2 = batch["text_embeddings_2"].to(device).to(dtype=torch.bfloat16)
                pooled_text_embeddings_2 = batch["pooled_text_embeddings_2"].to(device).to(dtype=torch.bfloat16)
                target_size = batch["target_size"]

                # Use autocast for mixed precision
                with torch.amp.autocast('cuda', dtype=dtype):
                    sigmas = get_sigmas(args.num_inference_steps).to(device)
                    sigma = sigmas[step % args.num_inference_steps]
                    sigma = sigma.expand(latents.size(0))

                    timestep = torch.ones(latents.shape[0], device=device).long() * (step % args.num_inference_steps)

                    loss = training_loss(
                        unet,
                        latents,
                        sigma,
                        text_embeddings,
                        text_embeddings_2,
                        pooled_text_embeddings_2,
                        timestep,
                        target_size
                    ) / args.gradient_accumulation_steps

                # Apply tag-based weighting
                loss = tag_weighter.update_training_loss(loss, batch["tags"])

                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    ema_model.update_parameters(unet)

                    if args.finetune_vae and step % args.vae_train_freq == 0:
                        vae_losses = vae_finetuner.training_step(
                            batch["latents"].to("cuda"),
                            batch["original_images"].to("cuda")
                        )

                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "epoch": f"{epoch+1}/{args.num_epochs}",
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    **({"vae_loss": f"{vae_losses['total_loss']:.4f}"} if args.finetune_vae and step % args.vae_train_freq == 0 else {})
                })

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        try:
            # Save model
            if args.push_to_hub:
                logger.info("Pushing model to hub...")
                unet.push_to_hub(
                    args.hub_model_id,
                    private=args.hub_private,
                    commit_message=f"Epoch {args.num_epochs}"
                )
            
            # Local save
            logger.info("Saving model locally...")
            save_path = os.path.join(args.output_dir, "final_model")
            unet.save_pretrained(save_path)
            
            # Save config with proper serialization
            save_training_config(
                args=args,
                output_dir=args.output_dir,
                total_steps=total_steps,
                final_loss=loss.item() if 'loss' in locals() else None
            )
            
        except Exception as save_error:
            logger.error(f"Error during model saving: {save_error}")
            # Emergency save
            try:
                emergency_path = os.path.join(args.output_dir, "emergency_save")
                torch.save(unet.state_dict(), os.path.join(emergency_path, "unet.pt"))
                logger.info(f"Emergency save successful at: {emergency_path}")
            except:
                logger.error("Emergency save failed")

    if args.use_wandb:
        wandb.finish()

def save_image_grid(images, path, nrow=1, normalize=True):
    """Save a list of images as a grid"""
    grid = make_grid(images, nrow=nrow, normalize=normalize)
    TF.to_pil_image(grid).save(path)

class ModelValidator:
    def __init__(self, model, vae, tokenizer, text_encoder, device="cuda"):
        self.model = model.to(device)
        self.vae = vae.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.device = device

    def generate_at_sigma(self, prompt, target_sigma, sigma_max=20000.0):
        """Generate a sample denoising from sigma_max down to target_sigma"""
        try:
            # Create custom sigma schedule from sigma_max down to target_sigma
            sigmas = torch.linspace(sigma_max, target_sigma, steps=10).to(self.device)
            
            # Initialize random latents
            latents = torch.randn((1, 4, 64, 64)).to(self.device)
            
            # Encode text prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                prompt_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
                
                # Denoise from sigma_max down to target_sigma
                for sigma in sigmas:
                    noise_pred = self.model(
                        latents,
                        sigma[None].to(self.device),
                        encoder_hidden_states=prompt_embeds
                    ).sample
                    
                    latents = latents - noise_pred * (sigma ** 2)
                
                # Decode final latents
                image = self.vae.decode(latents / 0.18215).sample
                
            return image
            
        except Exception as e:
            logger.error(f"Sample generation at sigma {target_sigma} failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return torch.zeros((1, 3, 1024, 1024)).to(self.device)  # Updated size

    def validate_ztsnr(self, prompt="completely black"):
        """
        Validate Zero Terminal SNR as shown in Figure 2 of the paper.
        Tests if model can generate pure black images when prompted.
        """
        results = {
            'ztsnr': self.generate_with_ztsnr(prompt),
            'no_ztsnr': self.generate_without_ztsnr(prompt)
        }
        
        metrics = {
            'ztsnr_mean_brightness': results['ztsnr'].mean(),
            'no_ztsnr_mean_brightness': results['no_ztsnr'].mean()
        }
        
        return results, metrics

    def validate_high_res_coherence(self, prompt, sigma_max_values=[14.6, 20000.0]):
        """
        Validate high-resolution coherence as shown in Figure 6 of the paper.
        Tests if model maintains coherence with different sigma_max values.
        Note: σ ≈ 20000 is used as a practical approximation of infinity for ZTSNR.
        """
        results = {}
        step_sigmas = {
            14.6: [14.6, 10.8, 8.3, 6.6, 5.4],  # From Figure 7
            20000.0: [20000.0, 17.8, 12.4, 9.2, 7.2]  # ZTSNR regime
        }
        
        for sigma_max in sigma_max_values:
            steps = []
            for sigma in step_sigmas[sigma_max]:
                step = self.generate_at_sigma(prompt, sigma, sigma_max)
                steps.append(step)
            results[f'sigma_max_{sigma_max}'] = steps
            
        return results

    def validate_noise_schedule(self, image):
        """
        Validate noise schedule as shown in Figure 1 of the paper.
        Shows progressive noise addition up to final training timestep.
        """
        # Sigmas from Figure 1
        sigmas = [0, 0.447, 3.17, 14.6]
        noised_images = []
        
        for sigma in sigmas:
            noise = torch.randn_like(image)
            noised = image + sigma * noise
            noised_images.append(noised)
            
        return noised_images

    def run_paper_validation(self, prompts):
        results = {}
        try:
            # ZTSNR validation (using black prompt)
            logger.info("Running ZTSNR validation...")
            results['ztsnr'] = self.validate_ztsnr("completely black")
            
            # High-res coherence validation (using first prompt)
            logger.info("Running high-res coherence validation...")
            results['coherence'] = self.validate_high_res_coherence(prompts[0])
            
            # Prompt-based validation
            logger.info("Running prompt-based validation...")
            for prompt in prompts:
                results[prompt] = self.generate_sample(prompt)
                
            return results
            
        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}  # Return empty dict instead of failing

    def save_validation_images(self, results, output_dir):
        """Save validation images in a format matching paper figures"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ZTSNR comparison (Figure 2)
        save_image_grid(
            [results['ztsnr']['ztsnr'], results['ztsnr']['no_ztsnr']],
            os.path.join(output_dir, 'ztsnr_comparison.png'),
            nrow=2,
            normalize=True
        )
        
        # Save high-res coherence comparison (Figure 6)
        save_image_grid(
            [results['coherence']['default_sigma'], 
             results['coherence']['high_sigma']],
            os.path.join(output_dir, 'coherence_comparison.png'),
            nrow=2,
            normalize=True
        )
        
        # Save denoising steps (Figure 7)
        for sigma_max, steps in results['denoising'].items():
            save_image_grid(
                steps,
                os.path.join(output_dir, f'denoising_{sigma_max}.png'),
                nrow=len(steps),
                normalize=True
            )

    def generate_with_ztsnr(self, prompt):
        """Generate image with ZTSNR (σ ≈ 20000)"""
        return self.generate_sample(prompt, sigma_max=20000.0)

    def generate_without_ztsnr(self, prompt):
        """Generate image without ZTSNR (default σ = 14.6)"""
        return self.generate_sample(prompt, sigma_max=14.6)

    def generate_sample(self, prompt, sigma_max=14.6, num_inference_steps=28):
        """Generate a sample image given a prompt"""
        try:
            # Get sigmas for sampling
            sigmas = get_sigmas(num_inference_steps, sigma_max=sigma_max).to(self.device)
            
            # Initialize random latents
            latents = torch.randn((1, 4, 64, 64)).to(self.device)
            
            # Encode text prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                prompt_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            # Denoise
            for i, sigma in enumerate(sigmas):
                with torch.no_grad():
                    noise_pred = self.model(
                        latents,
                        sigma[None].to(self.device),
                        encoder_hidden_states=prompt_embeds
                    ).sample
                
                latents = latents - noise_pred * (sigma[None, None, None] ** 2)
                
                if i < len(sigmas) - 1:
                    noise = torch.randn_like(latents)
                    latents = latents + noise * (sigmas[i + 1] ** 2 - sigmas[i] ** 2) ** 0.5
            
            # Decode latents
            with torch.no_grad():
                image = self.vae.decode(latents / 0.18215).sample
                
            return image
            
        except Exception as e:
            logger.error(f"Sample generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return torch.zeros((1, 3, 1024, 1024)).to(self.device)  # Updated size

        

def save_training_config(args, output_dir, total_steps, final_loss=None):
    """Save training config with proper type handling"""
    config = {
        'args': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                for k, v in vars(args).items()},
        'training_steps': total_steps,
        'final_loss': float(final_loss) if final_loss is not None else None,
    }
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved training config to {config_path}")

def create_model_card(args, train_dataloader, validation_results=None):
    model_card = f"""
    ---
    language: en
    tags:
    - stable-diffusion-xl
    - text-to-image
    - diffusers
    - ztsnr
    license: mit
    ---

    # {args.hub_model_id}

    ## Training Details
    - Base model: SDXL 1.0
    - Training steps: {args.num_epochs * len(train_dataloader)}
    - Batch size: {args.batch_size}
    - Learning rate: {args.learning_rate}
    - Validation frequency: {args.validation_frequency} epochs
    - Validation prompts: {args.validation_prompts}

    ## Validation Results
    """
    
    if validation_results:
        model_card += "Latest validation results included in the validation_results directory."
    else:
        model_card += "No validation results available."
    
    return model_card

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)
