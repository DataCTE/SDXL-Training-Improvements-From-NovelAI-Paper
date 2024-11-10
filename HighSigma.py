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
from transformers.optimization import Adafactor
import torchvision.models as models
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import os
import wandb
import json
import traceback
from collections import defaultdict

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
        sigma_min: Minimum sigma value (≈0.0292 from paper)
        sigma_max: Maximum sigma value (set to 20000 for practical ZTSNR)
    Returns:
        Tensor of sigma values
    """
    # Use rho=7.0 as specified in the paper
    rho = 7.0  
    t = torch.linspace(1, 0, num_inference_steps)
    # Karras schedule with ZTSNR modifications
    sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigmas

def v_prediction_scaling_factors(sigma, sigma_data=1.0):
    """
    Compute scaling factors for v-prediction as described in paper section 2.1
    """
    # α_t = 1/√(1 + σ²) from paper
    alpha_t = 1 / torch.sqrt(1 + sigma**2)
    
    # Scaling factors from paper appendix A.1
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = -sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    
    return alpha_t, c_skip, c_out, c_in

def get_ztsnr_schedule(num_steps=28, sigma_min=0.0292, sigma_max=20000.0, rho=7.0):
    """
    Generate ZTSNR noise schedule as described in paper section 2.2
    Using practical implementation from appendix A.2
    """
    t = torch.linspace(1, 0, num_steps)
    sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigmas



def training_loss_v_prediction(model, x_0, sigma, text_embeddings, added_cond_kwargs):
    """
    Training loss using v-prediction with MinSNR weighting (sections 2.1 and 2.4)
    """
    try:
        logger.debug("\n=== Starting v-prediction training loss calculation ===")
        logger.debug("Initial input shapes and values:")
        logger.debug(f"x_0: shape={x_0.shape}, dtype={x_0.dtype}, device={x_0.device}")
        logger.debug(f"sigma: shape={sigma.shape}, dtype={sigma.dtype}, range=[{sigma.min():.6f}, {sigma.max():.6f}]")
        logger.debug(f"text_embeddings: shape={text_embeddings.shape}, dtype={text_embeddings.dtype}")
        
        # Get noise and scaling factors
        logger.debug("\nGenerating noise and computing scaling factors:")
        noise = torch.randn_like(x_0)
        logger.debug(f"noise: shape={noise.shape}, dtype={noise.dtype}, std={noise.std():.6f}")
        
        alpha_t, c_skip, c_out, c_in = v_prediction_scaling_factors(sigma)
        logger.debug(f"alpha_t: range=[{alpha_t.min():.6f}, {alpha_t.max():.6f}]")
        logger.debug(f"c_skip: range=[{c_skip.min():.6f}, {c_skip.max():.6f}]")
        logger.debug(f"c_out: range=[{c_out.min():.6f}, {c_out.max():.6f}]")
        logger.debug(f"c_in: range=[{c_in.min():.6f}, {c_in.max():.6f}]")
        
        # Compute noisy sample x_t = x_0 + σε
        logger.debug("\nComputing noisy sample:")
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1)
        logger.debug(f"x_t: shape={x_t.shape}, range=[{x_t.min():.6f}, {x_t.max():.6f}]")
        
        # Compute v-target = α_t * ε - (1 - α_t) * x_0
        logger.debug("\nComputing v-target:")
        v_target = alpha_t.view(-1, 1, 1, 1) * noise - (1 - alpha_t).view(-1, 1, 1, 1) * x_0
        logger.debug(f"v_target: shape={v_target.shape}, range=[{v_target.min():.6f}, {v_target.max():.6f}]")
        
        # Get model prediction
        logger.debug("\nGetting model prediction:")
        v_pred = model(
            x_t,
            sigma,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        logger.debug(f"v_pred: shape={v_pred.shape}, range=[{v_pred.min():.6f}, {v_pred.max():.6f}]")
        
        # MinSNR weighting
        logger.debug("\nComputing MinSNR weights:")
        snr = 1 / (sigma**2)  # SNR = 1/σ²
        gamma = 1.0  # SNR clipping value
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        logger.debug(f"SNR: range=[{snr.min():.6f}, {snr.max():.6f}]")
        logger.debug(f"min_snr_gamma: range=[{min_snr_gamma.min():.6f}, {min_snr_gamma.max():.6f}]")
        
        # Compute weighted MSE loss
        logger.debug("\nComputing final loss:")
        mse_loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = (min_snr_gamma.view(-1, 1, 1, 1) * mse_loss).mean()
        
        # Collect detailed metrics
        loss_metrics = {
            'loss/total': loss.item(),
            'loss/mse_mean': mse_loss.mean().item(),
            'loss/mse_std': mse_loss.std().item(),
            'loss/snr_mean': snr.mean().item(),
            'loss/min_snr_gamma_mean': min_snr_gamma.mean().item(),
            'model/v_pred_std': v_pred.std().item(),
            'model/v_target_std': v_target.std().item(),
            'model/alpha_t_mean': alpha_t.mean().item(),
            'noise/sigma_mean': sigma.mean().item(),
            'noise/x_t_std': x_t.std().item()
        }
        
        logger.debug("Loss metrics:")
        for key, value in loss_metrics.items():
            logger.debug(f"{key}: {value:.6f}")
        
        return loss, loss_metrics

    except Exception as e:
        logger.error("\n=== Error in v-prediction training ===")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Log state of variables at time of error
        logger.error("\nVariable state at error:")
        local_vars = locals()
        for name, value in local_vars.items():
            if isinstance(value, torch.Tensor):
                try:
                    logger.error(f"{name}: shape={value.shape}, dtype={value.dtype}, "
                               f"range=[{value.min():.6f}, {value.max():.6f}]")
                except:
                    logger.error(f"{name}: <tensor stats unavailable>")
            else:
                logger.error(f"{name}: type={type(value)}")
        raise


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
        
        # Initialize class weights with default value of 1.0
        self.class_weights = {class_name: 1.0 for class_name in self.tag_classes}
        
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

    def set_class_weight(self, class_name, weight):
        """
        Set weight for a specific class.
        
        Args:
            class_name (str): Name of the class
            weight (float): Weight value
        """
        if class_name in self.class_weights:
            self.class_weights[class_name] = weight

    def calculate_tag_weights(self, tags):
        try:
            # Convert nested lists to tuples for hashing
            if isinstance(tags, list):
                # Handle nested lists by converting inner lists to tuples
                tags = tuple(tuple(t) if isinstance(t, list) else t for t in tags)
            
            weights = []
            for class_name in self.tag_classes:
                try:
                    class_tags = set(self.tag_classes[class_name])
                    tag_set = set(tags) if isinstance(tags, (list, tuple)) else {tags}
                    
                    class_intersection = class_tags.intersection(tag_set)
                    weight = self.class_weights.get(class_name, 1.0)
                    weights.append(weight if class_intersection else 1.0)
                    
                except Exception as class_error:
                    logger.error(f"Error processing class {class_name}: {str(class_error)}")
                    logger.error(f"Class traceback: {traceback.format_exc()}")
                    weights.append(1.0)
            
            # Calculate final weight and clamp between min and max
            final_weight = torch.tensor(weights).mean()
            final_weight = torch.clamp(final_weight, self.min_weight, self.max_weight)
            
            return final_weight
            
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
            
            # Convert VAE to bfloat16 and enable memory efficient attention
            self.vae = self.vae.to(dtype=torch.bfloat16)
            self.vae.enable_xformers_memory_efficient_attention()
            
            # Enable gradient checkpointing
            self.vae.enable_gradient_checkpointing()
            
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
            latents = latents.to(dtype=torch.bfloat16)  # Ensure bfloat16
            if self.latent_means is None:
                self.latent_means = torch.zeros(latents.size(1), device=latents.device, dtype=torch.bfloat16)
                self.latent_m2 = torch.zeros(latents.size(1), device=latents.device, dtype=torch.bfloat16)
                
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
            
    def training_step(self, latents=None, original_images=None):
        try:
            self.optimizer.zero_grad()
            
            # Process in smaller chunks if batch size is large
            if original_images is None:
                if isinstance(latents, dict):
                    original_images = latents["pixel_values"].to(self.vae.device, dtype=torch.bfloat16)
                else:
                    raise ValueError("Either original_images or a batch dict must be provided")
            else:
                original_images = original_images.to(self.vae.device, dtype=torch.bfloat16)
            
            # Ensure proper shape [B, C, H, W]
            if original_images.dim() == 3:
                original_images = original_images.unsqueeze(0)  # Add batch dimension
            elif original_images.dim() > 4:
                original_images = original_images.squeeze(1)  # Remove extra dimensions
                
            # Validate shape
            if original_images.dim() != 4:
                raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {original_images.shape}")
            
            # Split batch into chunks if needed
            batch_size = original_images.shape[0]
            chunk_size = min(batch_size, 2)  # Process max 2 images at once
            
            total_loss = 0
            for i in range(0, batch_size, chunk_size):
                chunk = original_images[i:i+chunk_size]
                
                # Clear cache before processing chunk
                torch.cuda.empty_cache()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Updated autocast syntax
                    # Encode
                    latents_chunk = self.vae.encode(chunk).latent_dist.sample()
                    
                    # Update statistics for this chunk
                    self.update_statistics(latents_chunk.detach())
                    
                    # Get current statistics
                    means, stds = self.get_statistics()
                    if means is not None and stds is not None:
                        latents_chunk = (latents_chunk - means[None,:,None,None]) / stds[None,:,None,None]
                        decode_latents = latents_chunk * stds[None,:,None,None] + means[None,:,None,None]
                    else:
                        decode_latents = latents_chunk
                    
                    # Decode
                    decoded = self.vae.decode(decode_latents).sample
                
                # Calculate loss for this chunk
                chunk_loss = F.mse_loss(decoded, chunk, reduction="mean")
                chunk_loss = chunk_loss / (batch_size / chunk_size)  # Scale loss by number of chunks
                chunk_loss.backward()
                
                total_loss += chunk_loss.item() * (batch_size / chunk_size)
            
            self.optimizer.step()
            
            return {
                "total_loss": total_loss,
                "latent_means": self.latent_means.detach().cpu() if self.latent_means is not None else None,
                "latent_stds": stds.detach().cpu() if stds is not None else None
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            logger.error(f"Training traceback: {traceback.format_exc()}")
            logger.error(f"Device info - VAE: {self.vae.device}")
            return {"total_loss": float('inf')}  # Return infinite loss on error
            
    def save_pretrained(self, path):
        """vae diffusers compatible save"""
        try:
            os.makedirs(path, exist_ok=True)
            self.vae.save_pretrained(path, safe_serialization=True)
            
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

        # Now we can watch the model after it's initialized
        if args.use_wandb:
            wandb.watch(unet, log='all', log_freq=100)
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
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
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
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            
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

                    loss, loss_metrics = training_loss_v_prediction(
                        unet,
                        latents,
                        sigma,
                        text_embeddings,
                        {
                            "text_embeds": pooled_text_embeddings_2,
                            "time_ids": torch.tensor([
                                1024,  # Original height
                                1024,  # Original width
                                1024,  # Target height
                                1024,  # Target width
                                0,    # Crop top
                                0,    # Crop left
                            ], device=device, dtype=torch.bfloat16)
                        }
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps

                # Apply tag-based weighting
                weighted_loss = tag_weighter.update_training_loss(loss, batch["tags"])
                weighted_loss.backward()

                # Update metrics
                epoch_loss += weighted_loss.item()
                for metric_name, metric_value in loss_metrics.items():
                    epoch_metrics[metric_name] += metric_value

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

            # Log epoch-level metrics
            if args.use_wandb:
                wandb.log({
                    'epoch/average_loss': epoch_loss / len(train_dataloader),
                    'epoch/current_epoch': epoch + 1,
                    **{f'epoch/{k}': v / len(train_dataloader) for k, v in epoch_metrics.items()}
                })

        # Create output directory if it doesn't exist
         
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving diffusers format in full model to {output_dir}")
        # Save each component in its proper subdirectory
        unet.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=True)
        vae.save_pretrained(os.path.join(output_dir, "vae"), safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        tokenizer_2.save_pretrained(os.path.join(output_dir, "tokenizer_2"))
        text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"), safe_serialization=True)
        text_encoder_2.save_pretrained(os.path.join(output_dir, "text_encoder_2"), safe_serialization=True)



        print(f"Models saved successfully to {output_dir}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
    finally:
        if args.use_wandb:
            wandb.finish()

def save_image_grid(images, path, nrow=1, normalize=True):
    """Save a list of images as a grid"""
    grid = make_grid(images, nrow=nrow, normalize=normalize)
    TF.to_pil_image(grid).save(path)

class ModelValidator:
    def __init__(self, model, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device="cuda"):
        """
        Initialize the ModelValidator with SDXL models and tokenizers.
        
        Args:
            model: The UNet model
            vae: The VAE model
            tokenizer: First SDXL tokenizer
            tokenizer_2: Second SDXL tokenizer
            text_encoder: First SDXL text encoder
            text_encoder_2: Second SDXL text encoder
            device: Device to run on (default: "cuda")
        """
        self.model = model.to(device)
        self.vae = vae.to(device)
        # Load default SDXL VAE for decoding validation images
        self.default_vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="vae",
            torch_dtype=torch.bfloat16  # Match model dtype
        ).to(device)
        self.default_vae.eval()  # Ensure VAE is in eval mode
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder.to(device)
        self.text_encoder_2 = text_encoder_2.to(device)
        self.device = device

    def generate_at_sigma(self, prompt, target_sigma, sigma_max=20000.0):
        """Generate a sample denoising from sigma_max down to target_sigma"""
        try:
            # Create custom sigma schedule from sigma_max down to target_sigma
            sigmas = torch.linspace(sigma_max, target_sigma, steps=10).to(self.device, dtype=torch.bfloat16)
            
            # Initialize random latents
            latents = torch.randn((1, 4, 64, 64), dtype=torch.bfloat16).to(self.device)
            
            # Encode text prompt for both encoders
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # SDXL text encoder 2 input
            text_input_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                # Process with first text encoder
                text_input_ids = text_input.input_ids.to(self.device)
                prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=torch.bfloat16)
                
                # Process with second text encoder
                text_input_ids_2 = text_input_2.input_ids.to(self.device)
                prompt_embeds_2 = self.text_encoder_2(text_input_ids_2)
                pooled_prompt_embeds = prompt_embeds_2[0].to(dtype=torch.bfloat16)
                text_embeds = prompt_embeds_2.pooler_output.to(dtype=torch.bfloat16)
            
            # Create micro-conditioning tensors for 1024x1024 output
            time_ids = torch.tensor([
                1024,  # Original height
                1024,  # Original width
                1024,  # Target height
                1024,  # Target width
                0,    # Crop top
                0,    # Crop left
            ], device=self.device, dtype=torch.bfloat16)
            
            time_ids = time_ids.unsqueeze(0)  # Add batch dimension
            
            # Prepare added conditioning kwargs
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids
            }
            
            # Combine text embeddings
            prompt_embeds = torch.cat([prompt_embeds, pooled_prompt_embeds], dim=-1)
            
            # Denoise
            for i, sigma in enumerate(sigmas):
                with torch.no_grad():
                    # Ensure sigma is bfloat16
                    sigma = sigma.to(dtype=torch.bfloat16)
                    noise_pred = self.model(
                        latents,
                        sigma[None].to(self.device),
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                
                latents = latents - noise_pred * (sigma[None, None, None] ** 2)
                
                if i < len(sigmas) - 1:
                    noise = torch.randn_like(latents)
                    latents = latents + noise * (sigmas[i + 1] ** 2 - sigmas[i] ** 2) ** 0.5
            
            # Decode latents using default VAE
            with torch.no_grad():
                image = self.default_vae.decode(latents / 0.18215).sample
                
            return image
            
        except Exception as e:
            logger.error(f"Sample generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return torch.zeros((1, 3, 1024, 1024), dtype=torch.bfloat16).to(self.device)
        

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
        
        def prepare_image(img):
            """Convert image tensor to proper format for saving"""
            try:
                # If the input is latents, decode with default VAE
                if img.shape[1] == 4:  # Latent space has 4 channels
                    with torch.no_grad():
                        img = self.default_vae.decode(img / 0.18215).sample
                
                # Convert to float32 if needed
                if img.dtype == torch.bfloat16:
                    img = img.to(torch.float32)
                
                # Remove batch dimension if present
                if img.dim() == 4:
                    img = img.squeeze(0)
                
                # Ensure we have a valid image tensor [C, H, W]
                assert img.dim() == 3, f"Expected 3D tensor after processing, got shape {img.shape}"
                return img
                
            except Exception as e:
                logger.error(f"Error preparing image: {str(e)}")
                logger.error(f"Input tensor shape: {img.shape}")
                logger.error(f"Input tensor dtype: {img.dtype}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        try:
            # Save ZTSNR comparison (Figure 2)
            if 'ztsnr' in results:
                ztsnr_results, _ = results['ztsnr']  # Unpack tuple
                images = [
                    prepare_image(ztsnr_results['ztsnr']),
                    prepare_image(ztsnr_results['no_ztsnr'])
                ]
                save_image_grid(
                    images,
                    os.path.join(output_dir, 'ztsnr_comparison.png'),
                    nrow=2,
                    normalize=True
                )
            
            # Save high-res coherence comparison (Figure 6)
            if 'coherence' in results:
                coherence_steps = []
                for steps in results['coherence'].values():
                    coherence_steps.extend([prepare_image(step) for step in steps])
                save_image_grid(
                    coherence_steps,
                    os.path.join(output_dir, 'coherence_steps.png'),
                    nrow=len(steps),
                    normalize=True
                )
            
            # Save individual prompt results
            for key, image in results.items():
                if isinstance(image, torch.Tensor):
                    # Skip non-image results
                    if not (isinstance(key, str) and key in ['ztsnr', 'coherence']):
                        save_image_grid(
                            [prepare_image(image)],
                            os.path.join(output_dir, f'prompt_{key[:30]}.png'),
                            normalize=True
                        )
                    
        except Exception as e:
            logger.error(f"Failed to save validation images: {str(e)}")
            logger.error(f"Results keys: {results.keys()}")
            logger.error(f"Traceback: {traceback.format_exc()}")

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
            
            # Initialize random latents and convert to bfloat16
            latents = torch.randn((1, 4, 64, 64), dtype=torch.bfloat16).to(self.device)
            
            # Encode text prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # SDXL text encoder 2 input
            text_input_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                # Process with first text encoder
                text_input_ids = text_input.input_ids.to(self.device)
                prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=torch.bfloat16)
                
                # Process with second text encoder
                text_input_ids_2 = text_input_2.input_ids.to(self.device)
                prompt_embeds_2 = self.text_encoder_2(text_input_ids_2)
                pooled_prompt_embeds = prompt_embeds_2[0].to(dtype=torch.bfloat16)
                text_embeds = prompt_embeds_2.pooler_output.to(dtype=torch.bfloat16)
            
            # Create micro-conditioning tensors for 1024x1024 output
            time_ids = torch.tensor([
                1024,  # Original height
                1024,  # Original width
                1024,  # Target height
                1024,  # Target width
                0,    # Crop top
                0,    # Crop left
            ], device=self.device, dtype=torch.bfloat16)
            
            time_ids = time_ids.unsqueeze(0)  # Add batch dimension
            
            # Prepare added conditioning kwargs
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids
            }
            
            # Combine text embeddings
            prompt_embeds = torch.cat([prompt_embeds, pooled_prompt_embeds], dim=-1)
            
            # Denoise
            for i, sigma in enumerate(sigmas):
                with torch.no_grad():
                    # Ensure sigma is bfloat16
                    sigma = sigma.to(dtype=torch.bfloat16)
                    noise_pred = self.model(
                        latents,
                        sigma[None].to(self.device),
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                
                latents = latents - noise_pred * (sigma[None, None, None] ** 2)
                
                if i < len(sigmas) - 1:
                    noise = torch.randn_like(latents)
                    latents = latents + noise * (sigmas[i + 1] ** 2 - sigmas[i] ** 2) ** 0.5
            
            # Decode latents using default VAE
            with torch.no_grad():
                image = self.default_vae.decode(latents / 0.18215).sample
                
            return image
            
        except Exception as e:
            logger.error(f"Sample generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return torch.zeros((1, 3, 1024, 1024), dtype=torch.bfloat16).to(self.device)


        

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
