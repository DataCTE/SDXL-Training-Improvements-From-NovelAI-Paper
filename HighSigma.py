import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import logging
import argparse
import xformers
from PIL import Image
from bitsandbytes.optim import AdamW8bit
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from transformers.optimization import Adafactor
import torchvision.models as models
from diffusers.models import AutoencoderKL


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


# Global bfloat16 settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_sigmas(num_inference_steps=28, sigma_max=29.0, sigma_min=0.0292):
    """
    Generate sigmas using improved schedule from paper
    """
    rho = 7  # ρ parameter from paper
    t = torch.linspace(0, 1, num_inference_steps)
    sigmas = torch.sqrt((sigma_max ** (2 * (1 - t) / rho)) * (sigma_min ** (2 * t / rho)) + 1)
    return sigmas

def training_loss(model, x_0, sigma, text_embeddings, text_embeddings_2, pooled_text_embeds_2, timesteps):
    #print(f"text_embeddings shape: {text_embeddings.shape}")
    #print(f"text_embeddings_2 shape: {text_embeddings_2.shape}")
    #print(f"pooled_text_embeds_2 shape: {pooled_text_embeds_2.shape}")
    #print(f"timesteps shape: {timesteps.shape}")
    #print(f"x_0 shape before: {x_0.shape}")
    
    # Ensure x_0 has correct shape [B, C, H, W]
    x_0 = x_0.squeeze()  # Remove any extra dimensions
    if len(x_0.shape) == 3:
        x_0 = x_0.unsqueeze(0)
    #print(f"x_0 shape after: {x_0.shape}")
    
    # Create time_ids with correct shape
    batch_size = x_0.shape[0]
    time_ids = torch.zeros((batch_size, 6), device=x_0.device)
    
    # Ensure embeddings have correct shapes
    # Remove extra dimensions from embeddings
    text_embeddings = text_embeddings.squeeze()  # [B, 77, 768]
    text_embeddings_2 = text_embeddings_2.squeeze()  # [B, 77, 1280]
    pooled_text_embeds_2 = pooled_text_embeds_2.squeeze()  # [B, 1280]
    
    # Add batch dimension if needed
    if len(text_embeddings.shape) == 2:
        text_embeddings = text_embeddings.unsqueeze(0)
    if len(text_embeddings_2.shape) == 2:
        text_embeddings_2 = text_embeddings_2.unsqueeze(0)
    if len(pooled_text_embeds_2.shape) == 1:
        pooled_text_embeds_2 = pooled_text_embeds_2.unsqueeze(0)
    
    # Create combined text embeddings
    combined_text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
    
   
    noise = torch.randn_like(x_0)
    sigma = sigma.view(-1, 1, 1, 1)
    x_t = x_0 + sigma * noise
    
    # V-prediction
    v = noise / (sigma ** 2 + 1).sqrt()
    target = v

    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds_2,
        "time_ids": time_ids
    }

    model_output = model(
        x_t,
        timesteps,
        encoder_hidden_states=combined_text_embeddings,
        added_cond_kwargs=added_cond_kwargs
    ).sample

    # ZTSNR loss calculation
    snr = 1 / (sigma ** 2)
    mse = F.mse_loss(model_output, target, reduction="none")
    
    # Zero terminal SNR
    min_snr_gamma = 0.5  # From paper
    snr_weight = (snr.squeeze() / (1 + snr.squeeze())).clamp(max=min_snr_gamma)
    loss = (mse * snr_weight.view(-1, 1, 1, 1)).mean()
    
    return loss

class Dataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, 
                 cache_dir="latents_cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

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

        self.transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Add aspect bucketing
        self.aspect_buckets = {
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
                bucket = min(self.aspect_buckets.keys(), 
                           key=lambda x: abs(x[0]/x[1] - ratio))
                self.aspect_buckets[bucket].append(img_path)

        # Add tag processing
        self.tag_list = self._build_tag_list()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # Add this line after initializing all the models and paths
        self._cache_latents_and_embeddings_optimized()  # Cache latents before using them

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
                max_length=77,  # Add this
                truncation=True  # Add this
            ).to("cuda")
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            return image_features, text_features

    def _cache_latents_and_embeddings_optimized(self):
        """Optimized version of caching that uses bfloat16"""
        self.vae.eval()
        self.vae.to("cuda")
        self.text_encoder.eval()
        self.text_encoder.to("cuda")
        self.text_encoder_2.eval()
        self.text_encoder_2.to("cuda")
        self.clip_model.eval()  
        self.clip_model.to("cuda")
          
        # Add progress bar
        print("Caching latents and embeddings...")
        for img_path, caption_path in tqdm(zip(self.image_paths, self.caption_paths), 
                                     total=len(self.image_paths),
                                     desc="Caching"):
            cache_latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
            cache_embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"
            
            if not cache_latents_path.exists() or not cache_embeddings_path.exists():
                print(f"Caching latents and embeddings for {img_path.name}")
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    # Get tags from caption
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                        tags = [t.strip() for t in caption.split(',')]

                    # Get CLIP embeddings
                    clip_image_embed, clip_tag_embeds = self._get_clip_embeddings(image, tags)
                    
                    # Regular processing
                    image_tensor = image_tensor.to("cuda", dtype=torch.bfloat16)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    # Process text embeddings
                    text_input = self.tokenizer(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    text_embeddings = self.text_encoder(text_input.input_ids)[0]
                    
                    # SDXL text encoder 2
                    text_input_2 = self.tokenizer_2(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    text_encoder_output_2 = self.text_encoder_2(text_input_2.input_ids)
                    text_embeddings_2 = text_encoder_output_2.last_hidden_state
                    pooled_text_embeddings_2 = text_encoder_output_2.pooler_output

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
                    
                    # Move models back to CPU
                    self.vae.to("cpu")
                    self.text_encoder.to("cpu")
                    self.text_encoder_2.to("cpu")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        latents_path = self.cache_dir / f"{self.image_paths[idx].stem}_latents.pt"
        embeddings_path = self.cache_dir / f"{self.image_paths[idx].stem}_embeddings.pt"
        
        # Load with weights_only=True
        latents = torch.load(latents_path, weights_only=True)
        embeddings = torch.load(embeddings_path, weights_only=True)
        
        # Load original image for VAE training
        original_image = Image.open(self.image_paths[idx]).convert("RGB")
        original_image_tensor = self.transform(original_image)
        
        # Ensure correct shapes for embeddings
        # We want [B, 77, 768] and [B, 77, 1280] and [B, 1280]
        text_embeddings = embeddings["text_embeddings"]
        text_embeddings_2 = embeddings["text_embeddings_2"]
        pooled_text_embeddings_2 = embeddings["pooled_text_embeddings_2"]
        
        # Add batch dimension if needed
        if len(text_embeddings.shape) == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if len(text_embeddings_2.shape) == 2:
            text_embeddings_2 = text_embeddings_2.unsqueeze(0)
        if len(pooled_text_embeddings_2.shape) == 1:
            pooled_text_embeddings_2 = pooled_text_embeddings_2.unsqueeze(0)
        
        return {
        "latents": latents,  # [B, C, H, W]
        "text_embeddings": text_embeddings,  # [B, 77, 768]
        "text_embeddings_2": text_embeddings_2,  # [B, 77, 1280]
        "pooled_text_embeddings_2": pooled_text_embeddings_2,  # [B, 1280]
        "clip_image_embed": embeddings["clip_image_embed"],
        "clip_tag_embeds": embeddings["clip_tag_embeds"],
        "tags": embeddings["tags"],
        "original_images": original_image_tensor 
        }

class PerceptualLoss:
    def __init__(self):
        # Update VGG initialization to use weights parameter
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.vgg = models.vgg16(weights=weights).features.eval().to("cuda")
        self.vgg.requires_grad_(False)
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_features(self, x):
        x = self.normalize(x)
        features = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def __call__(self, pred, target):
        pred_features = self.get_features(pred)
        target_features = self.get_features(target)
        
        loss = 0.0
        for key in pred_features:
            loss += F.mse_loss(pred_features[key], target_features[key])
        return loss


class VAEFineTuner:
    def __init__(self, vae, learning_rate=1e-6, perceptual_weight=1.0, l1_weight=1.0, kl_weight=0.1):
        self.vae = vae
        self.perceptual_loss = PerceptualLoss()
        
        # Add loss weights as instance variables
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.kl_weight = kl_weight
        
        # Use a single optimizer for both encoder and decoder
        self.optimizer = AdamW8bit(
            [
                {'params': self.vae.decoder.parameters(), 'lr': learning_rate},
                {'params': self.vae.encoder.parameters(), 'lr': learning_rate * 0.5}
            ],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )

    def training_step(self, latents, original_images):
        # Convert inputs to bfloat16 to match model parameters
        latents = latents.to(dtype=torch.bfloat16)
        original_images = original_images.to(dtype=torch.bfloat16)
        
        # Ensure inputs are on correct device and shape
        original_images = original_images.to(latents.device, memory_format=torch.channels_last)
        latents = latents.squeeze().unsqueeze(0) if len(latents.squeeze().shape) == 3 else latents
        
        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)
        
        # Decode latents
        decoded_images = self.vae.decode(latents).sample
        
        # Convert to float32 for loss computation
        decoded_images = decoded_images.float()
        original_images = original_images.float()
        
        # Calculate losses
        p_loss = self.perceptual_loss(decoded_images, original_images)
        l1_loss = F.l1_loss(decoded_images, original_images)
        
        # Optional KL loss
        kl_loss = torch.tensor(0.0, device=latents.device)
        if self.vae.training:
            posterior = self.vae.encode(original_images.bfloat16()).latent_dist
            kl_loss = torch.mean(-0.5 * torch.sum(1 + posterior.variance.log() - 
                               posterior.mean.pow(2) - posterior.variance, dim=1))
        
        # Combine losses
        total_loss = (self.perceptual_weight * p_loss + 
                     self.l1_weight * l1_loss + 
                     self.kl_weight * kl_loss)

        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Clear memory
        del decoded_images, original_images
        torch.cuda.empty_cache()
        
        return {
            'total_loss': total_loss.item(),
            'perceptual_loss': p_loss.item(),
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item()
        }




def setup_distributed():
    """Setup distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank(), dist.get_world_size()

def main(args):
    logger = setup_logging()
    
    
    
    # Load models with bfloat16
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
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
    dataset = Dataset(
        args.data_dir,
        vae,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        cache_dir=args.cache_dir
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
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
            vae, 
            learning_rate=args.vae_learning_rate,
            perceptual_weight=1.0,
            l1_weight=1.0,
            kl_weight=0.1
        )

    # Training loop
    logger.info("Starting training...")
    total_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU efficiently
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):  # Change to float32
                latents = batch["latents"].to("cuda", non_blocking=True)
                text_embeddings = batch["text_embeddings"].to("cuda", non_blocking=True)
                text_embeddings_2 = batch["text_embeddings_2"].to("cuda", non_blocking=True)
                pooled_text_embeddings_2 = batch["pooled_text_embeddings_2"].to("cuda", non_blocking=True)
                
                sigma = get_sigmas(args.num_inference_steps)[step % args.num_inference_steps].to(latents.device)
                timesteps = torch.ones(latents.shape[0], device=latents.device).long() * (step % args.num_inference_steps)

                # Forward pass
                loss = training_loss(
                    unet, latents, sigma, text_embeddings, 
                    text_embeddings_2, pooled_text_embeddings_2, timesteps
                ) / args.gradient_accumulation_steps

                # Backward pass
                loss.backward()

            # Optimization step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema_model.update_parameters(unet)

                # VAE finetuning
                if args.finetune_vae and step % args.vae_train_freq == 0:
                    vae_losses = vae_finetuner.training_step(
                        batch["latents"].to("cuda"), 
                        batch["original_images"].to("cuda")
                    )

                # Clear GPU memory
                del latents, text_embeddings, text_embeddings_2, pooled_text_embeddings_2
                torch.cuda.empty_cache()

            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix({
                "epoch": f"{epoch+1}/{args.num_epochs}",
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                **({"vae_loss": f"{vae_losses['total_loss']:.4f}"} if args.finetune_vae and step % args.vae_train_freq == 0 else {})
            })

        # Save checkpoint
        if args.save_checkpoints:
            checkpoint_path = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
            checkpoint_path.mkdir(exist_ok=True, parents=True)
            unet.save_pretrained(checkpoint_path / "unet")

    # Save models
    logger.info("Saving models...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save in Diffusers format
    unet.save_pretrained(output_dir / "unet")
    if args.finetune_vae:
        vae.save_pretrained(output_dir / "vae")
    text_encoder.save_pretrained(output_dir / "text_encoder")
    text_encoder_2.save_pretrained(output_dir / "text_encoder_2")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tokenizer_2.save_pretrained(output_dir / "tokenizer_2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--finetune_vae", action="store_true", help="Enable VAE finetuning")
    parser.add_argument("--vae_learning_rate", type=float, default=1e-6, help="VAE learning rate")
    parser.add_argument("--vae_train_freq", type=int, default=10, help="VAE training frequency")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--use_adafactor", action="store_true", help="Use Adafactor instead of AdamW8bit")

    args = parser.parse_args()
    main(args)
