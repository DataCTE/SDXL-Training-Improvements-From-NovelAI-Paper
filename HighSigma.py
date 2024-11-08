import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
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


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_sigmas(num_inference_steps=28, sigma_max=29.0, sigma_min=0.0292):
    """
    Generate sigmas using cosine schedule
    """
    t = torch.linspace(0, 1, num_inference_steps)
    f_t = torch.cos((t + 0.008) / 1.008 * math.pi / 2)
    sigmas = sigma_max * (f_t / f_t[0]) * (sigma_min / sigma_max) ** t
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

    target = noise
    snr = (sigma ** -2.0).squeeze()
    min_snr = 1.0
    snr_clipped = snr.clamp(max=min_snr)
    weight = (snr_clipped / snr).view(-1, 1, 1, 1)
    
    mse = F.mse_loss(model_output, target, reduction="none")
    loss = (mse * weight).mean()
    
    return loss

class Dataset(Dataset):
    def __init__(self, data_dir, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, cache_dir="latents_cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

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

        # Cache latents and text embeddings if they don't exist
        self.vae = vae
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self._cache_latents_and_embeddings()

    def _cache_latents_and_embeddings(self):
        self.vae.eval()
        self.vae.to("cuda")
        self.text_encoder.eval()
        self.text_encoder.to("cuda")
        self.text_encoder_2.eval()
        self.text_encoder_2.to("cuda")

        for img_path, caption_path in zip(self.image_paths, self.caption_paths):
            cache_latents_path = self.cache_dir / f"{img_path.stem}_latents.pt"
            cache_embeddings_path = self.cache_dir / f"{img_path.stem}_embeddings.pt"
            if not cache_latents_path.exists() or not cache_embeddings_path.exists():
                print(f"Caching latents and embeddings for {img_path.name}")
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    # Convert image to bf16 before processing
                    image = image.to("cuda", dtype=torch.bfloat16)
                    
                    # Compute latents
                    latents = self.vae.encode(image).latent_dist.sample()
                    latents = latents * 0.18215
                    latents = latents.cpu()

                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()

                    # Process text embeddings
                    text_input = self.tokenizer(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Convert to bf16 for text processing
                    text_input_ids = text_input.input_ids.to("cuda")
                    text_encoder_output = self.text_encoder(text_input_ids)
                    text_embeddings = text_encoder_output.last_hidden_state.cpu()

                    # Same for text encoder 2
                    text_input_2 = self.tokenizer_2(
                        caption,
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    text_input_ids_2 = text_input_2.input_ids.to("cuda")
                    text_encoder_output_2 = self.text_encoder_2(text_input_ids_2)
                    text_embeddings_2 = text_encoder_output_2.last_hidden_state.cpu()
                    pooled_text_embeddings_2 = text_encoder_output_2.pooler_output.cpu()

                    # Save embeddings
                    torch.save({
                        "text_embeddings": text_embeddings,
                        "text_embeddings_2": text_embeddings_2,
                        "pooled_text_embeddings_2": pooled_text_embeddings_2
                    }, cache_embeddings_path)

                    # Save latents
                    torch.save(latents, cache_latents_path)

        # Move models back to CPU
        self.vae.to("cpu")
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        latents_path = self.cache_dir / f"{self.image_paths[idx].stem}_latents.pt"
        embeddings_path = self.cache_dir / f"{self.image_paths[idx].stem}_embeddings.pt"
        
        # Load latents and ensure correct shape [B, C, H, W]
        latents = torch.load(latents_path)
        latents = latents.squeeze()  # Remove any extra dimensions
        if len(latents.shape) == 3:  # If [C, H, W]
            latents = latents.unsqueeze(0)  # Make it [B, C, H, W]
        
        # Load embeddings
        embeddings = torch.load(embeddings_path)
        text_embeddings = embeddings["text_embeddings"]
        text_embeddings_2 = embeddings["text_embeddings_2"]
        pooled_text_embeddings_2 = embeddings["pooled_text_embeddings_2"]
        
        # Ensure correct shapes for embeddings
        # We want [B, 77, 768] and [B, 77, 1280] and [B, 1280]
        text_embeddings = text_embeddings.squeeze()  # Remove extra dims
        text_embeddings_2 = text_embeddings_2.squeeze()
        pooled_text_embeddings_2 = pooled_text_embeddings_2.squeeze()
        
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
        }


def main(args):
    logger = setup_logging()
    
    dtype = torch.bfloat16
    
    # Load models with appropriate dtype
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
        torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=dtype
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
        torch_dtype=dtype
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder_2",
        torch_dtype=dtype
    )

    # Move UNet to GPU and enable xFormers
    unet.to("cuda")

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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = AdamW8bit(
        unet.parameters(),
        lr=args.learning_rate * args.batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

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
   

    # Training loop
    logger.info("Starting training...")
    total_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        dynamic_ncols=True
    )

    for _ in range(args.num_epochs):
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Convert batch tensors to appropriate dtype
            latents = batch["latents"].to("cuda", dtype=dtype)
            text_embeddings = batch["text_embeddings"].to("cuda", dtype=dtype)
            text_embeddings_2 = batch["text_embeddings_2"].to("cuda", dtype=dtype)
            pooled_text_embeddings_2 = batch["pooled_text_embeddings_2"].to("cuda", dtype=dtype)


            batch_size = latents.shape[0]

            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Get sigmas for this batch
                sigmas = get_sigmas(num_inference_steps=args.num_inference_steps).to(latents.device, dtype=dtype)
                sigma_indices = torch.randint(0, len(sigmas), (batch_size,), device=latents.device)
                sigma = sigmas[sigma_indices]

                # Calculate timesteps from sigma indices
                timesteps = sigma_indices.to(latents.device).float()

                # Calculate loss
                loss = training_loss(
                    unet,  # Use unet directly instead of model_engine
                    latents,
                    sigma,
                    text_embeddings,
                    text_embeddings_2,
                    pooled_text_embeddings_2,
                    timesteps
                ) / args.gradient_accumulation_steps

           # Standard backward pass
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Update EMA model
                ema_model.update_parameters(unet)

            progress_bar.update(1)
            loss_value = loss.item()
            epoch_loss += loss_value
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "epoch_loss": f"{epoch_loss/(step+1):.4f}"})
            

    # After training loop ends
    logger.info("Saving models...")
    output_dir = Path("./highsigmamax")
    output_dir.mkdir(exist_ok=True, parents=True)
    
  
    
    # Save all components in Diffusers format
    unet.save_pretrained(output_dir / "unet")
    vae.save_pretrained(output_dir / "vae")
    text_encoder.save_pretrained(output_dir / "text_encoder")
    text_encoder_2.save_pretrained(output_dir / "text_encoder_2")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tokenizer_2.save_pretrained(output_dir / "tokenizer_2")
    
    logger.info(f"Complete model saved in Diffusers format to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="latents_cache")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    args = parser.parse_args()
    main(args)
