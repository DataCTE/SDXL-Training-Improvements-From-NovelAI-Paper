import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import logging
from pathlib import Path
import time
from PIL import Image
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from training.loss import get_sigmas, v_prediction_scaling_factors
import traceback
import math

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using trained SDXL model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated images"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use float16 precision"
    )

    return parser.parse_args()

def encode_prompt(prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device):
    """Encode the prompt using both text encoders"""
    # Tokenize prompts
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Get text embeddings
    text_encoder.eval()
    text_encoder_2.eval()
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(device)
        )[0]
        text_embeddings_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device)
        )
        pooled_text_embeddings_2 = text_embeddings_2[1]
        text_embeddings_2 = text_embeddings_2[0]
    
    # Concatenate embeddings - verify dimensions
    print(f"Text embeddings 1 shape: {text_embeddings.shape}")  # Should be [batch_size, seq_len, 768]
    print(f"Text embeddings 2 shape: {text_embeddings_2.shape}")  # Should be [batch_size, seq_len, 1280]
    text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
    print(f"Combined embeddings shape: {text_embeddings.shape}")  # Should be [batch_size, seq_len, 2048]
    
    return text_embeddings, pooled_text_embeddings_2

def load_models(model_path, dtype):
    """Load all required models"""
    logger.info("Loading models...")
    
    # Load UNet with proper config
    unet = UNet2DConditionModel.from_pretrained(
        f"{model_path}/unet",
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")
    
    # Print UNet config for debugging
    print("UNet config:", unet.config)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        f"{model_path}/vae",
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")
    vae.eval()
    
    # Load text encoders and tokenizers
    text_encoder = CLIPTextModel.from_pretrained(
        f"{model_path}/text_encoder",
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")
    
    text_encoder_2 = CLIPTextModel.from_pretrained(
        f"{model_path}/text_encoder_2",
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")
    
    tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer_2")
    
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2
    }

def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    
    # Reshape for broadcasting
    args = timesteps[:, None].float() * freqs[None, :]
    
    # Create embeddings
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    # Handle odd embedding dimensions
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding

def create_time_ids(args, device, dtype):
    # Create tensor with correct shape [batch_size, 6]
    add_time_ids = torch.zeros((1, 6), device=device, dtype=dtype)
    
    # Fill in the values according to SDXL's requirements
    add_time_ids[0] = torch.tensor([
        args.width,   # Original width
        args.height,  # Original height
        args.width,   # Target width
        args.height,  # Target height
        0,           # Crop coordinates
        0,           # Crop coordinates
    ], device=device, dtype=dtype)
    
    # Handle classifier-free guidance
    if args.guidance_scale > 1.0:
        add_time_ids = add_time_ids.repeat(2, 1)  # Shape: [2, 6]
    
    return add_time_ids

def generate_images(models, args):
    device = "cuda"
    dtype = torch.float16 if args.use_fp16 else torch.float32
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Get sigmas for sampling
    sigmas = get_sigmas(args.num_inference_steps)
    sigmas = sigmas.to(device=device, dtype=dtype)
    sigmas = sigmas.flip(0)
    
    # Encode prompts
    text_embeddings, pooled_text_embeddings_2 = encode_prompt(
        args.prompt,
        models["tokenizer"],
        models["tokenizer_2"],
        models["text_encoder"],
        models["text_encoder_2"],
        device
    )
    
    # Handle classifier-free guidance
    if args.guidance_scale > 1.0:
        uncond_embeddings, uncond_pooled = encode_prompt(
            args.negative_prompt,
            models["tokenizer"],
            models["tokenizer_2"],
            models["text_encoder"],
            models["text_encoder_2"],
            device
        )
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        pooled_text_embeddings_2 = torch.cat([uncond_pooled, pooled_text_embeddings_2])
    
    # Create time embeddings
    time_ids = create_time_ids(args, device, dtype)
    
    # Generate multiple images
    start_time = time.time()
    for img_idx in range(args.num_images):
        # Initialize latents
        latents = torch.randn(
            (1, 4, args.height // 8, args.width // 8),
            device=device,
            dtype=dtype
        )
        
        if args.guidance_scale > 1.0:
            latents = latents.repeat(2, 1, 1, 1)
        
        # Denoising loop
        for step_idx, sigma in enumerate(sigmas):
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_out = -sigma / (sigma ** 2 + 1).sqrt()
            
            latent_model_input = c_in * latents
            
            with torch.no_grad():
                noise_pred = models["unet"](
                    latent_model_input,
                    sigma,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs={
                        "text_embeds": pooled_text_embeddings_2,
                        "time_ids": time_ids
                    }
                ).sample
            
            if args.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Take only the text conditional latents, not the unconditional ones
                latents = latents[:1]
            
            latents = c_out * latents - c_in * noise_pred
            
            if step_idx % 5 == 0:
                print(f"Step {step_idx}")
                print(f"  Sigma: {sigma:.3f}")
                print(f"  Latents range: {latents.min():.3f} to {latents.max():.3f}")
                print(f"  Model input range: {latent_model_input.min():.3f} to {latent_model_input.max():.3f}")
                print(f"  Noise pred range: {noise_pred.min():.3f} to {noise_pred.max():.3f}")
        
        # Decode the final latents
        with torch.no_grad():
            latents = 1 / 0.13025 * latents
            decoded = models["vae"].decode(latents).sample
            images = (decoded / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
        
        # Save the images
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for j, image in enumerate(pil_images):
            image_path = output_dir / f"generation_{timestamp}_{img_idx}_{j}.png"
            image.save(image_path)
            logger.info(f"Saved image to {image_path}")
        
        generation_time = time.time() - start_time
        logger.info(f"Generated image {img_idx+1}/{args.num_images} in {generation_time:.2f}s")

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load models
        dtype = torch.float16 if args.use_fp16 else torch.float32
        models = load_models(args.model_path, dtype)
        
        # Generate images
        generate_images(models, args)
        
        logger.info("Generation complete!")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()