import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import logging
from pathlib import Path
import time
from PIL import Image
import numpy as np
from training.loss import get_sigmas, v_prediction_scaling_factors
import traceback

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
    
    # Concatenate embeddings
    text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
    
    return text_embeddings, pooled_text_embeddings_2

def load_models(model_path, dtype):
    """Load all required models"""
    logger.info("Loading models...")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        f"{model_path}/unet",
        torch_dtype=dtype,
        use_safetensors=True
    ).to("cuda")
    unet.eval()
    
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

def generate_images(models, args):
    """Generate images using v-prediction"""
    device = "cuda"
    dtype = torch.float16 if args.use_fp16 else torch.float32
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Get sigmas for inference
    sigmas = get_sigmas(
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width
    ).to(device)
    
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
    time_ids = torch.tensor(
        [1024, 1024, 1024, 1024, 0, 0],
        device=device,
        dtype=dtype
    ).repeat(args.num_images * (2 if args.guidance_scale > 1.0 else 1), 1)
    
    # Prepare for generation
    latents_shape = (args.num_images, 4, args.height // 8, args.width // 8)
    
    for i in range(args.num_images):
        start_time = time.time()
        
        # Initialize random latents
        latents = torch.randn(latents_shape, device=device, dtype=dtype)
        
        # Denoise latents
        for _, sigma in enumerate(sigmas):
            # Get scaling factors
            c_skip, c_out, c_in = v_prediction_scaling_factors(sigma)
            
            # Prepare network input
            latent_input = c_in * latents
            
            # Get model prediction
            with torch.no_grad():
                noise_pred = models["unet"](
                    latent_input,
                    sigma,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs={
                        "text_embeds": pooled_text_embeddings_2,
                        "time_ids": time_ids
                    }
                ).sample
            
            # Handle classifier-free guidance
            if args.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = c_skip * latents + c_out * noise_pred
        
        # Decode latents using VAE
        with torch.no_grad():
            # Scale latents according to VAE scaling factor
            latents = 1 / 0.18215 * latents
            
            # Decode latents to image tensors
            images = models["vae"].decode(latents).sample
        
        # Convert to PIL images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        # Save images
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for j, image in enumerate(pil_images):
            image_path = output_dir / f"generation_{timestamp}_{i}_{j}.png"
            image.save(image_path)
            logger.info(f"Saved image to {image_path}")
        
        generation_time = time.time() - start_time
        logger.info(f"Generated image {i+1}/{args.num_images} in {generation_time:.2f}s")

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