from diffusers import AutoencoderKL
import torch
import logging
import traceback
from typing import Dict, List, Optional, Tuple
import os
import math


logger = logging.getLogger(__name__)


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
        def save_image_grid(
                images, 
                path, 
                nrow=None, 
                normalize=True, 
                value_range=(-1, 1),
                padding=2
            ):
                """
                Save a grid of images to a file
                
                Args:
                    images (List[torch.Tensor]): List of image tensors [C, H, W]
                    path (str): Output file path
                    nrow (int, optional): Number of images per row. Defaults to sqrt(len(images))
                    normalize (bool): Whether to normalize the images. Defaults to True
                    value_range (tuple): Input value range for normalization. Defaults to (-1, 1)
                    padding (int): Padding between images. Defaults to 2
                """
                try:
                    # Import here to avoid dependency if not needed
                    from torchvision.utils import make_grid, save_image
                    
                    # Convert list to tensor if needed
                    if isinstance(images, list):
                        # Ensure all images are on the same device
                        device = images[0].device
                        images = [img.to(device) for img in images]
                        images = torch.stack(images)
                    
                    # Default to square grid if nrow not specified
                    if nrow is None:
                        nrow = int(math.ceil(math.sqrt(len(images))))
                        
                    # Create grid
                    grid = make_grid(
                        images,
                        nrow=nrow,
                        padding=padding,
                        normalize=normalize,
                        value_range=value_range
                    )
                    
                    # Convert to uint8 for saving
                    if normalize:
                        grid = (grid * 255).to(torch.uint8)
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # Save image
                    save_image(grid, path)
                    logger.debug(f"Saved image grid to {path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save image grid: {str(e)}")
                    logger.error(f"Path: {path}")
                    logger.error(f"Image shapes: {[img.shape for img in images]}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
                    
        def prepare_image(img):
            """
            Convert image tensor to proper format for saving
            
            Args:
                img (torch.Tensor): Input image tensor
                
            Returns:
                torch.Tensor: Processed image tensor [C, H, W]
            """
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
                
                # Ensure values are in [-1, 1]
                if img.min() < -1 or img.max() > 1:
                    img = img.clamp(-1, 1)
                    
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


