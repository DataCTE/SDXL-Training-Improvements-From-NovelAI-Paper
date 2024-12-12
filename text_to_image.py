import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import os
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf
import PIL.Image
import math
from contextlib import autocast
from sgm.inference.helpers import get_batch, get_unique_embedder_keys_from_conditioner

class SDXLTextToImage:
    def __init__(
        self,
        model,  # SDXL UNet model
        tokenizer,  # CLIP tokenizer
        text_encoder,  # CLIP text encoder
        vae,  # SDXL VAE
        device: str = "cuda",
        sigma_max: float = 20000.0,  # NAI V3 high sigma_max
        sigma_min: float = 0.0292,   # NAI V3 low sigma_min
        rho: float = 7.0,
        sigma_data: float = 0.5,
        base_area: int = 1024 * 1024  # Base area for resolution-aware scaling
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.vae = vae.to(device)
        self.device = device
        
        # NAI V3 noise schedule parameters
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.sigma_data = sigma_data
        self.base_area = base_area

    def get_sigmas(self, n_steps: int) -> torch.Tensor:
        """Generate NAI V3 noise schedule with high sigma_max"""
        ramp = torch.linspace(0, 1, n_steps + 1, device=self.device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas

    @torch.no_grad()
    def encode_prompt(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Encode prompt using model's conditioner"""
        with self.model.ema_scope():
            batch_uc = {
                "txt": [""] * len(batch["txt"])  # Empty strings for unconditional
            }
            c, uc = self.model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[],
            )
            
            # Move to device
            for k in c:
                c[k], uc[k] = map(lambda y: y[k].to(self.device), (c, uc))
                
        return c, uc

    def denoise_latents(
        self,
        latents: torch.Tensor,
        cond: Dict,
        uc: Dict,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        """Denoise latents using NAI V3's improved noise schedule"""
        
        # Generate sigmas for sampling
        sigmas = self.get_sigmas(num_inference_steps)
        
        # Scale initial latents
        height, width = latents.shape[2:]
        area = height * width
        noise_scale = torch.sqrt(
            torch.tensor(area / self.base_area, device=self.device)
        )
        latents = latents * sigmas[0] * noise_scale
        
        # Sampling loop
        for i in range(num_inference_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Calculate scaling factors
            sigma_squared = sigma * sigma
            sigma_data_squared = self.sigma_data * self.sigma_data
            denominator = sigma_squared + sigma_data_squared
            denominator_sqrt = denominator.sqrt()
            
            c_skip = sigma_data_squared / denominator
            c_out = -sigma * self.sigma_data / denominator_sqrt
            c_in = 1 / denominator_sqrt
            
            # Scale latents for model input
            scaled_latents = latents * c_in
            
            # Get model prediction
            def denoiser(x, sigma, c):
                return self.model.denoiser(self.model.model, x, sigma, c)
            
            with autocast(self.device):
                with self.model.ema_scope():
                    v = denoiser(scaled_latents, sigma, cond)
                    if guidance_scale != 1.0:
                        v_uc = denoiser(scaled_latents, sigma, uc)
                        v = v_uc + guidance_scale * (v - v_uc)
            
            # Update latents
            denoised = latents * c_skip + v * c_out
            w = (sigma_next / sigma)**2
            noise = torch.randn_like(latents)
            latents = denoised + noise * (w * sigma_next**2).sqrt()
            
        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate images from text using NAI V3 improvements"""
        
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
            
        # Prepare batch
        batch = {"txt": prompt}
        
        # Encode text
        c, uc = self.encode_prompt(batch)
        
        # Generate initial noise
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            device=self.device
        )
        
        # Denoise
        latents = self.denoise_latents(
            latents,
            c,
            uc,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Decode latents
        with autocast(self.device):
            with self.model.ema_scope():
                images = self.model.decode_first_stage(latents)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        
        return images


# useage

text_to_image = SDXLTextToImage(model, tokenizer, text_encoder, vae)
images = text_to_image.generate("a beautiful image", height=1024, width=1024)

