import logging
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import wandb
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class Validator:
    """Handles validation during training"""
    def __init__(
        self,
        pipeline,
        device: torch.device,
        dtype: torch.dtype,
        output_dir: Optional[str] = None
    ):
        """
        Initialize validator
        
        Args:
            pipeline: Stable Diffusion pipeline
            device: Torch device
            dtype: Torch dtype
            output_dir: Directory to save validation images
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Ensure pipeline is in eval mode
        self.pipeline.unet.eval()
        if hasattr(self.pipeline, 'text_encoder'):
            self.pipeline.text_encoder.eval()
        if hasattr(self.pipeline, 'text_encoder_2'):
            self.pipeline.text_encoder_2.eval()
            
    def run_validation(
        self,
        prompts: List[str],
        output_dir: Optional[str] = None,
        log_to_wandb: bool = False,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        height: int = 1024,
        width: int = 1024
    ) -> Dict[str, float]:
        """
        Run validation by generating images from prompts
        
        Args:
            prompts: List of prompts to generate images from
            output_dir: Directory to save validation images
            log_to_wandb: Whether to log images to W&B
            num_images_per_prompt: Number of images to generate per prompt
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            height: Image height
            width: Image width
            
        Returns:
            Dictionary containing validation metrics
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        metrics = {}
        validation_images = []
        
        try:
            # Generate images for each prompt
            for idx, prompt in enumerate(prompts):
                logger.info(f"\nGenerating validation image {idx+1}/{len(prompts)}")
                logger.info(f"Prompt: {prompt}")
                
                with torch.no_grad():
                    images = self.pipeline(
                        prompt=prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width
                    ).images
                
                # Save images
                for img_idx, image in enumerate(images):
                    if output_dir:
                        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_'))
                        image_path = output_dir / f"validation_{idx}_{img_idx}_{safe_prompt}.png"
                        image.save(image_path)
                        
                    if log_to_wandb:
                        validation_images.append(wandb.Image(
                            image,
                            caption=f"Prompt: {prompt}"
                        ))
            
            # Log to W&B
            if log_to_wandb and validation_images:
                wandb.log({
                    "validation/images": validation_images,
                    "validation/num_images": len(validation_images)
                })
                
            metrics["num_images"] = len(validation_images)
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {"error": str(e)}
            
    def validate_image_dimensions(self, width: int, height: int) -> tuple[bool, Optional[tuple[int, int]]]:
        """Basic image dimension validation"""
        try:
            if width <= 0 or height <= 0:
                return False, None
            if width > 2048 or height > 2048:
                return False, None
            return True, (width, height)
        except Exception as e:
            return False, None
            
    def validate_dataset(self, data_dir: str) -> tuple[bool, Any]:
        """Basic dataset validation - only checks if directory exists and contains images"""
        try:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                return False, "Dataset directory does not exist"
                
            image_files = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))
            if not image_files:
                return False, "No image files found in dataset directory"
                
            return True, {"num_images": len(image_files)}
            
        except Exception as e:
            return False, str(e)

def validate_dataset(data_dir):
    """Basic dataset validation - only checks if directory exists and contains images"""
    try:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return False, "Dataset directory does not exist"
            
        image_files = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))
        if not image_files:
            return False, "No image files found in dataset directory"
            
        return True, {"num_images": len(image_files)}
        
    except Exception as e:
        return False, str(e)

def validate_image_dimensions(width, height):
    """Basic image dimension validation"""
    try:
        if width <= 0 or height <= 0:
            return False, None
        if width > 2048 or height > 2048:
            return False, None
        return True, (width, height)
    except Exception as e:
        return False, None