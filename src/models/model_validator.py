import torch
from diffusers import StableDiffusionXLPipeline
import logging
from pathlib import Path
import time
from PIL import Image
import wandb
import numpy as np

logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model_path, device="cuda", dtype=torch.float16, zsnr=True):
        """
        Initialize the model validator
        
        Args:
            model_path (str): Path to the base model
            device (str): Device to use for inference
            dtype (torch.dtype): Data type for model
            zsnr (bool): Whether to use ZSNR noise scheduling
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.v_prediction = True  # Always use v-prediction
        self.zsnr = zsnr
        
        # Initialize sigma parameters
        self.sigma_min = 0.0292
        self.sigma_data = 1.0
        
        logger.info(f"Initializing validator with model from {model_path}")
        
        # Load pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Create new scheduler config
        scheduler_config = dict(self.pipeline.scheduler.config)
        scheduler_config.update({
            "prediction_type": "v_prediction" if self.v_prediction else "epsilon",
            "sigma_min": self.sigma_min,
            "sigma_data": self.sigma_data,
        })
        
        # Create new scheduler with updated config
        self.pipeline.scheduler = type(self.pipeline.scheduler).from_config(scheduler_config)
        
        # Enable memory optimization
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
        
    def karras_rescale(self, cond, uncond, guidance_scale):
        """Karras CFG rescale method"""
        std_cond = torch.std(cond, dim=tuple(range(1, cond.ndim)))
        std_uncond = torch.std(uncond, dim=tuple(range(1, uncond.ndim)))
        std_cond = std_cond.where(std_cond != 0, torch.ones_like(std_cond))
        std_uncond = std_uncond.where(std_uncond != 0, torch.ones_like(std_uncond))
        
        # Calculate rescale multiplier
        k = self.rescale_multiplier
        alpha = k * std_uncond / std_cond
        
        # Rescale conditional
        rescaled_cond = cond * alpha.view(-1, *(1,) * (cond.ndim - 1))
        
        # Apply guidance
        return uncond + guidance_scale * (rescaled_cond - uncond)

    def generate_images(self, prompts, num_images_per_prompt=1, negative_prompt=None, **kwargs):
        """Generate images with custom ZTSNR and CFG rescale settings"""
        logger.info(f"Generating {len(prompts) * num_images_per_prompt} validation images")
        
        default_params = {
            "num_inference_steps": 28,
            "guidance_scale": 5.0,
            "height": 1024,
            "width": 1024,
        }
        default_params.update(kwargs)
        
        generated_images = []
        generation_times = []
        
        try:
            for prompt in prompts:
                start_time = time.time()
                
                # Override pipeline's guidance function if using CFG rescale
                if self.rescale_cfg:
                    original_guidance = self.pipeline._guidance_scale_embed
                    self.pipeline._guidance_scale_embed = lambda x, y, z: self.karras_rescale(x, y, z)
                
                # Generate images
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    **default_params
                )
                
                # Restore original guidance function
                if self.rescale_cfg:
                    self.pipeline._guidance_scale_embed = original_guidance
                
                gen_time = time.time() - start_time
                generation_times.append(gen_time)
                
                for idx, image in enumerate(output.images):
                    generated_images.append({
                        "prompt": prompt,
                        "image": image,
                        "generation_time": gen_time,
                        "image_index": idx
                    })
                
                logger.debug(f"Generated {num_images_per_prompt} images for prompt: {prompt}")
                logger.debug(f"Generation time: {gen_time:.2f}s")
                
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
            
        return generated_images

    def run_validation(self, prompts, output_dir=None, log_to_wandb=True, **generation_kwargs):
        """
        Run full validation suite
        
        Args:
            prompts (list): List of prompts to validate with
            output_dir (str): Directory to save validation images
            log_to_wandb (bool): Whether to log results to W&B
            **generation_kwargs: Additional arguments for image generation
        
        Returns:
            dict: Validation metrics
        """
        logger.info("Starting validation run")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        validation_metrics = {
            "generation_times": [],
            "images_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0
        }
        
        try:
            # Generate validation images
            generated_images = self.generate_images(prompts, **generation_kwargs)
            
            validation_metrics["images_generated"] = len(generated_images)
            validation_metrics["successful_generations"] = len(generated_images)
            
            # Calculate average generation time
            avg_gen_time = sum(img["generation_time"] for img in generated_images) / len(generated_images)
            validation_metrics["avg_generation_time"] = avg_gen_time
            
            # Save images if output directory provided
            if output_dir:
                for idx, gen in enumerate(generated_images):
                    image_path = output_dir / f"validation_{idx}.png"
                    gen["image"].save(image_path)
                    logger.debug(f"Saved validation image to {image_path}")
            
            # Log to W&B if enabled
            if log_to_wandb and wandb.run is not None:
                wandb_images = []
                for idx, gen in enumerate(generated_images):
                    wandb_images.append(
                        wandb.Image(
                            gen["image"], 
                            caption=f"Prompt: {gen['prompt']}\nGeneration time: {gen['generation_time']:.2f}s"
                        )
                    )
                
                wandb.log({
                    "validation/images": wandb_images,
                    "validation/avg_generation_time": avg_gen_time,
                    "validation/successful_generations": validation_metrics["successful_generations"],
                    "validation/failed_generations": validation_metrics["failed_generations"]
                })
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            validation_metrics["failed_generations"] = len(prompts)
            raise
        
        finally:
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        return validation_metrics

    def validate_checkpoint(self, checkpoint_path, prompts, **kwargs):
        """
        Validate a specific checkpoint
        
        Args:
            checkpoint_path (str): Path to checkpoint to validate
            prompts (list): List of prompts to validate with
            **kwargs: Additional arguments for validation
        
        Returns:
            dict: Validation metrics
        """
        logger.info(f"Validating checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        try:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                checkpoint_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            
            # Run validation
            metrics = self.run_validation(prompts, **kwargs)
            
            # Add checkpoint info to metrics
            metrics["checkpoint_path"] = checkpoint_path
            
            return metrics
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {str(e)}")
            raise
        
        finally:
            # Clear GPU memory
            torch.cuda.empty_cache()
