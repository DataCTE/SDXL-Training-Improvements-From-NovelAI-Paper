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
    def __init__(self, model_path, device="cuda", dtype=torch.float16,
                 zsnr=True, sigma_min=0.0292, sigma_data=1.0, min_snr_gamma=5.0,
                 variant="fp16"):
        """
        Initialize the ModelValidator with specified parameters.

        Args:
            model_path (str): Path to the pre-trained model.
            device (str): Device to run on, e.g., "cuda" or "cpu".
            dtype (torch.dtype): Data type for the model.
            zsnr (bool): Whether to use ZSNR.
            sigma_min (float): Minimum value for sigma.
            sigma_data (float): Maximum value for sigma.
            min_snr_gamma (float): Minimum SNR gamma value.
            variant (str): Variant to load, e.g., "fp16".
        """
        logger.info(f"Initializing ModelValidator with model path: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.zsnr = zsnr
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.min_snr_gamma = min_snr_gamma
        self.variant = variant
        
        try:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant=self.variant
            ).to(self.device)
            
            logger.info("Model successfully loaded to device.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_images(self, prompts, **generation_kwargs):
        """
        Generate images based on the provided prompts.

        Args:
            prompts (list): List of prompts to validate with.
            **generation_kwargs: Additional arguments for image generation.

        Returns:
            list: Generated images.
        """
        logger.info("Starting image generation.")
        
        generated_images = []
        generation_times = []
        
        try:
            for idx, prompt in enumerate(prompts):
                logger.debug(f"Generating image for prompt: {prompt}")
                
                start_time = time.time()
                
                # Generate image
                output = self.pipeline(prompt=prompt, **generation_kwargs)
                gen_time = time.time() - start_time
                
                generation_times.append(gen_time)
                
                logger.debug(f"Generated image for prompt: {prompt} in {gen_time:.2f}s")
                
                generated_image = {
                    "prompt": prompt,
                    "image": output.images[0],
                    "generation_time": gen_time,
                    "image_index": idx
                }
                
                generated_images.append(generated_image)
                
            logger.info("All images successfully generated.")
            
        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
        
        finally:
            torch.cuda.empty_cache()
        
        return generated_images, generation_times
    
    def run_validation(self, prompts, output_dir=None, log_to_wandb=True, **generation_kwargs):
        """
        Run full validation suite.

        Args:
            prompts (list): List of prompts to validate with.
            output_dir (str): Directory to save validation images.
            log_to_wandb (bool): Whether to log results to W&B.
            **generation_kwargs: Additional arguments for image generation.

        Returns:
            dict: Validation metrics.
        """
        logger.info("Starting validation run.")
        
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
            generated_images, generation_times = self.generate_images(prompts, **generation_kwargs)
            
            validation_metrics["images_generated"] = len(generated_images)
            validation_metrics["successful_generations"] = len(generated_images)
            
            # Calculate average generation time
            avg_gen_time = sum(gen_time for gen_time in generation_times) / len(generation_times)
            validation_metrics["avg_generation_time"] = avg_gen_time
            
            logger.info(f"Average generation time: {avg_gen_time:.2f}s")
            
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
                    "validation/avg_generation_time": avg_gen_time
                })
                
            logger.info("Validation metrics successfully logged to W&B.")
            
        except Exception as e:
            logger.error(f"Error during validation run: {str(e)}")
            raise
        
        finally:
            torch.cuda.empty_cache()
        
        return validation_metrics
    
    def generate_images(self, prompts, **generation_kwargs):
        """
        Generate images based on the provided prompts.

        Args:
            prompts (list): List of prompts to validate with.
            **generation_kwargs: Additional arguments for image generation.

        Returns:
            list: Generated images and their generation times.
        """
        logger.info("Starting image generation.")
        
        generated_images = []
        generation_times = []
        
        try:
            # Generate images
            outputs = self.pipeline(prompts=prompts, **generation_kwargs)
            
            for output in outputs:
                gen_time = time.time()
                
                generated_image = {
                    "prompt": output.prompt,
                    "image": output.images[0],
                    "generation_time": gen_time
                }
                
                generated_images.append(generated_image)
                generation_times.append(gen_time)
            
        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
        
        finally:
            torch.cuda.empty_cache()
        
        return generated_images, generation_times
    
    def log_validation_metrics(self, validation_metrics):
        """
        Log the validation metrics.

        Args:
            validation_metrics (dict): Validation metrics to log.
        """
        logger.info("Logging validation metrics.")
        
        try:
            if wandb.run is not None:
                wandb.log(validation_metrics)
            
            logger.info(f"Validation metrics successfully logged to W&B.")
            
        except Exception as e:
            logger.error(f"Error logging validation metrics: {str(e)}")
            raise
        
    def cleanup(self):
        """
        Clean up resources.
        """
        torch.cuda.empty_cache()
        logger.info("Resources cleaned up.")
