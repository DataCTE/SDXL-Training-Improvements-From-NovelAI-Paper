import torch
from diffusers import StableDiffusionXLPipeline
import logging
from pathlib import Path
import time
import wandb
from training.loss import get_sigmas  # Import our custom sigma scheduler
import numpy as np

logger = logging.getLogger(__name__)

class SDXLInference:
    def __init__(
        self,
        model_path=None,
        device="cuda",
        dtype=torch.float16,
        variant="fp16",
        use_resolution_binning=True,
        sigma_min=0.0292,
        sigma_data=1.0,
    ):
        """
        Initialize SDXL inference with custom training parameters.

        Args:
            model_path (str, optional): Path to the model. If None, pipeline must be set manually
            device (str): Device to run inference on
            dtype (torch.dtype): Model dtype
            variant (str): Model variant (e.g., 'fp16')
            use_resolution_binning (bool): Whether to use resolution-dependent sigma scaling
            sigma_min (float): Minimum sigma value for noise schedule
            sigma_data (float): Sigma data value for noise schedule
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.use_resolution_binning = use_resolution_binning
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        
        # Only load pipeline if model_path is provided
        if model_path is not None:
            logger.info(f"Initializing SDXL inference with model: {model_path}")
            try:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant=variant
                ).to(device)
                
                # Use our custom scheduler settings
                self.pipeline.scheduler.register_to_config(
                    use_resolution_binning=use_resolution_binning,
                    sigma_min=sigma_min,
                    sigma_data=sigma_data
                )
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise

    def generate_images(
        self,
        prompts,
        negative_prompt=None,
        num_images_per_prompt=1,
        guidance_scale=5.0,
        num_inference_steps=28,
        height=1024,
        width=1024,
        output_dir=None,
        **additional_kwargs
    ):
        """
        Generate images using the SDXL model.

        Args:
            prompts (str or list): Prompt(s) to generate images from
            negative_prompt (str, optional): Negative prompt for generation
            num_images_per_prompt (int): Number of images per prompt
            guidance_scale (float): CFG scale
            num_inference_steps (int): Number of denoising steps
            height (int): Image height
            width (int): Image width
            output_dir (str, optional): Directory to save generated images
            **additional_kwargs: Additional arguments for pipeline
        
        Returns:
            dict: Generation results including images and metadata
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        results = []
        
        try:
            # Update noise schedule based on resolution if enabled
            if self.use_resolution_binning:
                sigmas = get_sigmas(
                    num_inference_steps=num_inference_steps,
                    sigma_min=self.sigma_min,
                    height=height,
                    width=width
                ).to(self.device)
                self.pipeline.scheduler.sigmas = sigmas
            
            # Generate images
            for prompt in prompts:
                start_time = time.time()
                
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    **additional_kwargs
                )
                
                generation_time = time.time() - start_time
                
                # Save images if output directory provided
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    for idx, image in enumerate(output.images):
                        image_path = output_dir / f"{len(results)}_{idx}.png"
                        image.save(image_path)
                
                # Collect results
                result = {
                    "prompt": prompt,
                    "images": output.images,
                    "generation_time": generation_time,
                    "metadata": {
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "height": height,
                        "width": width
                    }
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
            
        finally:
            torch.cuda.empty_cache()

    def run_validation(
        self,
        prompts,
        output_dir=None,
        log_to_wandb=False,
        **generation_kwargs
    ):
        """
        Run validation using the model.

        Args:
            prompts (list): Validation prompts
            output_dir (str, optional): Directory to save validation images
            log_to_wandb (bool): Whether to log results to W&B
            **generation_kwargs: Arguments for image generation
        
        Returns:
            dict: Validation metrics
        """
        logger.info("Starting validation run")
        
        try:
            # Generate validation images
            results = self.generate_images(
                prompts=prompts,
                output_dir=output_dir,
                **generation_kwargs
            )
            
            # Calculate metrics
            metrics = {
                "validation/avg_generation_time": np.mean([r["generation_time"] for r in results]),
                "validation/total_images": sum(len(r["images"]) for r in results),
                "validation/num_prompts": len(prompts)
            }
            
            # Log to W&B if enabled
            if log_to_wandb and wandb.run is not None:
                wandb_images = []
                for result in results:
                    for img in result["images"]:
                        wandb_images.append(
                            wandb.Image(
                                img,
                                caption=f"Prompt: {result['prompt']}\n"
                                f"Time: {result['generation_time']:.2f}s"
                            )
                        )
                
                wandb.log({
                    "validation/images": wandb_images,
                    **metrics
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
