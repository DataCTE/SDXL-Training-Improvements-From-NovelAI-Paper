import torch
import logging
import numpy as np
import time
from pathlib import Path
import wandb
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    AutoencoderKL
)
from src.training.loss import get_sigmas
from src.utils.prompt_utils import process_prompt, get_prompt_embeds
from src.utils.latent_utils import  get_latents_from_seed

logger = logging.getLogger(__name__)

class SDXLInference:
    def __init__(
        self,
        model_path=None,
        device="cuda",
        dtype=torch.float16,
        variant="fp16",
        use_v_prediction=True,
        use_resolution_binning=True,
        use_zero_terminal_snr=True,
        sigma_min=0.0292,
        sigma_max=160.0,
        sigma_data=1.0,
        min_snr_gamma=5.0,
        noise_offset=0.0357,
        prompt_max_length=77,
    ):
        """
        Initialize SDXL inference with NovelAI improvements.

        Args:
            model_path (str, optional): Path to the model. If None, pipeline must be set manually
            device (str): Device to run inference on
            dtype (torch.dtype): Model dtype
            variant (str): Model variant (e.g., 'fp16')
            use_v_prediction (bool): Whether to use v-prediction parameterization
            use_resolution_binning (bool): Whether to use resolution-dependent sigma scaling
            use_zero_terminal_snr (bool): Whether to use Zero Terminal SNR
            sigma_min (float): Minimum sigma value for noise schedule
            sigma_max (float): Maximum sigma value for noise schedule
            sigma_data (float): Sigma data value for noise schedule
            min_snr_gamma (float): Minimum SNR value for loss weighting
            noise_offset (float): Noise offset for improved sample quality
            prompt_max_length (int): Maximum prompt length
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.use_v_prediction = use_v_prediction
        self.use_resolution_binning = use_resolution_binning
        self.use_zero_terminal_snr = use_zero_terminal_snr
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.min_snr_gamma = min_snr_gamma
        self.noise_offset = noise_offset
        self.prompt_max_length = prompt_max_length
        
        # Only load pipeline if model_path is provided
        if model_path is not None:
            logger.info("Initializing SDXL inference with model: %s", model_path)
            try:
                # Initialize scheduler with NovelAI improvements
                scheduler = EulerDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                    use_karras_sigmas=True,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    steps_offset=1,
                )
                
                # Initialize pipeline
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant=variant,
                    scheduler=scheduler
                ).to(device)
                
                # Update VAE config
                if isinstance(self.pipeline.vae, AutoencoderKL):
                    self.pipeline.vae.config.scaling_factor = 0.13025
                
                # Enable memory efficient attention
                self.pipeline.enable_xformers_memory_efficient_attention()
                
                # Set custom configs
                self.pipeline.scheduler.register_to_config(
                    use_v_prediction=use_v_prediction,
                    use_resolution_binning=use_resolution_binning,
                    use_zero_terminal_snr=use_zero_terminal_snr,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sigma_data=sigma_data,
                    min_snr_gamma=min_snr_gamma
                )
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error("Failed to load model: %s", str(e), exc_info=True)
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
        seed=None,
        output_dir=None,
        rescale_cfg=True,
        scale_method="karras",
        rescale_multiplier=0.7,
        **additional_kwargs
    ):
        """
        Generate images using SDXL with NovelAI improvements.

        Args:
            prompts (str or list): Prompt(s) to generate images from
            negative_prompt (str, optional): Negative prompt for generation
            num_images_per_prompt (int): Number of images per prompt
            guidance_scale (float): CFG scale
            num_inference_steps (int): Number of denoising steps
            height (int): Image height
            width (int): Image width
            seed (int, optional): Random seed for reproducibility
            output_dir (str, optional): Directory to save generated images
            rescale_cfg (bool): Whether to use CFG rescaling
            scale_method (str): Method for CFG rescaling ('karras' or 'simple')
            rescale_multiplier (float): Multiplier for CFG rescaling
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
                    sigma_max_base=self.sigma_max,
                    height=height,
                    width=width
                ).to(self.device)
                self.pipeline.scheduler.sigmas = sigmas
            
            # Process prompts and get embeddings
            prompt_embeds = []
            for prompt in prompts:
                processed_prompt = process_prompt(
                    prompt,
                    max_length=self.prompt_max_length
                )
                embeds = get_prompt_embeds(
                    processed_prompt,
                    self.pipeline.tokenizer,
                    self.pipeline.text_encoder,
                    self.device,
                    self.dtype
                )
                prompt_embeds.append(embeds)
            
            # Generate images
            for idx, (prompt, embed) in enumerate(zip(prompts, prompt_embeds)):
                start_time = time.time()
                
                # Get latents from seed if provided
                latents = None
                if seed is not None:
                    latents = get_latents_from_seed(
                        seed + idx,
                        num_images_per_prompt,
                        height,
                        width,
                        self.pipeline.vae.config.latent_channels,
                        self.device,
                        self.dtype
                    )
                
                # Apply CFG rescaling if enabled
                effective_guidance_scale = guidance_scale
                if rescale_cfg:
                    if scale_method == "karras":
                        effective_guidance_scale = guidance_scale * rescale_multiplier * (1 - torch.exp(-guidance_scale))
                    else:  # simple
                        effective_guidance_scale = guidance_scale * rescale_multiplier
                
                # Generate with NovelAI improvements
                output = self.pipeline(
                    prompt_embeds=embed,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=effective_guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    latents=latents,
                    noise_offset=self.noise_offset if self.use_v_prediction else None,
                    **additional_kwargs
                )
                
                generation_time = time.time() - start_time
                
                # Save images if output directory provided
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    for img_idx, image in enumerate(output.images):
                        image_path = output_dir / f"{len(results)}_{img_idx}.png"
                        image.save(image_path)
                
                # Collect results
                result = {
                    "prompt": prompt,
                    "images": output.images,
                    "generation_time": generation_time,
                    "metadata": {
                        "guidance_scale": guidance_scale,
                        "effective_guidance_scale": effective_guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "height": height,
                        "width": width,
                        "seed": seed + idx if seed is not None else None,
                        "use_v_prediction": self.use_v_prediction,
                        "use_resolution_binning": self.use_resolution_binning,
                        "use_zero_terminal_snr": self.use_zero_terminal_snr,
                        "rescale_cfg": rescale_cfg,
                        "scale_method": scale_method,
                        "rescale_multiplier": rescale_multiplier
                    }
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error("Error during image generation: %s", str(e), exc_info=True)
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
                                f"Time: {result['generation_time']:.2f}s\n"
                                f"CFG: {result['metadata']['guidance_scale']:.1f} "
                                f"(effective: {result['metadata']['effective_guidance_scale']:.1f})"
                            )
                        )
                
                wandb.log({
                    "validation/images": wandb_images,
                    **metrics
                })
            
            return metrics
            
        except Exception as e:
            logger.error("Validation failed: %s", str(e), exc_info=True)
            raise
