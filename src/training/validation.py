import torch
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from contextlib import nullcontext
from src.training.metrics import MetricsManager
from src.training.loss_functions import forward_pass
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
from src.data.prompt.caption_processor import CaptionProcessor
from src.utils.progress import ProgressTracker


logger = logging.getLogger(__name__)

def generate_validation_images(
    pipeline: StableDiffusionXLPipeline,
    prompts: List[str],
    save_dir: str,
    device: torch.device,
    num_inference_steps: int = 28,
    guidance_scale: float = 5.0,
    height: int = 1024,
    width: int = 1024,
    num_images_per_prompt: int = 1,
    progress_callback: Optional[Callable] = None
) -> List[str]:
    """Generate validation images using SDXL pipeline."""
    os.makedirs(save_dir, exist_ok=True)
    generated_paths = []
    
    # Pipeline should already be on device from trainer
    
    for idx, prompt in enumerate(prompts):
        try:
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    output_type="pil"
                )
                
                # Save generated images
                for img_idx, image in enumerate(output.images):
                    save_path = os.path.join(save_dir, f"val_{idx}_{img_idx}.png")
                    image.save(save_path)
                    generated_paths.append(save_path)
                
                if progress_callback:
                    progress_callback(1, {
                        'current_prompt': prompt,
                        'images_generated': len(generated_paths)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to generate validation image for prompt {prompt}: {str(e)}")
            continue
            
    return generated_paths

def run_validation(
    args,
    models: Dict[str, Any],
    components: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    metrics_manager: Optional[MetricsManager] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, float]:
    """Run validation with proper error handling and metrics tracking."""
    try:
        # Set eval mode for all models
        for model in models.values():
            if isinstance(model, torch.nn.Module):
                model.eval()

        metrics = {}
        
        # Run validation on training data if available
        val_dataloader = components.get("val_dataloader")
        if val_dataloader is not None:
            total_val_loss = 0.0
            num_val_steps = 0

            # Run validation steps
            with torch.no_grad():
                for batch in val_dataloader:
                    with torch.cuda.amp.autocast() if args.mixed_precision != "no" else nullcontext():
                        val_loss = forward_pass(
                            args,
                            models,
                            batch,
                            device,
                            dtype
                        )
                    total_val_loss += val_loss.item()
                    num_val_steps += 1
                    
                    if progress_callback:
                        progress_callback(1, {
                            'val_loss': val_loss.item(),
                            'avg_val_loss': total_val_loss / num_val_steps
                        })

            avg_val_loss = total_val_loss / num_val_steps if num_val_steps > 0 else 0.0
            metrics["val_loss"] = avg_val_loss
            
            logger.info(
                "Validation [Step %d] Average Loss: %.6f",
                global_step,
                avg_val_loss,
            )

        # Generate validation images if configured
        if hasattr(args, "validation_prompts") and args.validation_prompts:
            pipeline = StableDiffusionXLPipeline(
                vae=models["vae"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                unet=models["unet"],
                scheduler=models["scheduler"],
            ).to(device)
            
            # Get caption processor if available
            caption_processor = components.get("caption_processor")
            
            save_dir = os.path.join(args.output_dir, f"validation_images/step_{global_step}")
            
            # Create a nested progress tracker for image generation
            with ProgressTracker(
                "Generating Validation Images",
                total=len(args.validation_prompts),
                wandb_run=None  # We'll handle wandb logging at the parent level
            ) as img_progress:
                generated_paths = generate_validation_images(
                    pipeline=pipeline,
                    prompts=args.validation_prompts,
                    save_dir=save_dir,
                    device=device,
                    num_inference_steps=getattr(args, "validation_num_inference_steps", 28),
                    guidance_scale=getattr(args, "validation_guidance_scale", 5.5),
                    height=getattr(args, "validation_image_height", 1024),
                    width=getattr(args, "validation_image_width", 1024),
                    num_images_per_prompt=getattr(args, "validation_num_images_per_prompt", 1),
                    progress_callback=img_progress.update
                )
            
            metrics["num_validation_images"] = len(generated_paths)
            logger.info(
                "Generated %d validation images at step %d",
                len(generated_paths),
                global_step,
            )

        # Update metrics
        if metrics_manager is not None:
            for name, value in metrics.items():
                metrics_manager.update_metric(name, value)

        return metrics

    except Exception as e:
        logger.error("Validation failed: %s", str(e))
        raise