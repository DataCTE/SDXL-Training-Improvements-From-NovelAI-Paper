import logging
import torch
import os
import traceback
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModel
import json

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_dir, models, train_components):
    """ load using diffusers """
    try:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        return StableDiffusionXLPipeline.from_pretrained(model_path=checkpoint_dir, torch_dtype=torch.float16)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise

def save_checkpoint(checkpoint_dir, models, train_components, training_state):
    """Save checkpoint in diffusers format with safetensors support"""
    try:
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model configs first
        os.makedirs(os.path.join(checkpoint_dir, "unet"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "vae"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "text_encoder"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "text_encoder_2"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "scheduler"), exist_ok=True)
        
        with open(os.path.join(checkpoint_dir, "unet/config.json"), "w") as f:
            json.dump(models["unet"].config.to_dict(), f, indent=2)
        with open(os.path.join(checkpoint_dir, "vae/config.json"), "w") as f:
            json.dump(models["vae"].config.to_dict(), f, indent=2)
        with open(os.path.join(checkpoint_dir, "text_encoder/config.json"), "w") as f:
            json.dump(models["text_encoder"].config.to_dict(), f, indent=2)
        with open(os.path.join(checkpoint_dir, "text_encoder_2/config.json"), "w") as f:
            json.dump(models["text_encoder_2"].config.to_dict(), f, indent=2)
        with open(os.path.join(checkpoint_dir, "scheduler/scheduler_config.json"), "w") as f:
            json.dump(models["scheduler"].config, f, indent=2)
            
        # Save model_index.json
        model_index = {
            "_class_name": "StableDiffusionXLPipeline",
            "_diffusers_version": "0.21.4",
            "force_zeros_for_empty_prompt": True,
            "add_watermarker": False,
            "requires_safety_checker": False
        }
        with open(os.path.join(checkpoint_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
            
        # Save pipeline with weights
        pipeline = StableDiffusionXLPipeline(
            unet=models["unet"],
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            scheduler=models["scheduler"],
            torch_dtype=torch.float16
        )
        pipeline.save_pretrained(checkpoint_dir, safe_serialization=True)
        
        # Save training state
        if training_state is not None:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save(training_state, training_state_path)
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error(f"Checkpoint directory: {checkpoint_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def save_final_outputs(args, models, training_history, train_components):
    """Save final model outputs in proper diffusers format"""
    try:
        logger.info("Saving final model outputs...")
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        
        # Save model configs first
        os.makedirs(os.path.join(final_path, "unet"), exist_ok=True)
        os.makedirs(os.path.join(final_path, "vae"), exist_ok=True)
        os.makedirs(os.path.join(final_path, "text_encoder"), exist_ok=True)
        os.makedirs(os.path.join(final_path, "text_encoder_2"), exist_ok=True)
        os.makedirs(os.path.join(final_path, "scheduler"), exist_ok=True)
        
        with open(os.path.join(final_path, "unet/config.json"), "w") as f:
            json.dump(models["unet"].config.to_dict(), f, indent=2)
        with open(os.path.join(final_path, "vae/config.json"), "w") as f:
            json.dump(models["vae"].config.to_dict(), f, indent=2)
        with open(os.path.join(final_path, "text_encoder/config.json"), "w") as f:
            json.dump(models["text_encoder"].config.to_dict(), f, indent=2)
        with open(os.path.join(final_path, "text_encoder_2/config.json"), "w") as f:
            json.dump(models["text_encoder_2"].config.to_dict(), f, indent=2)
        with open(os.path.join(final_path, "scheduler/scheduler_config.json"), "w") as f:
            json.dump(models["scheduler"].config, f, indent=2)
            
        # Save model_index.json
        model_index = {
            "_class_name": "StableDiffusionXLPipeline",
            "_diffusers_version": "0.21.4",
            "force_zeros_for_empty_prompt": True,
            "add_watermarker": False,
            "requires_safety_checker": False
        }
        with open(os.path.join(final_path, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # Save final pipeline with weights
        pipeline = StableDiffusionXLPipeline(
            unet=models["unet"],
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            scheduler=models["scheduler"],
            torch_dtype=torch.float16
        )
        pipeline.save_pretrained(final_path, safe_serialization=True)
        
        # Save final EMA model if it exists
        if models.get("ema_model") is not None:
            logger.info("Saving final EMA model...")
            ema_path = os.path.join(args.output_dir, "final_ema")
            os.makedirs(ema_path, exist_ok=True)
            
            # Save EMA configs
            os.makedirs(os.path.join(ema_path, "unet"), exist_ok=True)
            with open(os.path.join(ema_path, "unet/config.json"), "w") as f:
                json.dump(models["ema_model"].config.to_dict(), f, indent=2)
                
            # Save EMA pipeline
            ema_pipeline = StableDiffusionXLPipeline(
                unet=models["ema_model"],
                vae=models["vae"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                scheduler=models["scheduler"],
                torch_dtype=torch.float16
            )
            ema_pipeline.save_pretrained(ema_path, safe_serialization=True)
        
        # Save final training metrics
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
            
        logger.info("Final outputs saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving final outputs: {str(e)}")
        raise
