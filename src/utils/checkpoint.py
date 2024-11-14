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
        pipeline = StableDiffusionXLPipeline.from_pretrained(checkpoint_dir, torch_dtype=torch.float16)
        
        models["unet"] = pipeline.unet
        models["vae"] = pipeline.vae
        models["text_encoder"] = CLIPTextModel.from_pretrained(pipeline.text_encoder.config)
        models["text_encoder_2"] = CLIPTextModel.from_pretrained(pipeline.text_encoder_2.config)
        models["tokenizer"] = pipeline.tokenizer
        models["tokenizer_2"] = pipeline.tokenizer_2
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
    


def save_checkpoint(checkpoint_dir, models, train_components, training_state):
    """
    Save checkpoint in diffusers format with safetensors support
    """
    try:
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        pipeline = StableDiffusionXLPipeline(
            unet=models["unet"],
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=models["tokenizer"],
            tokenizer_2=models["tokenizer_2"],
            torch_dtype=torch.float16
        )
        pipeline.save_pretrained(checkpoint_dir, safe_serialization=True)
        
        # Save training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, training_state_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error(f"Checkpoint directory: {checkpoint_dir}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def save_final_outputs(args, models, training_history, train_components):
    """Save final model outputs using safetensors"""
    try:
        logger.info("Saving final model outputs...")
        
        # Save final UNet
        final_model_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        models["unet"].save_pretrained(final_model_path, safe_serialization=True)
        
        # Save final EMA model if it exists
        if models.get("ema_model") is not None:
            logger.info("Saving final EMA model...")
            ema_path = os.path.join(args.output_dir, "final_ema")
            os.makedirs(ema_path, exist_ok=True)
            
            ema_model = models["ema_model"].get_model()
            ema_model.save_pretrained(ema_path, safe_serialization=True)
        
        # Save final training metrics
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
            
        logger.info("Final outputs saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving final outputs: {str(e)}")
        raise
