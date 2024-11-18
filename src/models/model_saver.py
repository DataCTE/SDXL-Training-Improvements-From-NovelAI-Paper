import os
import torch
import logging
from typing import Dict, Any, Optional
from src.models.SDXL.pipeline import StableDiffusionXLPipeline
from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class ModelSaver:
    """Handles saving SDXL models in diffusers format."""
    
    @staticmethod
    def save_diffusers_checkpoint(
        pipeline: StableDiffusionXLPipeline,
        save_dir: str,
        models: Dict[str, Any],
        epoch: int,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        commit_message: Optional[str] = None
    ):
        """Save pipeline checkpoint in diffusers format.
        
        Args:
            pipeline: StableDiffusionXLPipeline instance
            save_dir: Directory to save the checkpoint
            models: Dictionary of model components
            epoch: Current training epoch
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hugging Face Hub repo ID if pushing
            commit_message: Commit message if pushing
        """
        try:
            checkpoint_dir = os.path.join(save_dir, f"checkpoint-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model components
            component_mapping = {
                "vae": "vae",
                "text_encoder": "text_encoder",
                "text_encoder_2": "text_encoder_2", 
                "tokenizer": "tokenizer",
                "tokenizer_2": "tokenizer_2",
                "unet": "unet",
                "scheduler": "scheduler"
            }
            
            # Save each component
            for name, save_key in component_mapping.items():
                if name in models:
                    model = models[name]
                    component_path = os.path.join(checkpoint_dir, save_key)
                    os.makedirs(component_path, exist_ok=True)
                    
                    # Save model state dict
                    if hasattr(model, "state_dict"):
                        state_dict = model.state_dict()
                        save_file(
                            state_dict,
                            os.path.join(component_path, "diffusion_pytorch_model.safetensors")
                        )
                    
                    # Save config
                    if hasattr(model, "config"):
                        if isinstance(model.config, PretrainedConfig):
                            model.config.save_pretrained(component_path)
                        else:
                            # Handle custom configs
                            config_dict = model.config
                            if hasattr(config_dict, "to_dict"):
                                config_dict = config_dict.to_dict()
                            torch.save(config_dict, os.path.join(component_path, "config.json"))
                    
                    # Save special files for tokenizers
                    if name.startswith("tokenizer"):
                        if hasattr(model, "save_pretrained"):
                            model.save_pretrained(component_path)
                            
            # Save pipeline config
            pipeline.config.save_pretrained(checkpoint_dir)
            
            logger.info(f"Saved diffusers checkpoint to {checkpoint_dir}")
            
            # Push to Hub if requested
            if push_to_hub and repo_id:
                if commit_message is None:
                    commit_message = f"Epoch {epoch} checkpoint"
                    
                # Create or get repo
                create_repo(repo_id, exist_ok=True)
                
                # Upload checkpoint folder
                upload_folder(
                    repo_id=repo_id,
                    folder_path=checkpoint_dir,
                    commit_message=commit_message
                )
                
                logger.info(f"Pushed checkpoint to {repo_id}")
                
            return checkpoint_dir
            
        except Exception as e:
            logger.error(f"Failed to save diffusers checkpoint: {str(e)}")
            raise
