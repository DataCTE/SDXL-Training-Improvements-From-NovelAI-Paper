from huggingface_hub import HfApi
import logging
import traceback
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers.optimization import Adafactor
from diffusers.optimization import get_scheduler
from data.dataset import CustomDataset
from data.tag_weighter import TagBasedLossWeighter
from training.vae_finetuner import VAEFineTuner

logger = logging.getLogger(__name__)

def push_to_hub(model_id, model_path, private=False, model_card=None):
    """
    Push model to Hugging Face Hub
    
    Args:
        model_id (str): Hugging Face Hub model ID
        model_path (str): Local path to model files
        private (bool): Whether to create a private repository
        model_card (str): Model card content
    """
    try:
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=model_id,
                private=private,
                repo_type="model",
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repository creation warning (may already exist): {str(e)}")
        
        # Upload model files
        logger.info(f"Uploading model to {model_id}...")
        api.upload_folder(
            repo_id=model_id,
            folder_path=model_path,
            commit_message="Upload model files"
        )
        
        # Upload model card if provided
        if model_card:
            logger.info("Uploading model card...")
            api.upload_file(
                repo_id=model_id,
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                commit_message="Update model card"
            )
        
        logger.info(f"Successfully pushed model to https://huggingface.co/{model_id}")
        
    except Exception as e:
        logger.error(f"Error pushing to Hub: {str(e)}")
        raise
