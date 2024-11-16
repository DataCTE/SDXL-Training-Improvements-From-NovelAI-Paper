"""Hugging Face Hub integration utilities for SDXL models.

This module provides utilities for managing model cards and pushing models
to the Hugging Face Hub. It handles model card generation, formatting,
and safe model uploads.

Key Features:
- Model card generation with training metrics
- Safe model card saving
- Hub upload with proper error handling
- Training metrics documentation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import traceback

from huggingface_hub import HfApi, upload_folder
from tqdm import tqdm

logger = logging.getLogger(__name__)

def create_model_card(
    config: Dict[str, Any],
    training_history: Dict[str, Any],
    model_name: Optional[str] = None
) -> str:
    """Create a detailed model card with training metrics.
    
    Args:
        config: Training configuration
        training_history: Training metrics and history
        model_name: Optional custom model name
        
    Returns:
        str: Formatted model card markdown
        
    Note:
        Includes training parameters, dataset info, and performance metrics
    """
    try:
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Extract metrics
        final_loss = training_history.get("loss_history", [])[-1] if training_history.get("loss_history") else "N/A"
        best_validation_loss = min(training_history.get("validation_scores", [float('inf')]))
        total_steps = training_history.get("total_steps", 0)
        
        # Basic model info
        card_sections = [
            f"# {model_name or 'SDXL Fine-tuned Model'}",
            "\n## Model Details",
            f"- **Model Type:** SDXL Fine-tuned",
            f"- **Date:** {current_date}",
            f"- **Training Duration:** {config.num_epochs} epochs",
            f"- **Base Model:** {config.model_path}",
            "- **License:** CreativeML Open RAIL-M",
            
            "\n## Training Details",
            "\n### Dataset",
            f"- **Data Source:** {config.data_dir}",
            f"- **Batch Size:** {config.batch_size}",
            f"- **Gradient Accumulation Steps:** {config.gradient_accumulation_steps}",
            
            "\n### Training Parameters",
            f"- **Learning Rate:** {config.learning_rate}",
            f"- **Optimizer:** {'Adafactor' if config.use_adafactor else 'AdamW8bit'}",
            f"- **Weight Decay:** {config.weight_decay}",
            f"- **Warmup Steps:** {config.warmup_steps}",
            
            "\n### Performance Metrics",
            f"- **Final Loss:** {final_loss}",
            f"- **Best Validation Loss:** {best_validation_loss}",
            f"- **Total Training Steps:** {total_steps}",
        ]
        
        # Add VAE info if used
        if config.vae_path:
            card_sections.extend([
                "\n### VAE Configuration",
                f"- **VAE Path:** {config.vae_path}",
                f"- **VAE Learning Rate:** {config.vae_learning_rate}"
            ])
            
        # Add EMA info if used
        if config.use_ema:
            card_sections.extend([
                "\n### EMA Configuration",
                f"- **EMA Decay:** {config.ema_decay}",
                f"- **EMA Update Every:** {config.ema_update_every}"
            ])
            
        # Add model usage
        card_sections.extend([
            "\n## Usage",
            "This model can be used with the standard SDXL pipeline:",
            "```python",
            "from diffusers import StableDiffusionXLPipeline",
            "import torch",
            "",
            "pipeline = StableDiffusionXLPipeline.from_pretrained(",
            f"    '{config.hub_model_id if config.push_to_hub else 'path/to/model'}',",
            "    torch_dtype=torch.float16",
            ")",
            "pipeline.to('cuda')",
            "",
            "prompt = 'your prompt here'",
            "image = pipeline(prompt).images[0]",
            "image.save('output.png')",
            "```",
            
            "\n## Limitations and Biases",
            "This model inherits the limitations and biases from the base SDXL model. Users should be aware of potential biases in the training data.",
            
            "\n## License",
            "This model is licensed under the CreativeML Open RAIL-M license. Please see the LICENSE file for details."
        ])
        
        logger.info("Model card generated successfully for %s", model_name)
        return "\n".join(card_sections)
        
    except Exception as e:
        logger.error("Failed to create model card: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise

def save_model_card(
    model_card: str,
    output_dir: str,
    filename: str = "README.md"
) -> None:
    """Save model card to file with proper error handling.
    
    Args:
        model_card: Model card content
        output_dir: Directory to save the card
        filename: Output filename (default: README.md)
        
    Raises:
        Exception: If saving fails
    """
    try:
        save_path = Path(output_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(model_card)
            
        logger.info("Model card saved to %s", save_path)
        
    except Exception as e:
        logger.error("Failed to save model card: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise

def push_to_hub(
    model_id: str,
    model_path: str,
    private: bool = False,
    model_card: Optional[str] = None,
    token: Optional[str] = None
) -> None:
    """Push model to Hugging Face Hub with proper error handling.
    
    Args:
        model_id: Hugging Face Hub model ID
        model_path: Local path to model files
        private: Whether to create private repository
        model_card: Optional model card content
        token: Optional Hugging Face token
        
    Raises:
        Exception: If upload fails
    """
    try:
        logger.info("Pushing model to Hub: %s", model_id)
        
        # Initialize Hub API
        api = HfApi(token=token)
        
        # Create repository if needed
        try:
            api.create_repo(
                repo_id=model_id,
                private=private,
                repo_type="model",
                exist_ok=True
            )
        except Exception as e:
            logger.warning("Repository creation warning (may already exist): %s", str(e))
        
        # Save model card if provided
        if model_card:
            card_path = Path(model_path) / "README.md"
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(model_card)
        
        # Upload model files
        logger.info("Uploading model files from %s", model_path)
        upload_folder(
            folder_path=model_path,
            repo_id=model_id,
            repo_type="model",
            private=private,
            token=token
        )
        
        logger.info("Model successfully pushed to %s", model_id)
        
    except Exception as e:
        logger.error("Failed to push to Hub: %s", str(e))
        logger.debug("Traceback: %s", traceback.format_exc())
        raise
