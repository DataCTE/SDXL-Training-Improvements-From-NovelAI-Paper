import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

def create_model_card(args, training_history):
    """
    Create a model card with training details and performance metrics
    
    Args:
        args: Training arguments
        training_history: Dictionary containing training metrics and history
        
    Returns:
        str: Formatted model card markdown text
    """
    try:
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Format training metrics
        final_loss = training_history.get("final_loss", "N/A")
        best_validation_loss = training_history.get("best_validation_loss", "N/A")
        total_steps = training_history.get("total_steps", 0)
        
        # Create model card content
        model_card = f"""
# SDXL Fine-tuned Model

## Model Details

- **Model Type:** SDXL Fine-tuned
- **Date:** {current_date}
- **Training Duration:** {args.num_epochs} epochs
- **Base Model:** {args.model_path}
- **License:** CreativeML Open RAIL-M

## Training Details

### Dataset
- **Data Source:** {args.data_dir}
- **Batch Size:** {args.batch_size}
- **Gradient Accumulation Steps:** {args.gradient_accumulation_steps}

### Training Parameters
- **Learning Rate:** {args.learning_rate}
- **Optimizer:** {"Adafactor" if args.use_adafactor else "AdamW8bit"}
- **EMA Decay:** {args.ema_decay if hasattr(args, 'ema_decay') else 'N/A'}
- **VAE Finetuning:** {"Enabled" if args.finetune_vae else "Disabled"}
- **Total Training Steps:** {total_steps}

### Performance Metrics
- **Final Loss:** {final_loss}
- **Best Validation Loss:** {best_validation_loss}

### Model Configuration
- **Precision:** bfloat16
- **Compiled:** {"Yes" if args.enable_compile else "No"}
- **Compilation Mode:** {args.compile_mode if args.enable_compile else "N/A"}

## Usage

This model can be used with the standard SDXL pipeline:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "{args.hub_model_id if args.push_to_hub else 'path/to/model'}",
    torch_dtype=torch.float16
)
pipeline.to("cuda")

prompt = "your prompt here"
image = pipeline(prompt).images[0]
image.save("output.png")
```

## Limitations and Biases

This model inherits the limitations and biases from the base SDXL model. Users should be aware of potential biases in the training data.

## License

This model is licensed under the CreativeML Open RAIL-M license. Please see the LICENSE file for details.
"""
        return model_card
        
    except Exception as e:
        logger.error(f"Error creating model card: {str(e)}")
        return None

def save_model_card(model_card, output_dir):
    """
    Save model card to file
    
    Args:
        model_card (str): Model card content
        output_dir (str): Directory to save the model card
    """
    try:
        output_path = Path(output_dir) / "README.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        logger.info(f"Saved model card to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving model card: {str(e)}")

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
