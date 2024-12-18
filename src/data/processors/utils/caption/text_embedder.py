import torch
import logging
import warnings
import traceback
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from typing import Dict, List, Union, Optional, Any
import random
import numpy as np
from src.config.config import TextEmbedderConfig

logger = logging.getLogger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, 
    subfolder: str = "text_encoder",
    revision: Optional[str] = None
):
    """Load the correct text encoder class based on config."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder=subfolder,
        revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class TextEmbedder:
    def __init__(
        self, 
        config: TextEmbedderConfig, 
        tokenizers: Optional[Dict[str, Any]] = None,
        text_encoders: Optional[Dict[str, Any]] = None
    ):
        """Initialize text embedder with both encoders.
        
        Args:
            config: Configuration for the text embedder
            tokenizers: Optional pre-initialized tokenizers
            text_encoders: Optional pre-initialized text encoders
        """
        self.config = config
        
        # Load or use provided tokenizers
        if tokenizers is not None:
            self.tokenizer_one = tokenizers.get("tokenizer_one")
            self.tokenizer_two = tokenizers.get("tokenizer_two")
            logger.info("Using provided tokenizers")
        else:
            # Load tokenizers from config
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                config.model_name,
                subfolder="tokenizer",
                use_fast=False,
                revision=config.revision
            )
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                config.model_name,
                subfolder="tokenizer_2",
                use_fast=False,
                revision=config.revision
            )
            logger.info("Initialized new tokenizers")
        
        # Load or use provided text encoders
        if text_encoders is not None:
            self.text_encoder_one = text_encoders.get("text_encoder_one")
            self.text_encoder_two = text_encoders.get("text_encoder_two")
            logger.info("Using provided text encoders")
        else:
            # Load text encoders from config
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                config.model_name, 
                subfolder="text_encoder",
                revision=config.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                config.model_name,
                subfolder="text_encoder_2",
                revision=config.revision
            )
            
            self.text_encoder_one = text_encoder_cls_one.from_pretrained(
                config.model_name,
                subfolder="text_encoder",
                torch_dtype=config.dtype,
                revision=config.revision,
                variant=config.variant
            )
            self.text_encoder_two = text_encoder_cls_two.from_pretrained(
                config.model_name,
                subfolder="text_encoder_2",
                torch_dtype=config.dtype,
                revision=config.revision,
                variant=config.variant
            )
            logger.info("Initialized new text encoders")
        
        # Move to device and eval mode
        self.text_encoder_one.to(config.device).eval()
        self.text_encoder_two.to(config.device).eval()
        
        logger.info(
            f"Initialized TextEmbedder:\n"
            f"- Model: {config.model_name}\n"
            f"- Device: {config.device}\n"
            f"- Dtype: {config.dtype}\n"
            f"- Revision: {config.revision}\n"
            f"- Variant: {config.variant}"
        )

    @torch.no_grad()
    def __call__(
        self, 
        text: Union[str, List[str]], 
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        clean_caption: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Process text input and return embeddings.
        
        Args:
            text: Input text or list of texts
            device: Optional target device
            num_images_per_prompt: Number of images to generate per prompt
            clean_caption: Whether to clean/preprocess the captions
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing prompt_embeds and pooled_prompt_embeds
        """
        try:
            # Convert single string to list
            if isinstance(text, str):
                text = [text]
                
            # Clean captions if requested
            if clean_caption:
                text = [self._clean_caption(t) for t in text]
            
            # Use provided device or fall back to config device
            target_device = device if device is not None else self.config.device
            
            # Process with first encoder
            text_inputs_one = self.tokenizer_one(
                text,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(target_device)
            
            text_inputs_two = self.tokenizer_two(
                text,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(target_device)
            
            # Get embeddings from both encoders
            with torch.no_grad():
                # First text encoder
                encoder_output_one = self.text_encoder_one(
                    text_inputs_one.input_ids,
                    attention_mask=text_inputs_one.attention_mask,
                    output_hidden_states=True,
                    return_dict=False,
                )
                pooled_prompt_embeds_one = encoder_output_one[0]  # First element is pooled output
                prompt_embeds_one = encoder_output_one[-1][-2]  # Second to last hidden state
                
                # Second text encoder
                encoder_output_two = self.text_encoder_two(
                    text_inputs_two.input_ids,
                    attention_mask=text_inputs_two.attention_mask,
                    output_hidden_states=True,
                    return_dict=False,
                )
                pooled_prompt_embeds_two = encoder_output_two[0]  # First element is pooled output
                prompt_embeds_two = encoder_output_two[-1][-2]  # Second to last hidden state
                
                # Reshape prompt embeds if needed
                bs_embed, seq_len, _ = prompt_embeds_one.shape
                prompt_embeds_one = prompt_embeds_one.view(bs_embed, seq_len, -1)
                prompt_embeds_two = prompt_embeds_two.view(bs_embed, seq_len, -1)
            
            # Concatenate embeddings
            prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1)
            
            # Duplicate for multiple images per prompt
            if num_images_per_prompt > 1:
                prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
            
            return {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
            }
            
        except Exception as e:
            logger.error(f"Error in text embedding:\n  Type: {type(e).__name__}\n  Message: {str(e)}\n\nTraceback:\n  {traceback.format_exc()}")
            raise

    def encode_prompt_batch(
        self,
        batch: Dict[str, Any],
        caption_column: str,
        proportion_empty_prompts: float = 0,
        is_train: bool = True,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of prompts and return embeddings.
        
        Args:
            batch: Input batch containing captions
            caption_column: Name of caption column in batch
            proportion_empty_prompts: Proportion of prompts to replace with empty string
            is_train: Whether in training mode (affects caption selection)
            device: Optional target device
            **kwargs: Additional arguments passed to __call__
        """
        prompt_batch = batch[caption_column]

        # Handle empty prompts for classifier-free guidance
        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        return self(captions, device=device, **kwargs)
    
    def _clean_caption(self, caption: str) -> str:
        """Clean/preprocess a caption string."""
        # Add any caption cleaning logic here
        return caption.strip()
