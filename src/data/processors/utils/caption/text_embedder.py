import torch
import logging
import warnings
import traceback
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from typing import Dict, List, Union, Optional, Any, Tuple
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
        tokenizers: Optional[Union[Dict[str, Any], Tuple[Any, Any]]] = None,
        text_encoders: Optional[Union[Dict[str, Any], Tuple[Any, Any]]] = None
    ):
        """
        Initialize text embedder with both encoders.
        
        Args:
            config: Configuration for the text embedder
            tokenizers: Optional pre-initialized tokenizers (dict or tuple)
            text_encoders: Optional pre-initialized text encoders (dict or tuple)
        """
        self.config = config
        
        # Load or use provided tokenizers
        if tokenizers is not None:
            if isinstance(tokenizers, dict):
                self.tokenizer_one = tokenizers.get("tokenizer_one")
                self.tokenizer_two = tokenizers.get("tokenizer_two")
            else:  # Assume tuple
                self.tokenizer_one, self.tokenizer_two = tokenizers
            logger.info("Using provided tokenizers")
        else:
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.tokenizer_subfolder,
                use_fast=config.use_fast_tokenizer,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.tokenizer_2_subfolder,
                use_fast=config.use_fast_tokenizer,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            logger.info("Initialized new tokenizers")

        # Validate tokenizers
        if not self.tokenizer_one or not self.tokenizer_two:
            raise ValueError("Tokenizers must not be None. Please check config or provided tokenizers.")

        # Load or use provided text encoders
        if text_encoders is not None:
            if isinstance(text_encoders, dict):
                self.text_encoder_one = text_encoders.get("text_encoder_one")
                self.text_encoder_two = text_encoders.get("text_encoder_two")
            else:
                self.text_encoder_one, self.text_encoder_two = text_encoders
            logger.info("Using provided text encoders")
        else:
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.text_encoder_subfolder
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.text_encoder_2_subfolder
            )

            self.text_encoder_one = text_encoder_cls_one.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.text_encoder_subfolder,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            self.text_encoder_two = text_encoder_cls_two.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                subfolder=config.text_encoder_2_subfolder,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            logger.info("Initialized new text encoders")

        # Validate encoders
        if not self.text_encoder_one or not self.text_encoder_two:
            raise ValueError("Text encoders must not be None. Please check config or provided encoders.")

        # Move to device and set eval mode
        self.device = torch.device(config.device)
        self.text_encoder_one.to(self.device).eval()
        self.text_encoder_two.to(self.device).eval()

        logger.info(
            f"Initialized TextEmbedder:\n"
            f"- Model: {config.model_name}\n"
            f"- Device: {self.device}\n"
            f"- Fast tokenizer: {config.use_fast_tokenizer}\n"
            f"- Low CPU memory: {config.low_cpu_mem_usage}"
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
        """
        Process text input and return embeddings.
        
        Args:
            text: A string or list of strings to embed.
            device: Optional device override.
            num_images_per_prompt: Number of images to generate per prompt (for replicate).
            clean_caption: Whether to clean/preprocess the text input before embedding.
            **kwargs: Additional arguments.
        """
        # Determine device and move encoders if necessary
        target_device = device if device is not None else self.device
        if not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)

        if text is None:
            raise ValueError("Input text cannot be None.")

        if isinstance(text, str):
            text = [text]

        if clean_caption:
            text = [self._clean_caption(t) for t in text]

        # Tokenize inputs
        try:
            text_inputs_one = self.tokenizer_one(
                text,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs_two = self.tokenizer_two(
                text,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise

        # Move tokenized inputs to the target device
        text_inputs_one = {
            k: v.to(device=target_device, dtype=torch.long) if k == "input_ids" 
            else v.to(device=target_device)
            for k, v in text_inputs_one.items()
        }
        text_inputs_two = {
            k: v.to(device=target_device, dtype=torch.long) if k == "input_ids" 
            else v.to(device=target_device)
            for k, v in text_inputs_two.items()
        }

        # Ensure models are on the correct device
        self.text_encoder_one = self.text_encoder_one.to(target_device)
        self.text_encoder_two = self.text_encoder_two.to(target_device)

        # Encoder forward pass
        # Use return_dict=True to reference outputs by name
        encoder_output_one = self.text_encoder_one(
            input_ids=text_inputs_one["input_ids"],
            attention_mask=text_inputs_one["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        encoder_output_two = self.text_encoder_two(
            input_ids=text_inputs_two["input_ids"],
            attention_mask=text_inputs_two["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )

        # For CLIP models, last_hidden_state is typically (batch_size, seq_length, hidden_dim).
        # Some CLIP variants may also return a pooled_output in a "pooler_output" or "text_embeds" field.
        # Adjust here if your model differs.
        if hasattr(encoder_output_one, "pooler_output"):
            pooled_prompt_embeds_one = encoder_output_one.pooler_output
        else:
            pooled_prompt_embeds_one = encoder_output_one.last_hidden_state[:, 0]

        if hasattr(encoder_output_two, "pooler_output"):
            pooled_prompt_embeds_two = encoder_output_two.pooler_output
        else:
            pooled_prompt_embeds_two = encoder_output_two.last_hidden_state[:, 0]

        # Fourth from last or second to last hidden state could vary – adapt as needed.
        # For demonstration, we take last_hidden_state directly:
        prompt_embeds_one = encoder_output_one.last_hidden_state
        prompt_embeds_two = encoder_output_two.last_hidden_state

        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1)

        # Duplicate for multiple images per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)

        # Move final embedding tensors to CPU (common for distribution)
        return {
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
        }

    def encode_prompt_batch(
        self,
        batch: Dict[str, Any],
        caption_column: str,
        proportion_empty_prompts: float = 0,
        is_train: bool = True,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of prompts and return embeddings.

        Args:
            batch: Input batch containing captions
            caption_column: Name of caption column in batch
            proportion_empty_prompts: Proportion of prompts to replace with empty string
            is_train: Whether in training mode (affects caption selection)
            device: Optional target device
            **kwargs: Additional arguments passed to __call__
        """
        if caption_column not in batch:
            raise ValueError(f"Batch missing required caption column '{caption_column}'.")

        prompt_batch = batch[caption_column]
        captions = []

        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # Take a random caption if there are multiple
                chosen_caption = (
                    random.choice(caption) if is_train else caption[0]
                )
                captions.append(chosen_caption)
            else:
                raise ValueError("Caption must be a string or a list/array of strings.")

        return self(captions, device=device, **kwargs)

    def _clean_caption(self, caption: str) -> str:
        """
        Clean/preprocess a caption string.
        Adjust logic as needed for specific clipping, punctuation removal, etc.
        """
        return caption.strip()
