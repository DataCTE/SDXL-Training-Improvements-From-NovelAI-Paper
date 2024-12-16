import torch
from transformers import AutoTokenizer, PretrainedConfig
from typing import Dict, List, Union, Optional
import random

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, subfolder: str = "text_encoder"):
    """Load the correct text encoder class based on config."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class TextEmbedder:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: torch.device,
        max_length: int = 77,
        dtype: torch.dtype = torch.float16
    ):
        """Initialize SDXL text embedder.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained SDXL model
            device: torch device
            max_length: max token length
            dtype: torch dtype for models
        """
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        
        # Load tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False
        )

        # Load text encoders
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2"
        )

        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder"
        ).to(device).to(dtype)
        
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2"
        ).to(device).to(dtype)

        # Freeze text encoders
        self.text_encoder_one.eval()
        self.text_encoder_two.eval()
        for param in self.text_encoder_one.parameters():
            param.requires_grad = False
        for param in self.text_encoder_two.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def __call__(
        self, 
        prompt: Union[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Generate text embeddings for SDXL.
        
        Args:
            prompt: Text prompt or list of prompts
            proportion_empty_prompts: Proportion of prompts to make empty
            
        Returns:
            Dictionary containing text embeddings and pooled embeddings
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Handle empty prompts for training
        prompts = []
        for text in prompt:
            if random.random() < proportion_empty_prompts:
                prompts.append("")
            else:
                prompts.append(text)

        prompt_embeds_list = []
        
        # Get embeddings from each text encoder
        for tokenizer, text_encoder in [(self.tokenizer_one, self.text_encoder_one), 
                                      (self.tokenizer_two, self.text_encoder_two)]:
            # Simply truncate without warnings
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            text_input_ids = text_inputs.input_ids.to(self.device)
            
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
                return_dict=False
            )
            
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
            
            last_pooled = pooled_prompt_embeds.view(bs_embed, -1)
            
        # Combine embeddings from both encoders
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        
        return {
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": last_pooled.cpu()
        }

    def encode_prompt_list(self, prompts: List[str], proportion_empty_prompts: float = 0.0):
        """Batch process a list of prompts."""
        return self(prompts, proportion_empty_prompts)