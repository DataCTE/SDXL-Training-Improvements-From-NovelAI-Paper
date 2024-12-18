import torch
import gc
from transformers import AutoTokenizer, PretrainedConfig
from typing import Dict, List, Union, Optional, Any
import random
import warnings
import logging
import torch.amp
import time
from weakref import WeakValueDictionary
from src.data.processors.utils.batch_utils import get_gpu_memory_usage
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
from src.config.config import TextEmbedderConfig


logger = logging.getLogger(__name__)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, subfolder: str = "text_encoder"):
    """Load the correct text encoder class based on config."""
    # Temporarily suppress the specific warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using a model of type clip_text_model")
        
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

def setup_memory_efficient_attention(model: torch.nn.Module) -> bool:
    """Enable memory efficient attention using xformers if available."""
    try:
        import xformers
        import xformers.ops
        
        # Check if model supports attention processor setting
        if hasattr(model, "set_attention_processor"):
            from diffusers.models.attention_processor import XFormersAttnProcessor
            model.set_attention_processor(XFormersAttnProcessor())
            logger.info("Enabled xformers attention processor")
            return True
            
        # Fallback to legacy method
        elif hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled legacy xformers attention")
            return True
            
        else:
            logger.warning("Model does not support xformers attention")
            return False
            
    except ImportError:
        logger.warning("xformers not available, using standard attention")
        return False
    except Exception as e:
        logger.warning(f"Error setting up xformers: {e}")
        return False

class TextEmbedder:
    def __init__(self, config: TextEmbedderConfig):
        """Initialize text embedder with both encoders."""
        self.config = config
        
        # Load tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            config.model_name,
            subfolder="tokenizer",
            use_fast=config.use_fast_tokenizer
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            config.model_name,
            subfolder="tokenizer_2",
            use_fast=config.use_fast_tokenizer
        )
        
        # Load text encoders
        text_encoder_cls_one = self._import_model_class_from_model_name(config.model_name, "text_encoder")
        text_encoder_cls_two = self._import_model_class_from_model_name(config.model_name, "text_encoder_2")
        
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            config.model_name,
            subfolder="text_encoder",
            torch_dtype=config.dtype
        )
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            config.model_name,
            subfolder="text_encoder_2",
            torch_dtype=config.dtype
        )
        
        # Move to device and eval mode
        self.text_encoder_one.to(config.device).eval()
        self.text_encoder_two.to(config.device).eval()
        
        logger.info(
            f"Initialized TextEmbedder:\n"
            f"- Model: {config.model_name}\n"
            f"- Device: {config.device}\n"
            f"- Dtype: {config.dtype}\n"
            f"- Max length: {config.max_length}"
        )

    @torch.no_grad()
    def __call__(self, prompts: List[str], proportion_empty_prompts: float = 0.0) -> Dict[str, torch.Tensor]:
        """Process text with both encoders."""
        try:
            # Handle empty prompt replacement
            captions = []
            for prompt in prompts:
                if random.random() < proportion_empty_prompts:
                    captions.append("")
                else:
                    captions.append(prompt)

            # Process with first encoder
            text_inputs_one = self.tokenizer_one(
                captions,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids_one = text_inputs_one.input_ids.to(self.config.device)
            
            prompt_embeds_one = self.text_encoder_one(
                text_input_ids_one,
                output_hidden_states=True,
                return_dict=False
            )
            
            # Process with second encoder
            text_inputs_two = self.tokenizer_two(
                captions,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids_two = text_inputs_two.input_ids.to(self.config.device)
            
            prompt_embeds_two = self.text_encoder_two(
                text_input_ids_two,
                output_hidden_states=True,
                return_dict=False
            )
            
            # Get pooled and prompt embeddings
            pooled_prompt_embeds_one = prompt_embeds_one[0]
            prompt_embeds_one = prompt_embeds_one[-1][-2]
            
            pooled_prompt_embeds_two = prompt_embeds_two[0]
            prompt_embeds_two = prompt_embeds_two[-1][-2]
            
            # Combine embeddings
            prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_one, pooled_prompt_embeds_two], dim=-1)
            
            return {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
            }
            
        except Exception as e:
            log_error_with_context(e, "Error in text embedding")
            return {
                "prompt_embeds": torch.empty(0, device=self.config.device),
                "pooled_prompt_embeds": torch.empty(0, device=self.config.device)
            }

    def _import_model_class_from_model_name(self, pretrained_model_name: str, subfolder: str = "text_encoder"):
        """Import the correct text encoder class."""
        config = PretrainedConfig.from_pretrained(pretrained_model_name, subfolder=subfolder)
        model_class = config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection
            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")