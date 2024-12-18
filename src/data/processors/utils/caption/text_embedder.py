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
    def __init__(
        self,
        config: TextEmbedderConfig
    ):
        """Initialize SDXL text embedder with consolidated config."""
        try:
            self.config = config
            
            # Log initialization start
            logger.info("Initializing text embedder...")
            
            # Initialize models with progress logging
            logger.info("Loading text encoder one...")
            self.text_encoder_one = self._load_text_encoder("one")
            
            logger.info("Loading text encoder two...")
            self.text_encoder_two = self._load_text_encoder("two")
            
            logger.info("Loading tokenizers...")
            self.tokenizer_one = self._load_tokenizer("one")
            self.tokenizer_two = self._load_tokenizer("two")
            
            # Initialize tensor cache
            self._tensor_cache = WeakValueDictionary()
            
            # Enable optimizations
            if self.config.enable_memory_efficient_attention:
                setup_memory_efficient_attention(self.text_encoder_one)
                setup_memory_efficient_attention(self.text_encoder_two)
            
            # Log final configuration
            logger.info(
                f"Text embedder initialized:\n"
                f"- Device: {config.device}\n"
                f"- Dtype: {config.dtype}\n"
                f"- Batch size: {config.batch_size}\n"
                f"- Max length: {config.max_length}\n"
                f"- Memory efficient attention: {config.enable_memory_efficient_attention}"
            )
            log_system_metrics(prefix="Text embedder initialization: ")
            
        except Exception as e:
            log_error_with_context(e, "Error initializing text embedder")
            raise

    def __del__(self):
        """Cleanup when embedder is deleted."""
        self.cleanup()

    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=self.config.device)
            self._tensor_cache[key] = tensor
        return tensor

    @torch.no_grad()
    def _process_batch(
        self,
        prompts: List[str],
        tokenizer,
        text_encoder,
        start_idx: int,
        end_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Process a batch of prompts efficiently."""
        try:
            # Tokenize with padding
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=self.config.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Move to device efficiently
            text_input_ids = text_inputs.input_ids.to(self.config.device, non_blocking=True)
            attention_mask = text_inputs.attention_mask.to(self.config.device, non_blocking=True)
            
            # Use new autocast syntax
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                prompt_embeds = text_encoder(
                    text_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=False
                )
            
            # Extract and reshape embeddings
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            
            # Clean up intermediate tensors
            del text_inputs
            del text_input_ids
            del attention_mask
            torch.cuda.empty_cache()
            
            return {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {start_idx}:{end_idx}: {e}")
            # Clean up on error
            if 'text_inputs' in locals():
                del text_inputs
            if 'text_input_ids' in locals():
                del text_input_ids
            if 'attention_mask' in locals():
                del attention_mask
            if 'prompt_embeds' in locals():
                del prompt_embeds
            torch.cuda.empty_cache()
            raise

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Process text with detailed metrics tracking."""
        try:
            batch_stats = {
                'batch_size': len(prompts),
                'start_memory': get_gpu_memory_usage(self.config.device),
                'start_time': time.time()
            }
            
            # Process text through both encoders
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                # Process with first encoder
                text_embeddings_1 = self._process_batch(
                    prompts,
                    self.tokenizer_one,
                    self.text_encoder_one,
                    0,
                    len(prompts)
                )
                
                # Process with second encoder
                text_embeddings_2 = self._process_batch(
                    prompts,
                    self.tokenizer_two,
                    self.text_encoder_two,
                    0,
                    len(prompts)
                )
                
                # Combine embeddings
                text_embeddings = torch.cat([
                    text_embeddings_1['prompt_embeds'],
                    text_embeddings_2['prompt_embeds']
                ], dim=-1)
                
                pooled_embeddings = torch.cat([
                    text_embeddings_1['pooled_prompt_embeds'],
                    text_embeddings_2['pooled_prompt_embeds']
                ], dim=-1)
            
            # Update stats
            batch_stats.update({
                'end_memory': get_gpu_memory_usage(self.config.device),
                'duration': time.time() - batch_stats['start_time'],
                'memory_change': (
                    get_gpu_memory_usage(self.config.device) - 
                    batch_stats['start_memory']
                ),
                'avg_token_length': sum(len(p.split()) for p in prompts) / len(prompts)
            })
            
            # Log metrics
            log_metrics(
                metrics=batch_stats, 
                step=batch_stats['batch_size'], 
                step_type="text_embed",
                is_main_process=True,
                use_wandb=True
            )
            
            return {
                "prompt_embeds": text_embeddings,
                "pooled_prompt_embeds": pooled_embeddings
            }
            
        except Exception as e:
            log_error_with_context(e, "Error in text embedding")
            return {
                "prompt_embeds": torch.empty(0, device=self.config.device),
                "pooled_prompt_embeds": torch.empty(0, device=self.config.device)
            }

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear tensor cache
            if hasattr(self, '_tensor_cache'):
                self._tensor_cache.clear()
            
            # Clear CUDA cache if using GPU
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Clear model references
            if hasattr(self, 'text_encoder_one'):
                del self.text_encoder_one
            if hasattr(self, 'text_encoder_two'):
                del self.text_encoder_two
            if hasattr(self, 'tokenizer_one'):
                del self.tokenizer_one
            if hasattr(self, 'tokenizer_two'):
                del self.tokenizer_two
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Successfully cleaned up text embedder resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Try one last time to clear memory
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass

    async def process_text(
        self, 
        text: str, 
        tags: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Process text and tags asynchronously."""
        try:
            # Generate embeddings for the text
            embeddings = self.__call__([text], proportion_empty_prompts=0.0)
            
            if not isinstance(embeddings, dict) or 'prompt_embeds' not in embeddings:
                logger.error("Failed to generate embeddings")
                return None
                
            # Return combined data
            return {
                'embeds': embeddings['prompt_embeds'][0],
                'pooled_embeds': embeddings['pooled_prompt_embeds'][0],
                'tags': tags
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)[:200]}...")
            return None

    def _load_text_encoder(self, encoder_type: str):
        """Load text encoder with proper error handling."""
        try:
            # Determine model path and config
            if encoder_type == "one":
                subfolder = self.config.text_encoder_subfolder
            elif encoder_type == "two":
                subfolder = self.config.text_encoder_2_subfolder
            else:
                raise ValueError(f"Invalid encoder type: {encoder_type}")
            
            # Import correct model class
            model_class = import_model_class_from_model_name_or_path(
                self.config.model_name,
                subfolder=subfolder
            )
            
            # Load model with optimizations
            model = model_class.from_pretrained(
                self.config.model_name,
                subfolder=subfolder,
                torch_dtype=self.config.dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                device_map=None  # We'll move to device manually
            )
            
            # Move to device and optimize
            model = model.to(self.config.device)
            model.eval()
            
            if self.config.enable_memory_efficient_attention:
                setup_memory_efficient_attention(model)
            
            logger.info(f"Successfully loaded text encoder {encoder_type}")
            return model
            
        except Exception as e:
            log_error_with_context(
                e, 
                f"Error loading text encoder {encoder_type}"
            )
            raise

    def _load_tokenizer(self, tokenizer_type: str):
        """Load tokenizer with proper error handling."""
        try:
            # Determine tokenizer path
            if tokenizer_type == "one":
                subfolder = self.config.tokenizer_subfolder
            elif tokenizer_type == "two":
                subfolder = self.config.tokenizer_2_subfolder
            else:
                raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                subfolder=subfolder,
                use_fast=self.config.use_fast_tokenizer
            )
            
            logger.info(f"Successfully loaded tokenizer {tokenizer_type}")
            return tokenizer
            
        except Exception as e:
            log_error_with_context(
                e,
                f"Error loading tokenizer {tokenizer_type}"
            )
            raise