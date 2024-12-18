import torch
import gc
from transformers import AutoTokenizer, PretrainedConfig
from typing import Dict, List, Union, Optional, Any
import random
import warnings
import logging
import torch.amp
import asyncio
from weakref import WeakValueDictionary
from src.data.processors.utils.batch_utils import (
    process_in_chunks,
    calculate_optimal_batch_size,
)
from src.data.processors.utils.progress_utils import (
    create_progress_tracker,
    update_tracker,
    log_progress,
    ProgressStats
)

# Internal imports from processors
from src.data.processors.utils.thread_config import get_optimal_cpu_threads
from src.config.config import TextEmbedderConfig, BatchProcessorConfig

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
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
            return True
        else:
            logger.warning("Model does not support xformers attention")
            return False
    except ImportError:
        logger.warning("xformers not available, using standard attention")
        return False

class TextEmbedder:
    def __init__(
        self,
        config: TextEmbedderConfig
    ):
        """Initialize SDXL text embedder with consolidated config."""
        self.config = config
        
        # Create a temporary BatchProcessorConfig to pass into calculate_optimal_batch_size
        temp_config = BatchProcessorConfig(
            device=config.device,
            batch_size=config.batch_size,
            min_batch_size=1,                # Minimal safe batch size
            max_batch_size=config.batch_size,
            memory_growth_factor=config.growth_factor,
            max_memory_usage=config.max_memory_usage,
            num_workers=16,                  # Or any reasonable default
            prefetch_factor=2                # Or any reasonable default
        )
        self.batch_size = calculate_optimal_batch_size(temp_config)
        
        # Initialize tensor cache
        self._tensor_cache = WeakValueDictionary()
        
        try:
            # Load tokenizers with caching
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                config.model_name,
                subfolder=config.tokenizer_subfolder,
                use_fast=config.use_fast_tokenizer,
                model_max_length=config.max_length
            )
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                config.model_name,
                subfolder=config.tokenizer_2_subfolder,
                use_fast=config.use_fast_tokenizer,
                model_max_length=config.max_length
            )

            # Load text encoders
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                config.model_name,
                subfolder=config.text_encoder_subfolder
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                config.model_name,
                subfolder=config.text_encoder_2_subfolder
            )
            
            # Load and optimize encoders
            self.text_encoder_one = text_encoder_cls_one.from_pretrained(
                config.model_name,
                subfolder=config.text_encoder_subfolder,
                torch_dtype=config.dtype,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            ).to(config.device)
            
            self.text_encoder_two = text_encoder_cls_two.from_pretrained(
                config.model_name,
                subfolder=config.text_encoder_2_subfolder,
                torch_dtype=config.dtype,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            ).to(config.device)

            # Enable memory efficient attention if available
            if config.enable_memory_efficient_attention:
                setup_memory_efficient_attention(self.text_encoder_one)
                setup_memory_efficient_attention(self.text_encoder_two)

            # Freeze text encoders and set to eval mode
            self.text_encoder_one.eval()
            self.text_encoder_two.eval()
            for param in self.text_encoder_one.parameters():
                param.requires_grad = False
            for param in self.text_encoder_two.parameters():
                param.requires_grad = False

            # Set optimal CPU threads for tokenization
            torch.set_num_threads(get_optimal_cpu_threads().num_threads)
            
            logger.info(
                f"Initialized TextEmbedder:\n"
                f"- Device: {config.device}\n"
                f"- Dtype: {config.dtype}\n"
                f"- Batch size: {self.batch_size}\n"
                f"- Max length: {config.max_length}\n"
                f"- Memory usage target: {config.max_memory_usage:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing TextEmbedder: {e}")
            self.cleanup()
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
        prompt: Union[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Generate text embeddings for SDXL with optimized batch processing."""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        try:
            # Handle empty prompts efficiently
            prompts = []
            for text in prompt:
                if random.random() < proportion_empty_prompts:
                    prompts.append("")
                else:
                    prompts.append(text)

            total_prompts = len(prompts)
            stats = create_progress_tracker(total_prompts)
            all_prompt_embeds = []
            all_pooled_embeds = []
            
            # Process in batches
            for i in range(0, total_prompts, self.batch_size):
                batch_end = min(i + self.batch_size, total_prompts)
                batch_prompts = prompts[i:batch_end]
                
                try:
                    # Process with both encoders
                    embeds_one = self._process_batch(
                        batch_prompts, 
                        self.tokenizer_one, 
                        self.text_encoder_one,
                        i,
                        batch_end
                    )
                    embeds_two = self._process_batch(
                        batch_prompts,
                        self.tokenizer_two,
                        self.text_encoder_two,
                        i,
                        batch_end
                    )
                    
                    # Combine embeddings
                    prompt_embeds = torch.cat([
                        embeds_one['prompt_embeds'],
                        embeds_two['prompt_embeds']
                    ], dim=-1)
                    
                    pooled_prompt_embeds = embeds_two['pooled_prompt_embeds']
                    
                    # Collect results
                    all_prompt_embeds.append(prompt_embeds.cpu())
                    all_pooled_embeds.append(pooled_prompt_embeds.cpu())
                    
                    # Clean up batch tensors
                    del embeds_one
                    del embeds_two
                    del prompt_embeds
                    del pooled_prompt_embeds
                    
                    # Update progress
                    update_tracker(stats, len(batch_prompts))
                    if stats.should_log():
                        log_progress(stats, prefix="Text Processing: ")
                        
                    # Periodic cleanup
                    if i % (self.batch_size * 10) == 0:  # Every 10 batches
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    continue

            try:
                # Combine all batches
                prompt_embeds = torch.cat(all_prompt_embeds, dim=0).to(self.config.device)
                pooled_prompt_embeds = torch.cat(all_pooled_embeds, dim=0).to(self.config.device)
                
                # Clean up intermediate lists
                del all_prompt_embeds
                del all_pooled_embeds
                gc.collect()
                
                return {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds
                }
                
            except Exception as e:
                logger.error(f"Error combining batches: {str(e)}")
                return {
                    "prompt_embeds": torch.empty(0, device=self.config.device),
                    "pooled_prompt_embeds": torch.empty(0, device=self.config.device)
                }
                
        except Exception as e:
            logger.error(f"Error in text embedding: {str(e)}")
            return {
                "prompt_embeds": torch.empty(0, device=self.config.device),
                "pooled_prompt_embeds": torch.empty(0, device=self.config.device)
            }
            
        finally:
            # Final cleanup
            gc.collect()
            torch.cuda.empty_cache()

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