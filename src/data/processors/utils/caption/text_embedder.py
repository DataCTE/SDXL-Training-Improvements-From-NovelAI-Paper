import torch
from transformers import AutoTokenizer, PretrainedConfig
from typing import Dict, List, Union, Optional, Any
import random
import warnings
import logging
import torch.amp
import asyncio

# Internal imports from utils
from src.data.processors.utils.system_utils import (
    get_gpu_memory_usage,
    get_optimal_workers,
    create_thread_pool
)
from src.data.processors.utils.batch_utils import (
    process_in_chunks,
    calculate_optimal_batch_size,
    BatchConfig
)
from src.data.processors.utils.progress_utils import (
    create_progress_stats,
    update_progress_stats,
    log_progress
)

# Internal imports from processors
from src.data.processors.utils.thread_config import get_optimal_cpu_threads

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

def enable_clip_memory_efficient_attention(model: torch.nn.Module) -> bool:
    """Enable memory efficient attention for CLIP models if possible."""
    try:
        import xformers
        import xformers.ops
        
        def forward_memory_efficient(self, x, attention_mask=None):
            """Memory efficient attention forward pass."""
            h_ = self.heads
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            
            # Split heads
            q = q.view(q.shape[0], -1, h_, q.shape[-1] // h_).transpose(1, 2)
            k = k.view(k.shape[0], -1, h_, k.shape[-1] // h_).transpose(1, 2)
            v = v.view(v.shape[0], -1, h_, v.shape[-1] // h_).transpose(1, 2)
            
            # Create attention mask
            if attention_mask is not None:
                attention_mask = attention_mask.view(attention_mask.shape[0], 1, attention_mask.shape[1], 1)
                attention_mask = attention_mask.expand(-1, h_, -1, attention_mask.shape[-1])
                attention_mask = (attention_mask < 0.5)
            
            # Apply memory efficient attention
            out = xformers.ops.memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                p=0.0,
                scale=self.scale,
                mask=attention_mask
            )
            
            # Merge heads
            out = out.transpose(1, 2).contiguous()
            out = out.view(out.shape[0], -1, h_ * out.shape[-1])
            
            return self.to_out(out)
        
        # Find and patch attention layers
        found = False
        for name, module in model.named_modules():
            if "attn" in name.lower() and hasattr(module, "to_q"):
                module.forward = forward_memory_efficient.__get__(module)
                found = True
                
        if found:
            logger.info("Enabled xformers memory efficient attention for CLIP model")
            return True
        else:
            logger.warning("Could not find attention layers to optimize")
            return False
            
    except ImportError:
        logger.warning("xformers not available, using standard attention")
        return False

class TextEmbedder:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: torch.device,
        max_length: int = 77,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 32,
        enable_memory_efficient_attention: bool = True,
        max_memory_usage: float = 0.9
    ):
        """Initialize SDXL text embedder.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained SDXL model
            device: torch device
            max_length: max token length
            dtype: torch dtype for models
            batch_size: maximum batch size for text processing
            enable_memory_efficient_attention: whether to use memory efficient attention
            max_memory_usage: maximum GPU memory usage (0-1)
        """
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.max_memory_usage = max_memory_usage
        
        # Calculate optimal batch size
        self.batch_size = calculate_optimal_batch_size(
            device=device,
            min_batch_size=1,
            max_batch_size=batch_size,
            target_memory_usage=max_memory_usage,
            growth_factor=0.3  # Conservative growth for text embedding
        )
        
        # Load tokenizers with caching
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=True,
            model_max_length=max_length
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=True,
            model_max_length=max_length
        )

        # Load text encoders
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2"
        )
        
        # Load and optimize encoders
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)

        # Enable memory efficient attention if available
        if enable_memory_efficient_attention:
            enable_clip_memory_efficient_attention(self.text_encoder_one)
            enable_clip_memory_efficient_attention(self.text_encoder_two)

        # Freeze text encoders and set to eval mode
        self.text_encoder_one.eval()
        self.text_encoder_two.eval()
        for param in self.text_encoder_one.parameters():
            param.requires_grad = False
        for param in self.text_encoder_two.parameters():
            param.requires_grad = False

        # Set optimal CPU threads for tokenization
        torch.set_num_threads(get_optimal_cpu_threads().num_threads)
        
        # Pre-allocate reusable tensors
        self.attention_mask_buffer = torch.ones(
            (self.batch_size, max_length),
            dtype=torch.long,
            device=device
        )
        
        logger.info(
            f"Initialized TextEmbedder:\n"
            f"- Device: {device}\n"
            f"- Dtype: {dtype}\n"
            f"- Batch size: {self.batch_size}\n"
            f"- Max length: {max_length}\n"
            f"- Memory usage target: {max_memory_usage:.1%}"
        )

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            if self.device.type == "cuda":
                return calculate_optimal_batch_size(
                    device=self.device,
                    min_batch_size=1,
                    max_batch_size=32,  # Keep original cap
                    target_memory_usage=self.max_memory_usage,
                    growth_factor=0.3  # Conservative growth factor for text embedding
                )
            return 8  # Default CPU batch size
        except Exception as e:
            logger.warning(f"Error calculating batch size: {e}, using default")
            return 8

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
        batch_prompts = prompts[start_idx:end_idx]
        
        # Tokenize with padding
        text_inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device efficiently
        text_input_ids = text_inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = text_inputs.attention_mask.to(self.device, non_blocking=True)
        
        # Use new autocast syntax
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            prompt_embeds = text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=False
            )
        
        # Extract and reshape embeddings
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        
        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds
        }

    @torch.no_grad()
    def __call__(
        self, 
        prompt: Union[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Generate text embeddings for SDXL with optimized batch processing.
        
        Args:
            prompt: Text prompt or list of prompts
            proportion_empty_prompts: Proportion of prompts to make empty
            
        Returns:
            Dictionary containing text embeddings and pooled embeddings
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Handle empty prompts efficiently
        prompts = []
        for text in prompt:
            if random.random() < proportion_empty_prompts:
                prompts.append("")
            else:
                prompts.append(text)

        total_prompts = len(prompts)
        stats = create_progress_stats(total_prompts)
        all_prompt_embeds = []
        all_pooled_embeds = []
        
        # Process in optimized batches using process_in_chunks
        async def process_chunk(chunk_prompts: List[str], chunk_id: int) -> Dict[str, torch.Tensor]:
            start_idx = chunk_id * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_prompts)
            
            # Process with both encoders
            embeds_one = self._process_batch(chunk_prompts, self.tokenizer_one, self.text_encoder_one, start_idx, end_idx)
            embeds_two = self._process_batch(chunk_prompts, self.tokenizer_two, self.text_encoder_two, start_idx, end_idx)
            
            # Combine embeddings
            prompt_embeds = torch.cat([
                embeds_one['prompt_embeds'],
                embeds_two['prompt_embeds']
            ], dim=-1)
            
            pooled_prompt_embeds = embeds_two['pooled_prompt_embeds']
            
            # Update progress
            update_progress_stats(stats, len(chunk_prompts))
            if stats.should_log():
                log_progress(stats, prefix="Text Processing: ")
            
            return {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds
            }
        
        # Process all chunks
        chunk_results =  process_in_chunks(
            items=prompts,
            chunk_size=self.batch_size,
            process_fn=process_chunk
        )
        
        # Combine results
        for result in chunk_results:
            all_prompt_embeds.append(result['prompt_embeds'])
            all_pooled_embeds.append(result['pooled_prompt_embeds'])

        # Combine all batches
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(all_pooled_embeds, dim=0)
        
        # Keep on GPU if needed
        if self.device.type == "cuda":
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds
            }
        else:
            return {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
            }

    def encode_prompt_list(self, prompts: List[str], proportion_empty_prompts: float = 0.0):
        """Batch process a list of prompts efficiently."""
        return self(prompts, proportion_empty_prompts)

    async def process_text(self, text: str, tags: List[str]) -> Optional[Dict[str, Any]]:
        """Process text and tags asynchronously.
        
        Args:
            text: Raw text to process
            tags: List of tags
            
        Returns:
            Dictionary containing embeddings and pooled embeddings
        """
        try:
            # Process text through embedder
            embeddings = await asyncio.to_thread(
                self.__call__,
                text,
                proportion_empty_prompts=0.0
            )
            
            if embeddings is None:
                return None
                
            # Return combined data
            return {
                'embeds': embeddings['prompt_embeds'],
                'pooled_embeds': embeddings['pooled_prompt_embeds'],
                'tags': tags
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)[:200]}...")
            return None