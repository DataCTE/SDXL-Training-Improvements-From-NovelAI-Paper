import torch
from typing import Dict
from transformers import CLIPTokenizer, CLIPTextModel

class TextEmbedder:
    def __init__(
        self,
        device: torch.device,
        tokenizer_paths: Dict[str, str] = {
            "base": "openai/clip-vit-large-patch14",
            "large": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        }
    ):
        self.device = device
        self.max_length = 77
        self.cache = {}
        
        # Enable TF32 for text encoder
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            
        # Load tokenizers with optimized settings
        self.tokenizers = {
            model_type: self._load_tokenizer(path)
            for model_type, path in tokenizer_paths.items()
        }
        
        # Load text encoders with optimized settings
        self.text_encoders = {
            model_type: self._load_text_encoder(path)
            for model_type, path in tokenizer_paths.items()
        }
        
        # Pre-allocate buffers for common operations
        self._setup_buffers()
        
    def _load_tokenizer(self, path: str) -> CLIPTokenizer:
        """Load tokenizer with optimized settings"""
        return CLIPTokenizer.from_pretrained(
            path,
            use_fast=True,  # Use fast tokenizer
            model_max_length=self.max_length,
            padding_side='right'
        )
        
    def _load_text_encoder(self, path: str) -> CLIPTextModel:
        """Load text encoder with optimized settings"""
        encoder = CLIPTextModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_auth_token=False
        ).to(self.device)
        
        # Enable memory optimizations
        encoder.gradient_checkpointing_enable()
        
        # Try to enable memory efficient attention if available
        if hasattr(encoder, 'enable_xformers_memory_efficient_attention'):
            encoder.enable_xformers_memory_efficient_attention()
        
        # Freeze encoder
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
            
        return encoder
        
    def _setup_buffers(self):
        """Pre-allocate buffers for common operations"""
        self.empty_text_embeds = {
            model_type: torch.zeros(
                (1, self.max_length, encoder.config.hidden_size),
                dtype=torch.bfloat16,
                device=self.device
            )
            for model_type, encoder in self.text_encoders.items()
        }
        
        self.empty_pooled_embeds = {
            model_type: torch.zeros(
                (1, encoder.config.hidden_size),
                dtype=torch.bfloat16,
                device=self.device
            )
            for model_type, encoder in self.text_encoders.items()
        }
        
    @torch.no_grad()
    def __call__(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Get embeddings with caching and optimized computation"""
        # Check cache first
        if prompt in self.cache:
            return self.cache[prompt]
            
        # Handle empty prompts
        if not prompt:
            empty_embeds = {}
            for model_type in self.tokenizers.keys():
                empty_embeds[f"{model_type}_text_embeds"] = self.empty_text_embeds[model_type].clone()
                empty_embeds[f"{model_type}_pooled_embeds"] = self.empty_pooled_embeds[model_type].clone()
            return empty_embeds
            
        # Process with each model type
        embeds = {}
        for model_type, tokenizer in self.tokenizers.items():
            # Tokenize efficiently
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                encoder = self.text_encoders[model_type]
                output = encoder(**tokens)
                
                # Ensure correct shapes
                text_embeds = output.last_hidden_state
                if text_embeds.dim() == 2:
                    text_embeds = text_embeds.unsqueeze(0)
                    
                pooled_embeds = output.pooler_output
                if pooled_embeds.dim() == 1:
                    pooled_embeds = pooled_embeds.unsqueeze(0)
                elif pooled_embeds.dim() == 3:
                    pooled_embeds = pooled_embeds.squeeze(1)
                    
                # Store embeddings
                embeds[f"{model_type}_text_embeds"] = text_embeds
                embeds[f"{model_type}_pooled_embeds"] = pooled_embeds
                
        # Cache results
        self.cache[prompt] = embeds
        return embeds
        
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        torch.cuda.empty_cache()