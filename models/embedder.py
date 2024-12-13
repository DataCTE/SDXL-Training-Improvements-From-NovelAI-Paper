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
        
        # Load tokenizers (remove subfolder paths)
        self.tokenizers = {
            "base": CLIPTokenizer.from_pretrained(tokenizer_paths["base"]),
            "large": CLIPTokenizer.from_pretrained(tokenizer_paths["large"])
        }
        
        # Load text encoders with bfloat16 (remove subfolder paths)
        self.text_encoders = {
            "base": CLIPTextModel.from_pretrained(tokenizer_paths["base"]).to(device).to(torch.bfloat16),
            "large": CLIPTextModel.from_pretrained(tokenizer_paths["large"]).to(device).to(torch.bfloat16)
        }
        
        for encoder in self.text_encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def __call__(self, prompt: str) -> Dict[str, torch.Tensor]:
        # Tokenize
        tokens = {
            k: tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            for k, tokenizer in self.tokenizers.items()
        }
        
        # Generate embeddings
        embeds = {}
        for k, encoder in self.text_encoders.items():
            # Move tokens to GPU but keep as int64 for embedding layer
            tokens_gpu = {key: val.to(self.device) for key, val in tokens[k].items()}
            
            # Generate embeddings
            output = encoder(**tokens_gpu)
            
            # Ensure text embeddings have shape [batch_size, seq_len, hidden_dim]
            last_hidden_state = output.last_hidden_state
            if last_hidden_state.dim() == 2:
                last_hidden_state = last_hidden_state.unsqueeze(0)
            
            # Ensure pooled embeddings have shape [batch_size, hidden_dim]
            pooled_output = output.pooler_output
            if pooled_output.dim() == 1:
                pooled_output = pooled_output.unsqueeze(0)
            elif pooled_output.dim() == 3:
                pooled_output = pooled_output.squeeze(1)
            
            # Keep embeddings on CPU with consistent dimensions
            embeds[f"{k}_text_embeds"] = last_hidden_state.cpu()  # [batch_size, seq_len, hidden_dim]
            embeds[f"{k}_pooled_embeds"] = pooled_output.cpu()   # [batch_size, hidden_dim]
            
        return embeds