"""Utilities for prompt processing and embedding generation."""

import torch
import re


def process_prompt(prompt, max_length=77):
    """
    Process prompt with NovelAI improvements.
    
    Args:
        prompt (str): Input prompt
        max_length (int): Maximum prompt length
        
    Returns:
        str: Processed prompt
    """
    # Remove excessive whitespace
    prompt = re.sub(r'\s+', ' ', prompt.strip())
    
    # Truncate if too long
    if len(prompt.split()) > max_length:
        prompt = ' '.join(prompt.split()[:max_length])
    
    return prompt


def get_prompt_embeds(prompt, tokenizer, text_encoder, device, dtype):
    """
    Get text embeddings with NovelAI improvements.
    
    Args:
        prompt (str): Input prompt
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        torch.Tensor: Text embeddings
    """
    # Tokenize text
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Get text embeddings
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )
        # Use penultimate hidden state as per NovelAI
        prompt_embeds = prompt_embeds.hidden_states[-2].to(dtype)
    
    return prompt_embeds
