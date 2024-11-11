import torch

def encode_prompts(prompts, tokenizer=None, text_encoder=None, max_length=77):
    """
    Encode text prompts to embeddings
    
    Args:
        prompts: List of text prompts
        tokenizer: Tokenizer model
        text_encoder: Text encoder model
        max_length: Maximum sequence length
        
    Returns:
        Text embeddings tensor [B, 77, 768/1024]
    """
    if tokenizer is None or text_encoder is None:
        return prompts
        
    # Tokenize prompts
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(text_encoder.device),
            attention_mask=text_inputs.attention_mask.to(text_encoder.device)
        )[0]
    
    return text_embeddings

def get_prompt_embeddings(prompt, neg_prompt, tokenizer, text_encoder, max_length=77):
    """
    Get combined prompt embeddings with classifier-free guidance
    
    Args:
        prompt: Positive text prompt
        neg_prompt: Negative text prompt
        tokenizer: Tokenizer model  
        text_encoder: Text encoder model
        max_length: Maximum sequence length
        
    Returns:
        Combined text embeddings tensor [B, 77, 768/1024]
    """
    # Encode positive and negative prompts
    pos_embeddings = encode_prompts([prompt], tokenizer, text_encoder, max_length)
    neg_embeddings = encode_prompts([neg_prompt], tokenizer, text_encoder, max_length)
    
    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([neg_embeddings, pos_embeddings])
    
    return text_embeddings 