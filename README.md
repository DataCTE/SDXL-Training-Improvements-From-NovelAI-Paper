# SDXL Training with High Sigma and VAE Finetuning

This repository contains an implementation for training Stable Diffusion XL (SDXL) with high sigma values and optional VAE finetuning, incorporating advanced features like CLIP embeddings and aspect ratio bucketing.

## Features

- High sigma training with Zero Terminal SNR (ZTSNR) schedule
- VAE finetuning with perceptual loss
- CLIP embeddings for enhanced training
- Efficient caching system for latents and embeddings
- Mixed precision training (bfloat16)
- Memory optimizations (xformers, gradient checkpointing)
- EMA model averaging
- Aspect ratio bucketing
- Gradient accumulation support
- Optional model compilation with torch.compile
- Support for distributed training
- Adafactor optimizer option

## Prerequisites

Required packages:
```bash
torch 
torchvision 
diffusers 
transformers 
bitsandbytes 
tqdm 
Pillow
sentencepiece
accelerate
xformers
```

## Basic Usage

Basic training command:
```bash
python HighSigma.py \
  --model_path /path/to/sdxl/model \
  --data_dir /path/to/training/data \
  --output_dir ./output \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1
```

### Key Arguments

```
--model_path                    : Path to base SDXL model
--data_dir                     : Training data directory
--learning_rate                : Learning rate (default: 1e-6)
--num_epochs                   : Number of training epochs
--batch_size                   : Training batch size per GPU
--gradient_accumulation_steps  : Number of steps for gradient accumulation
--finetune_vae                : Enable VAE finetuning
--vae_learning_rate           : VAE learning rate when finetuning
--use_adafactor               : Use Adafactor optimizer instead of AdamW8bit
--enable_compile              : Enable torch.compile optimization
--compile_mode                : Torch compile mode (default/reduce-overhead/max-autotune)
--save_checkpoints           : Save checkpoints after each epoch
--cache_dir                  : Directory for caching latents and embeddings
```

## Data Preparation

1. Place training images in your data directory
2. Create matching .txt files with captions (same filename, .txt extension)
3. First run will cache:
   - VAE latents
   - Text embeddings (SDXL dual encoders)
   - CLIP embeddings
   - Tag embeddings

## Advanced Features

### Aspect Ratio Bucketing
The implementation includes automatic aspect ratio bucketing with the following ratios:
- 1:1 (Square)
- 4:3 (Landscape)
- 3:4 (Portrait)
- 16:9 (Widescreen)
- 9:16 (Tall)

### CLIP Integration
- Incorporates CLIP embeddings for both images and tags
- Uses CLIP-ViT-Large-Patch14 model
- Enhances training with visual-semantic alignment

### Perceptual Loss
- VGG16-based perceptual loss for VAE finetuning
- Multiple layer feature matching
- Configurable loss weights

### Memory Optimization
- Efficient caching system
- bfloat16 precision
- Gradient checkpointing
- xformers memory efficient attention
- Optional model compilation

## Training Monitoring

The training progress includes:
- Loss values
- Learning rate
- VAE loss (when enabled)
- Progress bar with epoch tracking
- Checkpoint saving (optional)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

## License

Apache 2.0

## Acknowledgments

This implementation is based on the research presented in **"Improvements to SDXL in NovelAI Diffusion V3"** by Juan Ossa, Eren Doğan, Alex Birch, and F. Johnson ([arXiv:2409.15997](https://arxiv.org/pdf/2409.15997)). The high sigma training approach and various optimizations are derived from their findings.
