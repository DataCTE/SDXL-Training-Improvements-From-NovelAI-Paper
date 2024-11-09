# SDXL Training with ZTSNR and NovelAI V3 Improvements

This repository implements the improvements described in "Improvements to SDXL in NovelAI Diffusion V3" (arXiv:2409.15997v2), including Zero Terminal SNR (ZTSNR), high-resolution coherence, and VAE improvements.

## weights

[ ] 10k dataset proof of concept
[ ] 200k+ dataset finetune

## Key Improvements

1. **Zero Terminal SNR (ZTSNR)**
   - Implements σ ≈ 20000 as practical approximation of infinity
   - Enables true black image generation
   - Prevents mean-color leakage from noise

2. **High-Resolution Coherence**
   - Enhanced noise schedule for better global coherence
   - Supports higher resolution generation without artifacts
   - Progressive noise level visualization

3. **VAE Improvements**
   - Scale-and-shift normalization
   - Per-channel statistics
   - Improved decoder finetuning

4. **Training Optimizations**
   - Tag-based loss weighting
   - Aspect ratio bucketing
   - Mixed precision (bfloat16)
   - Memory optimizations (xformers)

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
wandb
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
  --gradient_accumulation_steps 1 \
  --use_wandb \
  --wandb_project "sdxl-training" \
  --wandb_run_name "high-sigma-training"
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
--use_wandb                   : Enable Weights & Biases logging
--wandb_project               : W&B project name
--wandb_run_name              : W&B run name
--push_to_hub                 : Push model to HuggingFace Hub
--hub_model_id                : HuggingFace Hub model ID
--hub_private                 : Make the HuggingFace repo private
```

## Validation

The training includes automatic validation of:
1. ZTSNR effectiveness (black image generation)
2. High-resolution coherence
3. Noise schedule visualization
4. Progressive denoising steps

Results are saved as images and logged to W&B if enabled.

## Dataset Preparation

1. Place training images in data directory
2. Create matching .txt files with comma-separated tags
3. Optional: Enable tag-based loss weighting for balanced training

## Paper Implementation Details

This codebase follows the paper's specifications:
- σ ≈ 20000 for ZTSNR
- CFG scale range: 3.5-5.0 (optimal)
- bfloat16 precision with tf32 optimizations
- Aspect ratio bucketing for proper framing
- Tag-based loss weighting for balanced concept learning


## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

## License

Apache 2.0

## Citation

all credit to Juan Ossa, Eren Doğan, Alex Birch, and F. Johnson for the research.

```
bibtex
@article{ossa2024improvements,
title={Improvements to SDXL in NovelAI Diffusion V3},
author={Ossa, Juan and Doğan, Eren and Birch, Alex and Johnson, F.},
journal={arXiv preprint arXiv:2409.15997v2},
year={2024}
}
```