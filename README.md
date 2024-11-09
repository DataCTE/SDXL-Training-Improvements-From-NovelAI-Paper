# SDXL Training with High Sigma and VAE Finetuning

This repository contains an experimental implementation for training Stable Diffusion XL (SDXL) with high sigma values and optional VAE finetuning. Please note that this is a work in progress and requires further testing and optimization.

## ⚠️ Current Status

This code is currently in an experimental state and needs several improvements:

- Memory management needs optimization
- Batch processing requires refinement
- VAE finetuning implementation needs validation
- Training stability needs improvement
- Documentation requires expansion

## Features

Current implementation includes:
- High sigma training based on improved noise schedules
- Optional VAE finetuning with perceptual loss
- Text embedding caching
- Mixed precision training (bfloat16)
- Memory optimizations (xformers, gradient checkpointing)
- EMA model averaging
- Aspect ratio bucketing (preliminary implementation)

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
  --num_epochs 1
```

### Key Arguments

```
--model_path          : Path to base SDXL model
--data_dir           : Training data directory
--learning_rate      : Learning rate (default: 1e-6)
--num_epochs         : Number of training epochs
--finetune_vae      : Enable VAE finetuning
--vae_learning_rate : VAE learning rate when finetuning
--use_adafactor     : Use Adafactor optimizer instead of AdamW8bit
```

## Data Preparation

1. Place training images in your data directory
2. Create matching .txt files with captions (same filename, .txt extension)
3. First run will cache latents and embeddings

## Known Issues

1. Memory Usage:
   - Current implementation may be memory intensive
   - Batch sizes may need adjustment based on GPU memory

2. Performance:
   - Training speed needs optimization
   - Gradient accumulation might need tuning

3. VAE Finetuning:
   - Experimental feature that needs validation
   - May require additional memory management

## TODO

- [ ] Optimize memory usage
- [ ] Improve batch processing
- [ ] Validate and improve VAE finetuning
- [ ] Add proper validation steps
- [ ] Expand documentation
- [ ] Add training monitoring
- [ ] Implement proper error handling

## Contributing

Given the experimental nature of this code, contributions and improvements are welcome. Please open issues or pull requests for any enhancements.

## License

Apache 2.0

## Disclaimer

This is experimental code that needs further development and testing. Use at your own risk and expect potential issues that will need addressing.
