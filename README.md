# SDXL Training with ZTSNR and NovelAI V3 Improvements

Most SDXL implementations blindly port the 14% maximum allowed deviation value from SD1.5/1.4, ignoring that larger-scale diffusion models require a wider deviation range to reach their full potential. Research from NovelAI (arXiv:2409.15997v2) demonstrates this significantly impacts color accuracy and composition. This along with including Zero Terminal SNR (ZTSNR) and VAE finetuning, are implemented in this repository.

### Weights

- [ ] 10k dataset proof of concept (currently training/ in testing)
- [ ] 200k+ dataset finetune (coming soon)

## Key Technical Improvements

1. **Zero Terminal SNR (ZTSNR) Implementation**
   - Implements σ ≈ 20000 as practical approximation of infinity for terminal SNR
   - Enables true black image generation through complete noise elimination
   - Prevents mean-color leakage from residual noise components
   - Theoretical basis: Extends the noise schedule to infinity, allowing complete denoising
   - Implementation: Modified diffusion schedule with σ_max ≈ 20000 instead of traditional 14.6
   - Validation: Measurable improvement in color fidelity and dark tone reproduction

2. **High-Resolution Coherence Enhancement**
   - Enhanced noise scheduling for improved global composition
   - Technical approach:
     * Progressive noise level reduction with optimized step sizes
     * Adaptive sigma scheduling based on image resolution
     * Improved attention mechanisms for large-scale features
   - Measurable improvements:
     * Reduced artifacting in complex scenes
     * Better preservation of global composition
     * Enhanced detail consistency across different scales

3. **VAE Architecture Improvements**
   - Scale-and-shift normalization implementation:
     * Per-channel statistics computation
     * Adaptive normalization based on latent statistics
     * Improved handling of extreme values
   - Technical modifications:
     * Enhanced decoder architecture with improved upsampling
     * Modified attention mechanisms for better feature preservation
     * Optimized bottleneck processing
   - Validation metrics:
     * Reduced reconstruction loss
     * Improved color space preservation
     * Better handling of extreme values in latent space

4. **Training Pipeline Optimizations**
   - Memory-efficient implementation:
     * xformers attention optimization
     * Gradient checkpointing with optimal tradeoffs
     * Mixed precision training (bfloat16) with TF32 optimizations
   - Data processing improvements:
     * Aspect ratio bucketing with dynamic batch assembly
     * Efficient latent caching system
     * Optimized data loading pipeline
   - Loss function enhancements:
     * Tag-based weighted loss computation
     * Adaptive loss scaling based on image complexity
     * Enhanced gradient flow through improved architecture

5. **Validation and Metrics**
   - Comprehensive validation suite:
     * ZTSNR effectiveness measurement
     * High-resolution coherence testing
     * Progressive denoising visualization
   - Quantitative metrics:
     * FID score tracking
     * CLIP score evaluation
     * Color accuracy measurements
   - Qualitative assessments:
     * Visual fidelity comparisons
     * Artifact analysis
     * Global composition evaluation


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