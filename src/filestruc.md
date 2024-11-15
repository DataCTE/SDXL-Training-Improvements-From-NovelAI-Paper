# Project Structure Documentation

## Overview
This project implements a Stable Diffusion XL training pipeline with Zero Terminal SNR (ZTSNR), high-resolution coherence enhancements, and VAE improvements from NovelAI research. The codebase is organized for maximum efficiency, modularity, and extensibility.

## Directory Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Advanced dataset with caching and dynamic resolution
│   ├── tag_weighter.py      # CLIP-based dynamic tag weighting
│   ├── ultimate_upscaler.py # High-resolution upscaling with coherence
│   ├── usdu_patch.py        # Ultimate SD Upscaler integration
│   └── utils.py            # Data processing and augmentation utilities
│
├── inference/
│   ├── __init__.py
│   ├── text_to_image.py    # Optimized inference pipeline
│   ├── img2img.py          # Image-to-image pipeline
│   └── Comfyui-zsnmode/    # ComfyUI integration nodes
│
├── models/
│   ├── __init__.py
│   ├── architecture.py     # Model architecture definitions
│   ├── attention.py        # Memory-efficient attention implementations
│   └── vae.py             # Enhanced VAE components
│
├── training/
│   ├── __init__.py
│   ├── ema.py             # Enhanced EMA with momentum scheduling
│   ├── loss.py            # ZTSNR and perceptual loss implementations
│   ├── trainer.py         # Memory-efficient training loop
│   └── vae_finetuner.py   # Advanced VAE fine-tuning
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py      # Robust model state management
│   ├── device.py         # Memory and device optimization
│   ├── hub.py            # HuggingFace integration
│   ├── logging.py        # Comprehensive training monitoring
│   ├── model_card.py     # Automated documentation
│   ├── setup.py          # Advanced configuration
│   └── validation.py     # Extensive model validation
│
└── main.py               # Entry point with CLI
```

## Component Details

### Data Module
- `dataset.py`: Advanced dataset implementation
  - Multi-resolution image handling
  - Efficient latent caching system
  - Dynamic batch composition
  - Automatic augmentation pipeline
  - Memory-mapped storage
  - Streaming capabilities
  - Multi-worker data loading

- `tag_weighter.py`: Dynamic tag weighting
  - CLIP embedding analysis
  - Frequency-based normalization
  - Category-specific weighting
  - Tag correlation analysis
  - Dropout regularization
  - Cache management
  - Real-time weight updates

### Models Module
- `architecture.py`: Enhanced SDXL architecture
  - Memory-efficient attention
  - Gradient checkpointing
  - Custom scaling rules
  - Dynamic resolution handling
  - Feature pyramid networks
  - Skip connections

- `attention.py`: Optimized attention mechanisms
  - Sparse attention patterns
  - Memory-efficient implementation
  - Dynamic head pruning
  - Cross-frame attention
  - Flash attention support
  - Xformers integration

### Training Module
- `loss.py`: Comprehensive loss system
  - ZTSNR implementation
  - Perceptual loss components
  - Dynamic weighting
  - Gradient scaling
  - Feature matching
  - Style transfer loss

- `trainer.py`: Advanced training loop
  - Gradient accumulation
  - Mixed precision training
  - Dynamic batch sizing
  - Memory optimization
  - Distributed training
  - Checkpoint management
  - Progress tracking

### Utils Module
- `checkpoint.py`: Robust state management
  - Safe state saving/loading
  - Incremental updates
  - Emergency recovery
  - Version control
  - Metadata tracking

- `device.py`: Resource optimization
  - Memory tracking
  - CUDA optimization
  - Cache management
  - Automatic mixed precision
  - Memory defragmentation
  - Device assignment

## Implementation Notes

### Memory Management
- Automatic garbage collection
- Gradient checkpointing
- Efficient attention
- Dynamic batching
- Cache optimization
- Memory-mapped data
- Streaming processing

### Training Features
- Multi-resolution support
- Dynamic tag weighting
- Adaptive learning rates
- Progressive training
- Validation pipeline
- Automated logging
- Emergency recovery

### Performance Optimizations
- Mixed precision training
- Memory-efficient attention
- Gradient accumulation
- Efficient data loading
- Caching strategies
- Device optimization

## Usage Guidelines

Refer to main README.md for:
- Detailed installation steps
- Training configurations
- Performance optimization
- Troubleshooting guide
- Best practices