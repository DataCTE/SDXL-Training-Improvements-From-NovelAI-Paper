# Project Structure Documentation

## Overview
This project implements a Stable Diffusion XL training pipeline with Zero Terminal SNR (ZTSNR), high-resolution coherence enhancements, and VAE improvements from NovelAI research.

## Directory Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Custom dataset with efficient caching
│   ├── tag_weighter.py      # Tag weighting implementation
│   ├── ultimate_upscaler.py # High-resolution upscaling
│   ├── usdu_patch.py       # USDU patching utilities
│   └── utils.py            # Data processing utilities
│
├── inference/
│   ├── __init__.py
│   ├── text_to_image.py    # Inference pipeline
│   └── Comfyui-zsnmode/    # ComfyUI integration
│
├── models/
│   └── __init__.py         # Model definitions
│
├── training/
│   ├── __init__.py
│   ├── _pycache_/
│   ├── ema.py             # Exponential Moving Average
│   ├── loss.py            # ZTSNR loss implementation
│   ├── trainer.py         # Main training loop
│   └── vae_finetuner.py   # Adaptive VAE fine-tuning
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py      # Model state management
│   ├── device.py         # Memory optimization
│   ├── hub.py            # Model publishing
│   ├── logging.py        # Training monitoring
│   ├── model_card.py     # Documentation
│   ├── setup.py          # Configuration
│   └── validation.py     # Model validation
│
└── main.py               # Entry point
```

## Component Details

### Data Module
- `dataset.py`: Advanced dataset implementation
  - Efficient latent caching system
  - Resolution-adaptive bucketing
  - CLIP embedding preprocessing
  - Memory-efficient batch processing
  - Dynamic tag processing
  - Automatic statistics tracking

### Models Module
- `model_validator.py`: ZTSNR validation system
  - Black image generation testing
  - High-resolution coherence validation
  - Noise schedule visualization
  - Progressive denoising validation
  - Statistical metrics tracking

- `setup.py`: Enhanced model initialization
  - High-sigma noise scheduling
  - Cross-attention optimization
  - Resolution-adaptive configurations
  - Memory-efficient model loading

- `tag_weighter.py`: Advanced loss weighting
  - CLIP-based tag analysis
  - Dynamic weight adjustment
  - Category-based weighting
  - Frequency normalization
  - Cached embedding system

- `vae_finetuner.py`: Improved VAE training
  - Welford algorithm implementation
  - Adaptive statistics normalization
  - Progressive batch processing
  - Memory-efficient attention
  - Real-time stability monitoring

### Training Module
- `loss.py`: Advanced loss calculations
  - ZTSNR loss implementation
  - High-resolution coherence loss
  - Perceptual loss components
  - Dynamic weight scaling
  - Statistical normalization

- `setup.py`: Training configuration
  - Optimizer selection (AdamW8bit/Adafactor)
  - Learning rate scheduling
  - Gradient accumulation
  - Memory optimization
  - Device management

- `trainer.py`: Enhanced training loop
  - Progressive denoising steps
  - Dynamic batch sizing
  - Automatic validation
  - Statistics tracking
  - Emergency checkpointing

- `utils.py`: Training utilities
  - Memory monitoring
  - Performance profiling
  - Error handling
  - Device optimization
  - Cache management

### Utils Module
- `checkpoint.py`: State management
  - Efficient state saving
  - Safe loading mechanisms
  - Emergency backup system
  - Validation on load
  - Partial state updates

- `device.py`: Resource optimization
  - Memory tracking
  - Device allocation
  - Cache clearing
  - Gradient optimization
  - Memory defragmentation

- `hub.py`: Model distribution
  - HuggingFace integration
  - Safe upload system
  - Version management
  - Documentation updates
  - Access control

- `logging.py`: Monitoring system
  - W&B integration
  - Performance tracking
  - Error logging
  - Statistics visualization
  - Progress monitoring

- `model_card.py`: Documentation
  - Automatic card generation
  - Training details logging
  - Performance metrics
  - Configuration recording
  - Usage instructions

## Implementation Notes

### ZTSNR Features
- Implements σ_max ≈ 20000.0
- Progressive noise reduction
- Resolution-adaptive steps
- Color fidelity improvements
- Composition stability

### Memory Management
- Automatic garbage collection
- Gradient checkpointing
- Efficient attention mechanisms
- Dynamic batch sizing
- Cache optimization

### Validation System
- Automated testing suite
- Performance metrics
- Quality assessments
- Statistical analysis
- Progress tracking

## Usage Guidelines

See main README.md for:
- Installation instructions
- Training commands
- Configuration options
- Performance tips
- Troubleshooting