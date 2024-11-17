# Project Structure Documentation

## Overview
This project implements a Stable Diffusion XL training pipeline with Zero Terminal SNR (ZTSNR), high-resolution coherence enhancements, and VAE improvements from NovelAI research. The codebase is organized for maximum efficiency, modularity, and extensibility.

## Directory Structure

```
src/
├── config/
│   ├── __init__.py
│   └── args.py            # Centralized configuration management
│
├── data/
│   ├── __init__.py
│   ├── caption_processor.py # Tag processing and weighting
│   ├── image_processor.py  # Image loading and transformation
│   ├── latent_cache.py    # VAE latent caching system
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── base.py        # Base dataset class
│   │   ├── bucket_manager.py # Resolution bucket management
│   │   ├── dataset.py     # Main dataset implementation
│   │   ├── dataset_initializer.py # Dataset setup
│   │   ├── image_grouper.py # Image resolution grouping
│   │   └── dataloader.py  # Custom data loading
│   └── utils.py          # Data processing utilities
│
├── models/
│   ├── __init__.py
│   ├── architecture.py   # Model architecture definitions
│   ├── attention.py      # Memory-efficient attention
│   └── vae.py           # Enhanced VAE components
│
├── training/
│   ├── __init__.py
│   ├── ema.py           # Enhanced EMA with scheduling
│   ├── loss.py          # ZTSNR loss implementations
│   ├── trainer.py       # Memory-efficient training loop
│   └── vae_finetuner.py # VAE fine-tuning
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py    # Model state management
│   ├── device.py       # Memory and device optimization
│   ├── hub.py          # HuggingFace integration
│   ├── logging.py      # Training monitoring
│   └── validation.py   # Model validation
│
└── main.py             # Entry point with CLI
```

## Component Details

### Config Module
- `args.py`: Centralized configuration
  - Training parameters
  - Dataset configuration
  - Model architecture settings
  - Tag weighting parameters
  - Optimization settings
  - Validation configuration

### Data Module
- `caption_processor.py`: Tag processing system
  - Tag extraction and normalization
  - Weight computation
  - Tag statistics tracking
  - Dynamic weight adjustment
  - Cache management

- `dataset/`: Advanced dataset implementation
  - `bucket_manager.py`: Resolution bucket system
    - Dynamic bucket generation
    - Aspect ratio preservation
    - Area constraints
    - Adaptive bucketing
  - `dataset.py`: Main dataset
    - Multi-resolution handling
    - Latent caching
    - Tag weighting
    - Memory optimization
  - `image_grouper.py`: Resolution grouping
    - Efficient bucket assignment
    - Parallel processing
    - Memory management


### Training Module
- `loss.py`: ZTSNR loss system
  - V-prediction loss
  - Dynamic weighting
  - Tag-weighted loss
  - Gradient scaling

- `trainer.py`: Advanced training loop
  - Mixed precision training
  - Dynamic batch sizing
  - Memory optimization
  - Progress tracking

### Utils Module
- `checkpoint.py`: State management
  - Safe state saving/loading
  - Version control
  - Metadata tracking

- `device.py`: Resource optimization
  - Memory tracking
  - CUDA optimization
  - Cache management

## Implementation Notes

### Memory Management
- Bucket-based batching
- Efficient latent caching
- Dynamic resolution handling
- Memory-mapped data
- Streaming processing

### Training Features
- Multi-resolution support
- Dynamic tag weighting
- Adaptive bucket selection
- Progressive training
- Automated logging

### Performance Optimizations
- Mixed precision training
- Memory-efficient attention
- Bucket-based batching
- Efficient data loading
- Caching strategies

## Usage Guidelines

Refer to main README.md for:
- Installation steps
- Training configurations
- Performance optimization
- Troubleshooting guide
- Best practices