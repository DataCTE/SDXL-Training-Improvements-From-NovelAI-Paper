# Project Structure Documentation

> **Note**: This documentation reflects the current development state. Many features are planned but not yet implemented.

## Current Directory Structure

```
src/
├── config/
│   ├── __init__.py
│   ├── args.py            # Training configuration and CLI args
│   └── defaults.json      # Default configuration values
│
├── data/
│   ├── cacheing/
│   │   ├── __init__.py
│   │   ├── memory.py      # Memory management
│   │   ├── text_embeds.py # Text embedding cache
│   │   └── vae.py        # VAE cache handling
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py       # Base dataset classes
│   │   ├── dataloader.py # Custom dataloader
│   │   ├── dataset_initializer.py
│   │   └── sampler.py    # Custom sampling
│   │
│   ├── image_processing/
│   │   ├── __init__.py
│   │   ├── loading.py    # Image loading utilities
│   │   ├── manipulations.py
│   │   ├── transforms.py # Image transformations
│   │   └── validation.py # Validation utilities
│   │
│   ├── multiaspect/
│   │   ├── __init__.py
│   │   ├── bucket_manager.py
│   │   ├── dataset.py    # Multi-aspect dataset
│   │   └── image_grouper.py
│   │
│   └── prompt/
│       ├── __init__.py
│       └── caption_processor.py
│
├── models/
│   ├── SDXL/
│   │   ├── __init__.py
│   │   ├── pipeline.py   # SDXL pipeline
│   │   └── model_loader.py
│   └── StateTracker.py   # Training state management
│
├── training/
│   ├── optimizers/
│   │   ├── adafactor/
│   │   ├── adamw_bfloat16/
│   │   ├── adamw8bit/
│   │   ├── soap/
│   │   └── setup_optimizers.py
│   │
│   ├── __init__.py
│   ├── trainer.py        # Training loop
│   ├── training_steps.py # Training steps
│   ├── loss_functions.py # Loss computation
│   └── metrics.py        # Training metrics
│
├── utils/
│   ├── __init__.py
│   ├── logging.py        # WandB and file logging
│   └── progress.py       # Progress tracking
│
└── main.py               # Entry point
```

## Implemented Components

### Config Module
- Basic configuration system
- CLI argument parsing
- Default configuration handling

### Data Module
- Basic image loading
- Text embedding caching
- Multi-aspect dataset support
- Bucket management

### Models Module
- SDXL model loading
- Basic pipeline implementation
- State tracking

### Training Module
- Basic training loop
- Mixed precision support
- Multiple optimizer options
- Gradient accumulation

### Utils Module
- Basic wandb logging
- Progress tracking
- File logging

## Under Development

### Core Features
- Memory optimization
- Advanced caching
- Proper validation
- Testing infrastructure

### Planned Features
- ZTSNR implementation
- Advanced noise scheduling
- Resolution handling
- Production optimizations

## Development Guidelines

### Current Focus
1. Core stability
2. Memory management
3. Testing infrastructure
4. Documentation accuracy

### Contributing
- Focus on core functionality
- Add tests for new features
- Update documentation
- Follow existing patterns