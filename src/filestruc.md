# Project Structure Documentation

## Overview
This project implements a Stable Diffusion XL training pipeline with modular components for efficient training, validation, and model management.

## Directory Structure

```
src/
├── data/
│   └── dataset.py           # Custom dataset implementation with caching
│
├── models/
│   ├── model_validator.py   # Model validation utilities
│   ├── setup.py            # Model initialization and setup
│   ├── tag_weighter.py     # Tag-based loss weighting system
│   └── vae_finetuner.py    # VAE fine-tuning implementation
│
├── training/
│   ├── loss.py             # Loss calculation utilities
│   ├── setup.py            # Training setup and configuration
│   ├── trainer.py          # Main training loop implementation
│   └── utils.py            # Training utility functions
│
├── utils/
│   ├── checkpoint.py       # Checkpoint saving and loading
│   ├── device.py          # Device management utilities
│   ├── hub.py             # Hugging Face Hub integration
│   ├── logging.py         # Logging configuration
│   └── model_card.py      # Model card generation
│
└── main.py                 # Main entry point
```

## Component Details

### Data Module
- `dataset.py`: Implements custom dataset with efficient caching of latents and embeddings
  - Handles aspect ratio bucketing
  - Implements CLIP embedding caching
  - Supports both single and batch processing

### Models Module
- `model_validator.py`: Validates model outputs and configurations
- `setup.py`: Handles model initialization and configuration
- `tag_weighter.py`: Implements tag-based loss weighting system
- `vae_finetuner.py`: Handles VAE fine-tuning with statistics tracking

### Training Module
- `loss.py`: Implements v-prediction loss and perceptual loss
- `setup.py`: Configures training components
- `trainer.py`: Implements main training loop
- `utils.py`: Contains training utility functions

### Utils Module
- `checkpoint.py`: Handles model checkpoint operations
- `device.py`: Manages model device placement
- `hub.py`: Handles Hugging Face Hub interactions
- `logging.py`: Configures logging system
- `model_card.py`: Generates model cards for documentation
