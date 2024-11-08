# SDXL Training Enhancements by NovelAI Diffusion V3

This repository contains code for training Stable Diffusion XL (SDXL) at a resolution of 1024x1024, incorporating several enhancements inspired by the paper:

**"Improvements to SDXL in NovelAI Diffusion V3"**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Memory Optimization Techniques](#memory-optimization-techniques)
- [Dataset Preparation](#dataset-preparation)
- [Training Details](#training-details)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## Introduction

This project implements several improvements to SDXL as described in the NovelAI Diffusion V3 paper. These enhancements aim to improve generation results, particularly for high-resolution image synthesis.

## Features

- **v-Prediction Parameterization**: Transitions from noise prediction to data prediction as signal-to-noise ratio (SNR) changes, ensuring better training across different noise levels.

- **Zero Terminal Signal-to-Noise Ratio (ZTSNR)**: Introduces training up to infinite noise levels, allowing the model to handle pure noise during inference and improving mean color prediction.

- **High-Noise Timesteps for High-Resolution Generation**: Adjusts the noise schedule to include higher sigma values, aiding in generating coherent high-resolution images without artifacts.

- **MinSNR Loss Weighting**: Balances the learning of different timesteps by weighting the loss according to the difficulty of each timestep.

- **Aspect-Ratio Bucketing**: Organizes training images into buckets based on aspect ratio, allowing for better framing and token efficiency compared to center-crop regimes.

- **Precomputed Text Embeddings**: Caches text embeddings to reduce computation and memory usage during training.

- **Memory Optimizations**:
  - Gradient Accumulation
  - Gradient Checkpointing
  - Attention Slicing
  - PyTorch Memory-Efficient Attention

- **Mixed-Precision Training**: BF16 precision.

## Installation

1. **Clone the repository**:

```bash
   git clone https://github.com/yourusername/SDXL-Enhancements-NovelAI-Diffusion-V3.git
   cd SDXL-Enhancements-NovelAI-Diffusion-V3
   pip install -r requirements.txt
```

## Usage

1.**Prepare Your Data**
Place your training images in a directory.
For each image, create a .txt file with the same name containing the caption.
Ensure your dataset is well-labeled and enriched with detailed captions.

2. Cache Latents and Embeddings
The script will automatically cache the latents and text embeddings during the first run. This reduces computation during training.

3. Run the Training Script
Use the following command to start training with DeepSpeed:

```bash
python HighSigma.py \
  --model_path /path/to/your/sdxl/model \
  --data_dir /path/to/your/data \
  --cache_dir ./latents_cache \
  --learning_rate 1e-5 \
  --num_epochs 20 \
  --num_inference_steps 28 \
  --gradient_accumulation_steps 4 \
  --ema_decay 0.9999

```
Replace /path/to/your/sdxl/model with the path to your SDXL model checkpoint, and /path/to/your/data with the path to your dataset.

4. Training Options
--model_path: Path to the pre-trained SDXL model.
--data_dir: Directory containing training images and captions.
--cache_dir: Directory to cache latents and embeddings.
--learning_rate: Learning rate for the optimizer.
--num_epochs: Number of training epochs.
--num_inference_steps: Number of inference steps for sigma schedule.
--gradient_accumulation_steps: Number of steps to accumulate gradients.
--ema_decay: Decay rate for Exponential Moving Average.
Memory Optimization Techniques
To enable training at 1024x1024 resolution without running into memory issues, several optimization techniques are employed:

### Acknowledgements
This project is inspired by the following paper:

**"Improvements to SDXL in NovelAI Diffusion V3"** by Juan Ossa, Eren Doğan, Alex Birch, and F. Johnson.

link: https://arxiv.org/pdf/2409.15997

## License

This project is licensed under the apachie 2.0 License. See the LICENSE file for details.
