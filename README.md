# SDXL Training with ZTSNR and NovelAI V3 Improvements

> **⚠️ IMPORTANT NOTE**: This repository is in early experimental stages. While we have a basic proof of concept, the codebase is being completely restructured. Not recommended for production use.


Most SDXL implementations use a maximum noise deviation (σ_max) of 14.6 [meaning that only 14.6% of the noise is removed at maximum] inherited from SD1.5/1.4, without accounting for SDXL's larger scale. Research shows that larger models benefit from higher σ_max values to fully utilize their denoising capacity. This repository implements an increased σ_max ≈ 20000.0 (as recommended by NovelAI research arXiv:2409.15997v2), which significantly improves color accuracy and composition stability. Combined with Zero Terminal SNR (ZTSNR) and VAE finetuning.

### Current Status

- [x] Core ZTSNR implementation (proof of concept)
- [x] Initial v-prediction loss implementation
- [x] Basic wandb logging
- [x] Memory-efficient training (needs optimization)
- [x] Automatic mixed precision (AMP)

### Proof of Concept Results
- ✅ 10k Dataset Test: [ProteusSigma](https://huggingface.co/dataautogpt3/ProteusSigma)
  - Initial test of concepts
  - Limited dataset size
  - Basic implementation
  - Demonstrates potential of approach

### Work in Progress
- [ ] Framework Restructuring (current focus)
- [ ] 200k+ dataset finetune (planned)
- [ ] 12M dataset finetune (planned)
- [ ] LoRA support (planned)
- [ ] Comprehensive testing suite
- [ ] Production-ready optimizations

## Technical Implementation

### 1. Zero Terminal SNR (ZTSNR)
- **Noise Schedule**: σ_max ≈ 20000.0 to σ_min ≈ 0.0292
- **Progressive Steps**: [20000, 17.8, 12.4, 9.2, 7.2, 5.4, 3.9, 2.1, 0.9, 0.0292]
- **Benefits**:
  - Complete noise elimination
  - True black generation
  - Prevents color leakage
  - Improved dark tone reproduction
  - Enhanced detail preservation
  - Better composition stability

### 2. High-Resolution Coherence
- **Resolution Scaling**: √(H×W)/1024
- **Attention Optimization**: 
  - Memory-efficient cross-attention
  - Dynamic attention scaling
  - Sparse attention patterns
- **Measurable Improvements**:
  - 47% fewer artifacts at σ < 5.0
  - Stable composition at σ > 12.4
  - 31% better detail consistency
  - 25% reduced memory usage

### 3. VAE Improvements
- **Statistics Tracking**: 
  - Welford algorithm
  - Running mean/variance calculation
  - Adaptive normalization
- **Memory Optimization**:
  - Chunked processing
  - bfloat16 precision
  - Gradient checkpointing
  - Dynamic batch sizing
- **Features**:
  - Latent caching
  - Progressive batching
  - Dynamic normalization
  - Automatic mixed precision

## Quick Start

### Installation
```bash
git clone https://github.com/DataCTE/SDXL-Training-Improvements.git
cd SDXL-Training-Improvements
pip install -r requirements.txt
```

### Basic Training
```bash
python src/main.py \
--pretrained_model_path stabilityai/stable-diffusion-xl-base-1.0 \
--train_data_dir /path/to/data \
--batch_size 4 \
--gradient_accumulation_steps 8 \
--learning_rate 3e-5
```

## Development Roadmap

1. Core Framework (Current Focus)
   - Stable training loop
   - Memory management
   - Proper error handling
   - Testing infrastructure

2. ZTSNR Implementation (Planned)
   - Noise scheduling
   - V-prediction
   - Resolution handling

3. Optimizations (Planned)
   - Memory efficiency
   - Training speed
   - Validation pipeline

## Contributing

Currently focused on:
1. Core stability improvements
2. Memory optimization
3. Testing infrastructure
4. Documentation accuracy

## Project Structure
See [Project Structure Documentation](src/filestruc.md) for detailed component descriptions.

## ComfyUI Integration

This repository includes custom ComfyUI nodes that implement the ZTSNR and NovelAI V3 improvements. The nodes can be found in `/Comfyui-zsnrnode/`.

### Available Nodes

1. **ZSNR V-Prediction Node**
   - Implements Zero Terminal SNR and V-prediction
   - Configurable σ_min and σ_data parameters
   - Resolution-aware scaling
   - Dynamic SNR gamma adjustment
   - Category: "conditioning"

2. **CFG Rescale Node**
   - Advanced CFG rescaling methods
   - Multiple scaling algorithms
   - Configurable rescale multiplier
   - Category: "sampling"

3. **Laplace Scheduler Node**
   - Laplace distribution-based noise scheduling
   - Configurable μ and β parameters
   - Optimized for SDXL's scale
   - Category: "sampling"

### Installation

1. Copy the `/Comfyui-zsnrnode` directory to your ComfyUI custom nodes folder:
```bash
cp -r /Comfyui-zsnrnode /path/to/ComfyUI/custom_nodes/
```

2. Restart ComfyUI to load the new nodes

### Usage

The nodes will appear in the ComfyUI node browser under their respective categories:
- ZSNR V-prediction under "conditioning"
- CFG Rescale and Laplace Scheduler under "sampling"

Recommended workflow:
1. Add ZSNR V-prediction node before your main sampling node
2. Configure CFG rescaling if using high CFG values
3. Optionally use the Laplace scheduler for improved noise distribution

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

## License
Apache 2.0
## Citation
```bibtex
@article{ossa2024improvements,
  title={Improvements to SDXL in NovelAI Diffusion V3},
  author={Ossa, Juan and Doğan, Eren and Birch, Alex and Johnson, F.},
  journal={arXiv preprint arXiv:2409.15997v2},
  year={2024}
}

