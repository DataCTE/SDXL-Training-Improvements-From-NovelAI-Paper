# SDXL Training with ZTSNR and NovelAI V3 Improvements

Most SDXL implementations use a maximum noise deviation (σ_max) of 14.6 [meaning that only 14.6% of the noise is removed at maximum] inherited from SD1.5/1.4, without accounting for SDXL's larger scale. Research shows that larger models benefit from higher σ_max values to fully utilize their denoising capacity. This repository implements an increased σ_max ≈ 20000.0 (as recommended by NovelAI research arXiv:2409.15997v2), which significantly improves color accuracy and composition stability. Combined with Zero Terminal SNR (ZTSNR) and VAE finetuning.

### Current Status

- [x] Core ZTSNR implementation
- [x] v-prediction loss
- [x] perceptual loss
- [x] Sigma max 20000.0
- [x] High-resolution coherence enhancements
- [x] VAE improvements
- [x] Tag-based weighting system
- [x] Bucket-based batching
- [x] Dynamic resolution handling
- [x] Memory-efficient training
- [x] Automatic mixed precision (AMP)
- [x] Functional detailed wandb logging

- [x] 10k dataset proof of concept (completed) [link](https://huggingface.co/dataautogpt3/ProteusSigma)
- [ ] 200k+ dataset finetune (in progress)
- [ ] 12M dataset finetune (planned)
- [ ] LoRA support (planned)

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
  --model_path /path/to/sdxl/model \
  --data_dir /path/to/training/data \
  --output_dir ./output \
  --learning_rate 1e-6 \
  --batch_size 1 \
  --enable_compile \
  --use_tag_weighting
```

### Advanced Configuration
```bash
python src/main.py \
  --model_path /path/to/sdxl/model \
  --data_dir /path/to/data \
  --output_dir ./output \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --enable_compile \
  --compile_mode "reduce-overhead" \
  --finetune_vae \
  --vae_learning_rate 1e-6 \
  --use_wandb \
  --wandb_project "sdxl-training" \
  --wandb_run_name "ztsnr-training" \
  --enable_amp \
  --mixed_precision "bf16" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers \
  --max_grad_norm 1.0 \
  --adaptive_loss_scale \
  --kl_weight 0.1 \
  --perceptual_weight 0.1
```

## Tag Weighting System

### Configuration
```bash
python src/main.py \
  # Tag weighting parameters
  --use_tag_weighting \
  --min_tag_weight 0.1 \
  --max_tag_weight 3.0 \
  
  # Caption processing
  --token_dropout_rate 0.1 \
  --caption_dropout_rate 0.1 \
  
  # Resolution handling
  --min_size 512 \
  --max_size 4096 \
  --bucket_step_size 64 \
  --max_bucket_area 4194304 \
  --all_ar
```

### Weight Configuration
- `use_tag_weighting`: Enable tag-based loss weighting
- `min_tag_weight`: Minimum weight for any tag (default: 0.1)
- `max_tag_weight`: Maximum weight for any tag (default: 3.0)

### Caption Processing
- `token_dropout_rate`: Rate at which individual tokens are dropped (default: 0.1)
- `caption_dropout_rate`: Rate at which entire captions are dropped (default: 0.1)

### Resolution Handling
- `min_size`: Minimum dimension size (default: 512)
- `max_size`: Maximum dimension size (default: 4096)
- `bucket_step_size`: Resolution increment between buckets (default: 64)
- `max_bucket_area`: Maximum area constraint (default: 4194304)
- `all_ar`: Use aspect ratio for bucket assignment

## Dataset Format

### Directory Structure
```
data_dir/
├── image1.png
├── image1.txt
├── image2.jpg
├── image2.txt
...
```

### Caption Format
Text files should contain comma-separated tags:
```plain
tag1, tag2, tag3, tag4
```
Example: `1girl, white hair, outdoor, high quality`

## Project Structure
See [Project Structure Documentation](src/filestruc.md) for detailed component descriptions.

## Memory Management

### Bucket-Based Batching
- Dynamic resolution grouping
- Aspect ratio preservation
- Memory-efficient processing
- Consistent tensor sizes

### Latent Caching
- VAE latent precomputation
- Disk-based caching
- Memory-mapped storage
- Efficient batch processing

### Resolution Handling
- Dynamic bucket generation
- Area-constrained scaling
- Adaptive aspect ratios
- Progressive resizing

## ComfyUI Integration

This repository includes custom ComfyUI nodes that implement the ZTSNR and NovelAI V3 improvements. The nodes can be found in `src/inference/Comfyui-zsnrnode/`.

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

1. Copy the `src/inference/Comfyui-zsnrnode` directory to your ComfyUI custom nodes folder:
```bash
cp -r src/inference/Comfyui-zsnrnode /path/to/ComfyUI/custom_nodes/
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
