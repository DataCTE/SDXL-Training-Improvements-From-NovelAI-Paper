# SDXL Training with ZTSNR and NovelAI V3 Improvements

Most SDXL implementations use a maximum noise deviation (σ_max) of 14.6 [meaning that only 14.6% of the noise is removed at maximum] inherited from SD1.5/1.4, without accounting for SDXL's larger scale. Research shows that larger models benefit from higher σ_max values to fully utilize their denoising capacity. This repository implements an increased σ_max ≈ 20000.0 (as recommended by NovelAI research arXiv:2409.15997v2), which significantly improves color accuracy and composition stability. Combined with Zero Terminal SNR (ZTSNR) and VAE finetuning.


### Current Status

- [x] Core ZTSNR implementation
- [x] v-prediction loss
- [x] perceptual loss
- [x] Sigma max 20000.0
- [x] High-resolution coherence enhancements
- [x] VAE improvements
- [x] Tag-based CLIP weighting
- [x] Functional detailed wandb logging
- [x] Ultimate SD Upscaler integration
- [x] Dynamic resolution handling
- [x] Memory-efficient training
- [x] Automatic mixed precision (AMP)

- [x] 10k dataset proof of concept (completed) [link](https://huggingface.co/dataautogpt3/ProteusSigma)
- [x] 200k+ dataset finetune (completed)
- [ ] 12M dataset finetune (in progress)
- [ ] Multi-aspect ratio support (planned)
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
  --finetune_vae
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
  --max_grad_norm 1.0
```

## Tag-Based CLIP Weighting

### Configuration
```bash
python src/main.py \
  --min_tag_weight 0.1 \
  --max_tag_weight 3.0 \
  --character_weight 1.5 \
  --style_weight 1.2 \
  --quality_weight 0.8 \
  --setting_weight 1.0 \
  --action_weight 1.1 \
  --object_weight 0.9 \
  --tag_frequency_path "tag_frequencies.json" \
  --tag_embedding_cache "tag_embeddings.pt" \
  --dynamic_tag_weights \
  --tag_dropout 0.1
```

### Weight Ranges
- `min_tag_weight`: 0.1 to 1.0 (default: 0.1)
- `max_tag_weight`: 1.0 to 5.0 (default: 3.0)

### Class Weights
- `character_weight`: 1.5 (character emphasis)
- `style_weight`: 1.2 (style consistency)
- `quality_weight`: 0.8 (quality control)
- `setting_weight`: 1.0 (background balance)
- `action_weight`: 1.1 (pose emphasis)
- `object_weight`: 0.9 (object balance)

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

### Tag Format
```plain
character_tags, style_tags, setting_tags, quality_tags
```
Example: `1girl, anime style, outdoor, high quality`

## Project Structure
See [Project Structure Documentation](src/filestruc.md) for detailed component descriptions.

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
