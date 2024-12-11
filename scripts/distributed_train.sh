#!/bin/bash
NUM_GPUS=${1:-1}
PORT=${2:-12355}

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --unet_path path/to/unet.safetensors \
    --resume_from_checkpoint path/to/checkpoint 