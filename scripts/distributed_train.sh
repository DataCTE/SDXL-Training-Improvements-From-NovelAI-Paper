#!/bin/bash
NUM_GPUS=${1:-1}
BASE_PORT=${2:-12355}

# Add network check
check_network() {
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if ! ping -c 1 localhost &>/dev/null; then
            echo "Network connectivity issue detected"
            exit 1
        fi
    done
}

# Add bandwidth check
check_bandwidth() {
    if [ $NUM_GPUS -gt 1 ]; then
        # Test NVLink/PCIe bandwidth
        nvidia-smi nvlink -s
        nvidia-smi topo -m
    fi
}

check_network
check_bandwidth

# Find available port in range
for PORT in $(seq $BASE_PORT $((BASE_PORT + 10))); do
    if ! netstat -tuln | grep -q ":$PORT "; then
        break
    fi
done

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --unet_path path/to/unet.safetensors 