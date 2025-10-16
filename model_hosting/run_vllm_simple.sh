#!/bin/bash

# Very simple vLLM server configuration
MODEL_PATH=$1

echo "Starting simple vLLM server..."
echo "Model: $MODEL_PATH"

# Minimal configuration that should work
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host localhost \
    --port 8000 \
    --gpu-memory-utilization 0.5 \
    --trust-remote-code \
    > out.log 2>&1 &

echo "Server starting... PID: $!"
