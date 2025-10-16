#!/bin/bash

# Working 2-GPU vLLM Server Script
# Based on successful single GPU configuration

MODEL_PATH=$1

# Set environment variables
export NCCL_P2P_DISABLE=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

echo "Starting vLLM server with 2 GPUs..."
echo "Model: $MODEL_PATH"
echo "Using GPUs: 0,1"

# Use the same working configuration as single GPU but with 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --swap-space 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --dtype float16 \
    --max-num-batched-tokens 2048 \
    --max-num-partial-prefills 1 \
    > out.log 2>&1 &

echo "Server starting... Check out.log for progress"
echo "PID: $!"
echo "Health check: curl http://localhost:8000/health"
