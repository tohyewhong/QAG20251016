#!/bin/bash

# 2-GPU vLLM Server Script for NVIDIA TITAN RTX (24GB each)
# Optimized for Llama-3.1-8B-Instruct with 2 GPUs

MODEL_PATH=$1

# Set environment variables for stability
export NCCL_P2P_DISABLE=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting vLLM server with 2 GPUs..."
echo "Model: $MODEL_PATH"
echo "GPUs: 0,1 (NVIDIA TITAN RTX 24GB each)"
echo "Tensor Parallel Size: 2"

# Use both GPUs with optimized settings for 24GB VRAM
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 128 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template chat_templates/tool_chat_template_llama3.1_json.jinja \
    --dtype bfloat16 \
    --swap-space 4 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    > out.log 2>&1 &

echo "Server starting... Check out.log for progress"
echo "PID: $!"
echo "Health check: curl http://localhost:8000/health"
