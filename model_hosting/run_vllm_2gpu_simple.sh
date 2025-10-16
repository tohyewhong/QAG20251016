#!/bin/bash

# Simple 2-GPU vLLM Server Script
# Conservative settings for reliable startup

MODEL_PATH=$1

# Set environment variables
export NCCL_P2P_DISABLE=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

echo "Starting vLLM server with 2 GPUs (simple config)..."
echo "Model: $MODEL_PATH"

# Simple, conservative configuration
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template chat_templates/tool_chat_template_llama3.1_json.jinja \
    --dtype bfloat16 \
    > out.log 2>&1 &

echo "Server starting... Check out.log for progress"
echo "PID: $!"
