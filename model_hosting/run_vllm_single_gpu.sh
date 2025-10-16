#!/bin/bash
MODEL_PATH=$1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Set environment variable to help with memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use only GPU 0 with conservative memory settings
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
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
    > out.log 2>&1

