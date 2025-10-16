MODEL_PATH=$1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" --host localhost --port 8000 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-num-seqs 32 --trust-remote-code --enable-auto-tool-choice --tool-call-parser llama3_json --dtype float16 --max-model-len 4096 --swap-space 4 --disable-log-requests > out.log