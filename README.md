# QAG-Agent
 
## Installation (One-time)
 
### Environment Set-up
 
1. Install uv in Linux by running:
 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 
2. Create the environment:
 
```
cd qag-agent
uv sync
```
 
3. Start the environment by running:
 
```
source .venv/bin/activate
export NCCL_P2P_DISABLE=1
```
 
4. In the `configs/` folder, prepare a config file. An example config file is provided in the folder, `default.yaml`. You can modify any of the fields as needed.
 
```
host: localhost
port: 8000
 
output_dir: output/musique
data_path: data/musique/train-data.jsonl
```
 
### Prepare Data
 
The data should be formatted in `.jsonl` formats, where each line is a data point containing the key `"text"` to a list of strings. An example of a jsonline data point is:
 
```
{"text": ["This is the first document for this data point.", "This is the second document for this data point."]}
```
 
An example of how data should be formatted can be found in the data folder after running the following command:
 
```
tar -zxvf data.tar.gz
```
 
The file structure is as follows:
 
```
├── data
│   └── musique
│       ├── train-data.jsonl
│       └── dev-data.jsonl
```
 
## Running (Every time)
 
### Terminal 1 (VLLM)
1. Start the environment by running:
 
```bash
source .venv/bin/activate
export NCCL_P2P_DISABLE=1
```
 
2. Start the vLLM server with the following command. You can replace `meta-llama/Llama-3.1-8B-Instruct` with any HuggingFace Repo ID or a path to a local model.
 
```bash
bash model_hosting/run_vllm.sh meta-llama/Llama-3.1-8B-Instruct
```
 
3. When the server is ready, you should see in `out.log`:
 
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
 
### Terminal 2 (Python)
 
1. Start the environment by running:
 
```bash
source .venv/bin/activate
(qag-agent) root@main-0:~/qag# export NCCL_P2P_DISABLE=1
```
 
 
2. Start the agent by running the following command:
 
```bash
python src/main.py -c default.yaml
```
 
3. The outputs are such that each document will have its own json file, and each file should have the input context, generated questions, answers, and explanations.
 
```
output/musique/musique_0.json
output/musique/musique_1.json
output/musique/musique_2.json
 
...
```
```
{
    "context": "Example context...",
    "questions": [
        "First question?",
        "Second question?"
    ],
    "answers": [
        {
            "answer": "Answer to the first question.",
            "explanation": "Explanation to the first question"
        },
        {
            "answer": "Answer to the second question.",
            "explanation": "Explanation to the second question"
        },
    ]
}
```
 
## GPU Configuration
 
In `model_hosting/run_vllm.sh`,
 
For a one GPU setup:
```bash
MODEL_PATH=$1
 
CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL_PATH} \
    --host localhost --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --dtype bfloat16 \
    > out.log 2>&1
```
 
For a two GPU setup:
```bash
MODEL_PATH=$1
 
CUDA_VISIBLE_DEVICES=0,1 vllm serve ${MODEL_PATH} \
    --host localhost --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --dtype bfloat16 \
    > out.log 2>&1
```
 
For a four GPU setup:
```bash
MODEL_PATH=$1
 
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${MODEL_PATH} \
    --host localhost --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --dtype bfloat16 \
    > out.log 2>&1
```
 