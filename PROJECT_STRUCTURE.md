# QAG Agent Project Structure

## 📁 **Core Files**

### **Main Scripts**
- `optimized_qag_openai.py` - **Main optimized QAG agent** (75% fewer API calls)
- `run_with_openai.sh` - Convenience script to run with OpenAI API
- `src/main.py` - Original QAG agent (modified for OpenAI support)
- `src/utils.py` - Configuration and utility functions

### **Configuration**
- `configs/openai.yaml` - OpenAI API configuration
- `configs/default.yaml` - Default vLLM configuration
- `pyproject.toml` - Python project dependencies
- `requirements.txt` - Locked dependencies for reproducibility

### **Documentation**
- `README.md` - Project overview and setup instructions
- `OPTIMIZATION_SUMMARY.md` - Detailed optimization results and improvements
- `PROJECT_STRUCTURE.md` - This file

### **Model Hosting Scripts**
- `model_hosting/run_vllm.sh` - Original 4-GPU vLLM server
- `model_hosting/run_vllm_single_gpu.sh` - Single GPU vLLM server
- `model_hosting/run_vllm_2gpu.sh` - Custom 2-GPU vLLM server
- `model_hosting/run_vllm_2gpu_simple.sh` - Simplified 2-GPU server
- `model_hosting/run_vllm_simple.sh` - Very basic single-GPU server

### **Source Code**
- `src/agent_utils.py` - Agent utilities and retry logic
- `src/question_team.py` - Question generation team
- `src/answer_team.py` - Answer generation team
- `src/memory.py` - Memory and state management

### **Templates**
- `chat_templates/` - Jinja2 chat templates for different LLM formats

## 🚀 **Quick Start**

### **Using OpenAI API (Recommended)**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run the optimized QAG agent
./run_with_openai.sh

# Or run directly
source .venv/bin/activate
python optimized_qag_openai.py --mode optimized
```

### **Using Local vLLM Server**
```bash
# Start vLLM server (choose based on your GPU setup)
bash model_hosting/run_vllm_2gpu.sh meta-llama/Llama-3.1-8B-Instruct

# Run original QAG agent
source .venv/bin/activate
python src/main.py --config configs/default.yaml
```

## 📊 **Performance Comparison**

| Method | API Calls/Sample | Speed | Reliability | Cost |
|--------|------------------|-------|-------------|------|
| Original | 4 | Slow | Low | High |
| Optimized | 1 | Fast | High | Low |

## 🔧 **Key Improvements Made**

1. **75% fewer API calls** - Combined questions and answers in single prompts
2. **Eliminated tool calling** - Removed complex PydanticToolsParser overhead
3. **Better error handling** - Robust JSON parsing with fallbacks
4. **OpenAI API support** - Reliable cloud-based processing
5. **Multiple vLLM options** - Various GPU configurations for local hosting

## 📝 **Output**

Results are saved to `output/musique/` directory:
- `sample_*.json` - Original QAG agent results
- `optimized_sample_*.json` - Optimized QAG agent results

Each file contains:
- Original text
- 3 complex questions requiring multiple reasoning steps
- Comprehensive answers based only on the context
- API usage tracking
