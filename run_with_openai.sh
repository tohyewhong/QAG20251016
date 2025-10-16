#!/bin/bash

# QAG Agent with OpenAI API
echo "🚀 Starting QAG Agent with OpenAI API"

# Check if API key is set in the config file
if grep -q "YOUR_OPENAI_API_KEY_HERE" configs/openai.yaml; then
    echo "❌ Error: Please set your OpenAI API key in configs/openai.yaml"
    echo ""
    echo "Edit configs/openai.yaml and replace 'YOUR_OPENAI_API_KEY_HERE' with your actual API key"
    echo "You can get an API key from: https://platform.openai.com/api-keys"
    exit 1
fi

echo "✅ OpenAI API key found in config file"
echo "📁 Using config: configs/openai.yaml"
echo "📊 Data: data/musique/train-data.jsonl"
echo "📤 Output: output/musique/"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run the QAG agent
echo "🎯 Starting QAG Agent..."
python src/main.py --config configs/openai.yaml

echo "✅ QAG Agent completed!"
echo "📁 Check output/musique/ for results"
