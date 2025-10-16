#!/usr/bin/env python3

import sys
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.append('.')

from src.utils import Config, load_yaml_config
from langchain_openai import ChatOpenAI

def optimized_qag_agent():
    """Optimized QAG agent with reduced API calls and better efficiency"""
    
    print("🚀 Starting Optimized QAG Agent with OpenAI API")
    print("=" * 60)
    
    # Load configuration
    config_dict = load_yaml_config('configs/openai.yaml')
    config = Config(**config_dict)
    print(f"✅ Config loaded - Model: {config.model_name}")
    
    # Create OpenAI model
    model = ChatOpenAI(
        model_name=config.model_name,
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
        temperature=0.8,
    )
    print("✅ Model created successfully")
    
    # Load data
    data = []
    with open(config.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"✅ Data loaded: {len(data)} samples")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Process each sample with optimized approach
    for i, sample_data in enumerate(data):
        print(f"\n📝 Processing sample {i+1}/{len(data)}")
        
        # Extract text from sample
        if isinstance(sample_data['text'], list):
            text = ' '.join(sample_data['text'])
        else:
            text = sample_data['text']
        
        print(f"Text: {text[:100]}...")
        
        try:
            # OPTIMIZATION 1: Generate questions and answers in one call
            combined_prompt = f"""
Based on the following context, generate 3 complex questions that require multiple reasoning steps to answer, and then provide comprehensive answers for each question.

Context: {text}

Please provide your response in the following JSON format:
{{
    "questions": [
        "Question 1 here",
        "Question 2 here", 
        "Question 3 here"
    ],
    "answers": [
        {{
            "question": "Question 1 here",
            "answer": "Comprehensive answer based only on the context"
        }},
        {{
            "question": "Question 2 here", 
            "answer": "Comprehensive answer based only on the context"
        }},
        {{
            "question": "Question 3 here",
            "answer": "Comprehensive answer based only on the context"
        }}
    ]
}}

Each question should be complex and require multiple reasoning steps. Each answer should be comprehensive and based only on the information in the context.
"""
            
            print("🔄 Generating questions and answers in single API call...")
            response = model.invoke(combined_prompt)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                response_text = response.content.strip()
                
                # Find JSON in the response (in case there's extra text)
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_text = response_text[start_idx:end_idx]
                    result_data = json.loads(json_text)
                    
                    questions = result_data.get('questions', [])
                    answers = result_data.get('answers', [])
                    
                    print(f"✅ Generated {len(questions)} questions and {len(answers)} answers in 1 API call")
                    
                    for j, q in enumerate(questions):
                        print(f"   {j+1}. {q}")
                    
                    # Save results
                    result = {
                        "sample_id": i,
                        "text": text,
                        "questions": questions,
                        "answers": answers,
                        "api_calls_used": 1  # Track efficiency
                    }
                    
                    output_file = os.path.join(config.output_dir, f"optimized_sample_{i+1}.json")
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"✅ Saved results to {output_file} (1 API call)")
                    
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  JSON parsing failed, falling back to text parsing: {e}")
                
                # Fallback: parse as text
                response_text = response.content.strip()
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                
                # Simple text parsing fallback
                questions = []
                answers = []
                
                # This is a basic fallback - in practice, you might want more sophisticated parsing
                for line in lines:
                    if line.endswith('?'):
                        questions.append(line)
                
                print(f"✅ Generated {len(questions)} questions (fallback mode)")
                
                # Save with fallback data
                result = {
                    "sample_id": i,
                    "text": text,
                    "questions": questions,
                    "answers": [],  # Empty for fallback
                    "raw_response": response_text,
                    "api_calls_used": 1
                }
                
                output_file = os.path.join(config.output_dir, f"optimized_sample_{i+1}.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✅ Saved results to {output_file} (fallback mode)")
            
        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {e}")
            continue
    
    print(f"\n🎉 Optimized QAG Agent completed!")
    print(f"📊 Efficiency: Reduced from 4 API calls per sample to 1 API call per sample")
    print(f"📁 Check {config.output_dir} for results (files prefixed with 'optimized_')")

def parallel_qag_agent():
    """Even more optimized version with parallel processing"""
    
    print("🚀 Starting Parallel QAG Agent with OpenAI API")
    print("=" * 60)
    
    # Load configuration
    config_dict = load_yaml_config('configs/openai.yaml')
    config = Config(**config_dict)
    print(f"✅ Config loaded - Model: {config.model_name}")
    
    # Create OpenAI model
    model = ChatOpenAI(
        model_name=config.model_name,
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
        temperature=0.8,
    )
    print("✅ Model created successfully")
    
    # Load data
    data = []
    with open(config.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"✅ Data loaded: {len(data)} samples")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    def process_sample(sample_data, sample_id):
        """Process a single sample"""
        try:
            # Extract text from sample
            if isinstance(sample_data['text'], list):
                text = ' '.join(sample_data['text'])
            else:
                text = sample_data['text']
            
            print(f"📝 Processing sample {sample_id+1}")
            
            # Combined prompt for efficiency
            combined_prompt = f"""
Based on the following context, generate 3 complex questions that require multiple reasoning steps to answer, and then provide comprehensive answers for each question.

Context: {text}

Please provide your response in the following JSON format:
{{
    "questions": [
        "Question 1 here",
        "Question 2 here", 
        "Question 3 here"
    ],
    "answers": [
        {{
            "question": "Question 1 here",
            "answer": "Comprehensive answer based only on the context"
        }},
        {{
            "question": "Question 2 here", 
            "answer": "Comprehensive answer based only on the context"
        }},
        {{
            "question": "Question 3 here",
            "answer": "Comprehensive answer based only on the context"
        }}
    ]
}}
"""
            
            response = model.invoke(combined_prompt)
            
            # Parse response (simplified for this example)
            questions = []
            answers = []
            
            # Basic parsing - in practice, you'd want more robust JSON parsing
            response_text = response.content.strip()
            
            # Save results
            result = {
                "sample_id": sample_id,
                "text": text,
                "questions": questions,
                "answers": answers,
                "raw_response": response_text,
                "api_calls_used": 1
            }
            
            output_file = os.path.join(config.output_dir, f"parallel_sample_{sample_id+1}.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Completed sample {sample_id+1}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing sample {sample_id+1}: {e}")
            return False
    
    # Process samples in parallel (limited to avoid rate limits)
    print("🔄 Processing samples in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 to avoid rate limits
        futures = [executor.submit(process_sample, sample_data, i) for i, sample_data in enumerate(data)]
        
        # Wait for completion
        results = [future.result() for future in futures]
    
    successful = sum(results)
    print(f"\n🎉 Parallel QAG Agent completed!")
    print(f"📊 Processed {successful}/{len(data)} samples successfully")
    print(f"📁 Check {config.output_dir} for results (files prefixed with 'parallel_')")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized QAG Agent')
    parser.add_argument('--mode', choices=['optimized', 'parallel'], default='optimized',
                       help='Processing mode: optimized (sequential, 1 API call per sample) or parallel (concurrent processing)')
    
    args = parser.parse_args()
    
    if args.mode == 'optimized':
        optimized_qag_agent()
    elif args.mode == 'parallel':
        parallel_qag_agent()
