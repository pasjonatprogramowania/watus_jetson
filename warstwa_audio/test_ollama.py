#!/usr/bin/env python3
"""
Benchmark script for testing Ollama models performance.
Measures tokens per second and response time for each model.
"""

import subprocess
import json
import time
import requests
from datetime import datetime


def get_available_models():
    """Get list of available models from ollama."""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[1:]  # Skip header
    models = []
    for line in lines:
        if line.strip():
            model_name = line.split()[0]
            models.append(model_name)
    return models


def test_model(model_name: str, prompt: str = "Napisz krótką bajkę o kocie w 3 zdaniach."):
    """
    Test a single model and return performance metrics.
    Uses the Ollama API to get detailed token statistics.
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract metrics from Ollama response
            total_duration = data.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
            load_duration = data.get('load_duration', 0) / 1e9
            prompt_eval_duration = data.get('prompt_eval_duration', 0) / 1e9
            eval_duration = data.get('eval_duration', 0) / 1e9
            
            prompt_eval_count = data.get('prompt_eval_count', 0)
            eval_count = data.get('eval_count', 0)
            
            # Calculate tokens per second
            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0
            prompt_tokens_per_second = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
            
            generated_text = data.get('response', '')
            
            print(f"\n📝 Response:\n{generated_text}")
            print(f"\n📊 Performance Metrics:")
            print(f"   ├─ Total time: {total_duration:.2f}s")
            print(f"   ├─ Model load time: {load_duration:.2f}s")
            print(f"   ├─ Prompt processing: {prompt_eval_duration:.2f}s ({prompt_eval_count} tokens)")
            print(f"   ├─ Generation time: {eval_duration:.2f}s ({eval_count} tokens)")
            print(f"   ├─ Prompt tokens/sec: {prompt_tokens_per_second:.2f}")
            print(f"   └─ Generation tokens/sec: {tokens_per_second:.2f} ⚡")
            
            return {
                "model": model_name,
                "success": True,
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_duration": eval_duration,
                "prompt_tokens": prompt_eval_count,
                "generated_tokens": eval_count,
                "tokens_per_second": tokens_per_second,
                "prompt_tokens_per_second": prompt_tokens_per_second,
                "response": generated_text
            }
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            return {"model": model_name, "success": False, "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        print(f"❌ Error: Timeout (exceeded 300s)")
        return {"model": model_name, "success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"model": model_name, "success": False, "error": str(e)}


def run_benchmark():
    """Run benchmark on all available models."""
    print("\n" + "=" * 70)
    print("🚀 OLLAMA MODELS BENCHMARK")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    models = get_available_models()
    print(f"\n📋 Found {len(models)} models to test:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    results = []
    
    for model in models:
        result = test_model(model)
        results.append(result)
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Create summary table
    successful_results = [r for r in results if r.get('success')]
    
    if successful_results:
        # Sort by tokens per second (descending)
        successful_results.sort(key=lambda x: x.get('tokens_per_second', 0), reverse=True)
        
        print(f"\n{'Model':<50} {'Tok/s':>10} {'Gen Time':>10} {'Tokens':>8}")
        print("-" * 80)
        
        for r in successful_results:
            print(f"{r['model']:<50} {r['tokens_per_second']:>10.2f} {r['eval_duration']:>9.2f}s {r['generated_tokens']:>8}")
        
        print("-" * 80)
        
        # Find fastest model
        fastest = successful_results[0]
        print(f"\n🏆 Fastest model: {fastest['model']} ({fastest['tokens_per_second']:.2f} tokens/sec)")
    
    # Show failed models
    failed_results = [r for r in results if not r.get('success')]
    if failed_results:
        print(f"\n❌ Failed models ({len(failed_results)}):")
        for r in failed_results:
            print(f"   - {r['model']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n✅ Benchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    run_benchmark()