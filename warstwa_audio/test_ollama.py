#!/usr/bin/env python3
"""
Skrypt benchmarkowy do testowania wydajności modeli Ollama.
Mierzy liczbę tokenów na sekundę i czas odpowiedzi dla każdego modelu.
"""

import subprocess
import json
import time
import requests
from datetime import datetime


def get_available_models():
    """Pobiera listę dostępnych modeli z ollama."""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[1:]  # Pomiń nagłówek
    models = []
    for line in lines:
        if line.strip():
            model_name = line.split()[0]
            models.append(model_name)
    return models


def test_model(model_name: str, prompt: str = "Napisz krótką bajkę o kocie w 3 zdaniach."):
    """
    Testuje pojedynczy model i zwraca metryki wydajności.
    Używa API Ollama do pobrania szczegółowych statystyk tokenów.
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"\n{'='*60}")
    print(f"Testowanie modelu: {model_name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            # Wyodrębnij metryki z odpowiedzi Ollama
            total_duration = data.get('total_duration', 0) / 1e9  # Konwersja nanosekund na sekundy
            load_duration = data.get('load_duration', 0) / 1e9
            prompt_eval_duration = data.get('prompt_eval_duration', 0) / 1e9
            eval_duration = data.get('eval_duration', 0) / 1e9
            
            prompt_eval_count = data.get('prompt_eval_count', 0)
            eval_count = data.get('eval_count', 0)
            
            # Oblicz tokeny na sekundę
            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0
            prompt_tokens_per_second = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
            
            generated_text = data.get('response', '')
            
            print(f"\n📝 Odpowiedź:\n{generated_text}")
            print(f"\n📊 Metryki Wydajności:")
            print(f"   ├─ Całkowity czas: {total_duration:.2f}s")
            print(f"   ├─ Czas ładowania modelu: {load_duration:.2f}s")
            print(f"   ├─ Przetwarzanie promptu: {prompt_eval_duration:.2f}s ({prompt_eval_count} tokenów)")
            print(f"   ├─ Czas generowania: {eval_duration:.2f}s ({eval_count} tokenów)")
            print(f"   ├─ Tokeny promptu/sek: {prompt_tokens_per_second:.2f}")
            print(f"   └─ Tokeny generowania/sek: {tokens_per_second:.2f} ⚡")
            
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
            print(f"❌ Błąd: HTTP {response.status_code}")
            return {"model": model_name, "success": False, "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        print(f"❌ Błąd: Upłynął limit czasu (przekroczono 300s)")
        return {"model": model_name, "success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ Błąd: {str(e)}")
        return {"model": model_name, "success": False, "error": str(e)}


def run_benchmark():
    """Uruchamia benchmark na wszystkich dostępnych modelach."""
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK MODELI OLLAMA")
    print(f"   Rozpoczęto: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    models = get_available_models()
    print(f"\n📋 Znaleziono {len(models)} modeli do przetestowania:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    results = []
    
    for model in models:
        result = test_model(model)
        results.append(result)
    
    # Wyświetl podsumowanie
    print("\n\n" + "=" * 70)
    print("📈 PODSUMOWANIE BENCHMARKU")
    print("=" * 70)
    
    # Utwórz tabelę podsumowującą
    successful_results = [r for r in results if r.get('success')]
    
    if successful_results:
        # Sortuj według tokenów na sekundę (malejąco)
        successful_results.sort(key=lambda x: x.get('tokens_per_second', 0), reverse=True)
        
        print(f"\n{'Model':<50} {'Tok/s':>10} {'Czas Gen':>10} {'Tokeny':>8}")
        print("-" * 80)
        
        for r in successful_results:
            print(f"{r['model']:<50} {r['tokens_per_second']:>10.2f} {r['eval_duration']:>9.2f}s {r['generated_tokens']:>8}")
        
        print("-" * 80)
        
        # Znajdź najszybszy model
        fastest = successful_results[0]
        print(f"\n🏆 Najszybszy model: {fastest['model']} ({fastest['tokens_per_second']:.2f} tokenów/sek)")
    
    # Pokaż modele, które zawiodły
    failed_results = [r for r in results if not r.get('success')]
    if failed_results:
        print(f"\n❌ Modele z błędami ({len(failed_results)}):")
        for r in failed_results:
            print(f"   - {r['model']}: {r.get('error', 'Nieznany błąd')}")
    
    print(f"\n✅ Benchmark zakończony: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    run_benchmark()
