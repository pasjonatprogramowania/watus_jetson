"""
Test sprawdzajacy czy Ollama dziala poprawnie i uzywa GPU.

Weryfikuje:
1. Czy serwer Ollama jest uruchomiony
2. Jakie modele sa dostepne
3. Czy model generuje odpowiedzi
4. Czy model dziala na GPU (ollama ps -> PROCESSOR)
5. Czy integracja z pydantic-ai (OpenAIModel) dziala
"""

import subprocess
import sys
import requests
import time
import json


OLLAMA_API = "http://localhost:11434"


def check_ollama_running():
    """Sprawdza czy serwer Ollama jest uruchomiony."""
    print("[1/5] Sprawdzanie czy Ollama jest uruchomiona...")
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            print(f"  [OK] Ollama dziala. Znaleziono {len(models)} modeli:")
            for m in models:
                name = m.get("name", "?")
                size_gb = m.get("size", 0) / 1e9
                print(f"    - {name} ({size_gb:.1f} GB)")
            return models
        else:
            print(f"  [BLAD] Ollama zwrocila status {r.status_code}")
            return None
    except requests.ConnectionError:
        print("  [BLAD] Nie mozna polaczyc z Ollama. Uruchom ja komenda: ollama serve")
        return None
    except Exception as e:
        print(f"  [BLAD] {e}")
        return None


def test_generation(model_name):
    """Testuje generowanie odpowiedzi przez model."""
    print(f"\n[2/5] Testowanie generowania ({model_name})...")
    url = f"{OLLAMA_API}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "Odpowiedz jednym zdaniem: co to jest Python?",
        "stream": False
    }

    try:
        start = time.time()
        r = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start

        if r.status_code == 200:
            data = r.json()
            response_text = data.get("response", "")
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 1) / 1e9  # ns -> s
            tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0

            print(f"  [OK] Odpowiedz: {response_text[:200]}")
            print(f"  Czas: {elapsed:.2f}s | Tokeny: {eval_count} | Predkosc: {tokens_per_sec:.1f} tok/s")
            return True
        else:
            print(f"  [BLAD] HTTP {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  [BLAD] {e}")
        return False


def check_gpu_usage():
    """Sprawdza czy Ollama uzywa GPU (ollama ps)."""
    print("\n[3/5] Sprawdzanie uzycia GPU (ollama ps)...")
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        print(f"  Wyjscie ollama ps:")
        for line in output.split("\n"):
            print(f"    {line}")

        # Sprawdz kolumne PROCESSOR
        lines = output.split("\n")
        if len(lines) <= 1:
            print("  [UWAGA] Brak zaladowanych modeli. Model moze byc juz wyladowany z pamieci.")
            print("  Uruchom ponownie generowanie i szybko sprawdz ollama ps.")
            return None

        output_lower = output.lower()
        if "gpu" in output_lower:
            print("  [OK] Model dziala na GPU!")
            return True
        elif "cpu" in output_lower:
            print("  [UWAGA] Model dziala na CPU, NIE na GPU!")
            print("  Mozliwe przyczyny:")
            print("    - Brak sterownikow CUDA")
            print("    - Model zbyt duzy na VRAM GPU")
            print("    - Ollama nie wykrywa GPU")
            print("  Sprawdz: nvidia-smi oraz ollama show <model> --json")
            return False
        else:
            print("  [?] Nie udalo sie jednoznacznie okreslic (GPU/CPU)")
            return None
    except FileNotFoundError:
        print("  [BLAD] Komenda 'ollama' nie znaleziona w PATH")
        return None
    except Exception as e:
        print(f"  [BLAD] {e}")
        return None


def check_nvidia_smi():
    """Sprawdza nvidia-smi dla dodatkowej weryfikacji GPU."""
    print("\n[4/5] Sprawdzanie nvidia-smi...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"  [OK] GPU info: {output}")
            return True
        else:
            print(f"  [UWAGA] nvidia-smi zwrocilo blad: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  [UWAGA] nvidia-smi nie znaleziono — brak sterownikow NVIDIA lub GPU")
        return False
    except Exception as e:
        print(f"  [BLAD] {e}")
        return False


def test_pydantic_ai_integration(model_name):
    """Testuje integracje z pydantic-ai przez OpenAI-compatible endpoint."""
    print(f"\n[5/5] Testowanie integracji pydantic-ai z Ollama ({model_name})...")
    try:
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai import Agent

        provider = OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama")
        model = OpenAIModel(model_name, provider=provider)

        agent = Agent(model=model, output_type=str, system_prompt="Odpowiadaj krotko po polsku.")

        start = time.time()
        result = agent.run_sync("Co to jest GPU? Odpowiedz w jednym zdaniu.")
        elapsed = time.time() - start

        print(f"  [OK] pydantic-ai odpowiedz: {result.output[:200]}")
        print(f"  Czas: {elapsed:.2f}s")
        return True
    except ImportError as e:
        print(f"  [BLAD] Brak biblioteki: {e}")
        print("  Zainstaluj: pip install pydantic-ai")
        return False
    except Exception as e:
        print(f"  [BLAD] Integracja pydantic-ai nie powiodla sie: {e}")
        return False


def main():
    print("=" * 60)
    print("TEST OLLAMA — GPU VERIFICATION")
    print("=" * 60)

    # 1. Sprawdz Ollama
    models = check_ollama_running()
    if not models:
        print("\n[KONIEC] Ollama nie jest uruchomiona. Nie mozna kontynuowac.")
        sys.exit(1)

    # Wybierz pierwszy dostepny model
    model_name = models[0]["name"]
    print(f"\nWybrany model do testow: {model_name}")

    # 2. Test generowania (to zaladuje model do pamieci)
    gen_ok = test_generation(model_name)
    if not gen_ok:
        print("\n[KONIEC] Generowanie nie powiodlo sie.")
        sys.exit(1)

    # 3. Sprawdz GPU (zaraz po generowaniu, bo model jest w pamieci)
    gpu_ok = check_gpu_usage()

    # 4. nvidia-smi
    nvidia_ok = check_nvidia_smi()

    # 5. Test pydantic-ai
    pai_ok = test_pydantic_ai_integration(model_name)

    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"  Ollama uruchomiona:    OK")
    print(f"  Generowanie:           {'OK' if gen_ok else 'BLAD'}")
    print(f"  GPU (ollama ps):       {'OK' if gpu_ok else 'BLAD/NIEZNANE' if gpu_ok is None else 'CPU (nie GPU!)'}")
    print(f"  nvidia-smi:            {'OK' if nvidia_ok else 'NIEDOSTEPNE'}")
    print(f"  pydantic-ai:           {'OK' if pai_ok else 'BLAD'}")

    if gpu_ok and gen_ok and pai_ok:
        print("\n[SUKCES] Ollama dziala na GPU i integracja z pydantic-ai jest poprawna!")
    elif gen_ok and pai_ok and gpu_ok is None:
        print("\n[CZESCIOWY SUKCES] Generowanie i pydantic-ai dzialaja, ale GPU nie potwierdzone.")
    else:
        print("\n[UWAGA] Niektore testy nie przeszly — sprawdz logi powyzej.")


if __name__ == "__main__":
    main()
