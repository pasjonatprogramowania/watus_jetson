# Optymalizacja WATUS Jetson - Instrukcja Migracji na TensorRT-LLM

## Spis Tresci
1. [Wprowadzenie](#wprowadzenie)
2. [Architektura Docelowa](#architektura-docelowa)
3. [Przygotowanie Srodowiska Jetson](#przygotowanie-srodowiska-jetson)
4. [Konwersja Modelu do TensorRT](#konwersja-modelu-do-tensorrt)
5. [Serwer OpenAI-Compatible API](#serwer-openai-compatible-api)
6. [Integracja z Watus](#integracja-z-watus)
7. [Optymalizacja Modeli Wizji](#optymalizacja-modeli-wizji)
8. [Zarzadzanie Procesami](#zarzadzanie-procesami)
9. [Checklist Migracji](#checklist-migracji)
10. [Rozwiazywanie Problemow](#rozwiazywanie-problemow)

---

## Wprowadzenie

### Cel Dokumentu
Ten dokument opisuje pelna migracje z Ollamy na natywne uruchamianie modeli LLM bezposrednio na GPU Jetson AGX Orin z wykorzystaniem TensorRT-LLM.

### Dlaczego Migracja?

| Problem z Ollama | Rozwiazanie TensorRT-LLM |
|------------------|--------------------------|
| Ollama dziala na CPU lub przez emulacje | Natywne wykonanie na GPU |
| Tracimy 80%+ mocy GPU | Pelne wykorzystanie GPU |
| Wolna inferencja (~5 tok/s) | Szybka inferencja (~30 tok/s) |
| Brak optymalizacji dla Jetson | Zoptymalizowane dla Ampere/Ada |

### Oczekiwane Przyspieszenie

| Komponent | Przed (Ollama/PyTorch) | Po (TensorRT) | Przyspieszenie |
|-----------|------------------------|---------------|----------------|
| LLM (7B) First Token | ~2000ms | ~200ms | 10x |
| LLM (7B) Token/s | ~5 tok/s | ~30 tok/s | 6x |
| YOLO Inference | ~80ms/frame | ~15ms/frame | 5x |
| Klasyfikatory | ~100ms | ~20ms | 5x |

---

## Architektura Docelowa

### Diagram Systemu
```
+---------------------------------------------------------------------+
|                        Jetson AGX Orin                               |
+---------------------------------------------------------------------+
|                                                                      |
|  +---------------+     +----------------------------------------+    |
|  |  Watus Audio  |---->|  OpenAI-Compatible API Server          |    |
|  |  (Python)     |     |  (TensorRT-LLM + FastAPI)              |    |
|  +---------------+     |                                        |    |
|                        |  POST /v1/chat/completions             |    |
|  +---------------+     |  POST /v1/completions                  |    |
|  |  Watus LLM    |---->|                                        |    |
|  |  (pydantic-ai)|     |  +----------------------------------+  |    |
|  +---------------+     |  |  TensorRT-LLM Engine             |  |    |
|                        |  |  (llama-7b.engine)               |  |    |
|                        |  |  GPU: 2048 CUDA + 64 Tensor      |  |    |
|                        |  +----------------------------------+  |    |
|                        +----------------------------------------+    |
|                                                                      |
|  +---------------+     +----------------------------------------+    |
|  |  Watus Wizja  |---->|  YOLO TensorRT (.engine)               |    |
|  |  (OpenCV)     |     |  + Klasyfikatory (.onnx/.engine)       |    |
|  +---------------+     +----------------------------------------+    |
+---------------------------------------------------------------------+
```

### Komponenty
- **Watus Audio** - Modul audio (STT/TTS) wysyla zapytania do serwera LLM
- **Watus LLM** - Modul pydantic-ai laczy sie z lokalnym serwerem OpenAI-compatible
- **Watus Wizja** - Modul wizji uzywa YOLO w formacie TensorRT .engine
- **TRT Server** - Serwer FastAPI opakowujacy TensorRT-LLM

---

## Przygotowanie Srodowiska Jetson

### Krok 1: Weryfikacja JetPack

**Czynnosci:**
1. Zaloguj sie na Jetson przez SSH lub terminal
2. Sprawdz wersje JetPack:

```bash
cat /etc/nv_tegra_release
```

**Oczekiwany wynik:**
```
# R35 (release), REVISION: 4.1, ...
```

3. Sprawdz szczegolowa wersje:
```bash
sudo apt-cache show nvidia-jetpack | grep Version
```

**Oczekiwany wynik:**
```
Version: 5.1.2-xxx
```

**Wymagania minimalne:**
- JetPack >= 5.1
- CUDA >= 11.4
- cuDNN >= 8.6
- TensorRT >= 8.5

### Krok 2: Weryfikacja CUDA

**Uruchomienie:**
```bash
nvcc --version
```

**Oczekiwany wynik:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 11.4, V11.4.xxx
```

### Krok 3: Weryfikacja TensorRT

**Uruchomienie:**
```bash
dpkg -l | grep tensorrt
```

**Oczekiwany wynik:**
```
ii  libnvinfer8                         8.5.x.x-1+cuda11.4   arm64   ...
ii  tensorrt                            8.5.x.x-1+cuda11.4   arm64   ...
```

### Krok 4: Instalacja TensorRT-LLM

**Metoda 1: Kompilacja ze zrodel (zalecana dla Jetson)**

```bash
# Krok 4.1: Klonowanie repozytorium
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Krok 4.2: Instalacja zaleznosci
pip install -r requirements.txt

# Krok 4.3: Kompilacja (trwa 1-2h)
python setup.py install
```

**Metoda 2: Docker (alternatywa)**

```bash
docker pull nvcr.io/nvidia/tensorrt:23.08-py3
```

**Weryfikacja instalacji:**
```bash
python -c "import tensorrt_llm; print('TRT-LLM OK')"
```

**Oczekiwany wynik:**
```
TRT-LLM OK
```

### Krok 5: Pobranie Modelu Bazowego

**Zalecane modele dla Jetson AGX Orin (32GB RAM):**
- Mistral 7B Instruct
- Llama 2 7B
- Phi-2 (mniejszy, szybszy)

**Uruchomienie:**
```bash
# Krok 5.1: Instalacja Hugging Face CLI
pip install huggingface-cli

# Krok 5.2: Logowanie
huggingface-cli login

# Krok 5.3: Pobranie modelu (Mistral 7B)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ./models/mistral-7b-instruct
```

**Oczekiwany wynik:**
Folder `./models/mistral-7b-instruct` zawierajacy pliki modelu (~14GB).

**Alternatywa dla mniejszego RAM - wersja GPTQ:**
```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
    --local-dir ./models/mistral-7b-gptq
```

---

## Konwersja Modelu do TensorRT

### Krok 1: Konwersja Wag

**Lokalizacja:** Wewnatrz katalogu TensorRT-LLM

```bash
cd TensorRT-LLM/examples/llama
```

**Uruchomienie:**
```bash
python convert_checkpoint.py \
    --model_dir ../../models/mistral-7b-instruct \
    --output_dir ./trt_ckpt \
    --dtype float16 \
    --tp_size 1
```

**Parametry:**
| Parametr | Wartosc | Opis |
|----------|---------|------|
| --model_dir | sciezka | Folder z modelem HuggingFace |
| --output_dir | ./trt_ckpt | Folder wyjsciowy |
| --dtype | float16 | Precyzja (float16 dla Jetson) |
| --tp_size | 1 | Tensor Parallelism (1 GPU = 1) |

**Oczekiwany wynik:**
Folder `./trt_ckpt` z plikami checkpointu.

### Krok 2: Budowanie Engine

**Uruchomienie:**
```bash
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 4 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_beam_width 1
```

**Parametry:**
| Parametr | Wartosc | Opis |
|----------|---------|------|
| --max_batch_size | 4 | Rownoczesne zapytania (4 dla 32GB RAM) |
| --max_input_len | 2048 | Max tokenow wejsciowych |
| --max_output_len | 512 | Max tokenow wyjsciowych |
| --max_beam_width | 1 | Beam search (1 = greedy) |

**Czas budowania:** 30-60 minut dla modelu 7B

**Oczekiwany wynik:**
```
./trt_engine/
|-- config.json
|-- rank0.engine
```

### Krok 3: Weryfikacja Engine

**Uruchomienie:**
```bash
python run.py \
    --engine_dir ./trt_engine \
    --max_output_len 100 \
    --input_text "Czesc, jestem Watus. Kim jestem?"
```

**Oczekiwany wynik:**
Model generuje odpowiedz na pytanie w terminalu.

---

## Serwer OpenAI-Compatible API

### Struktura Serwera

**Pliki do utworzenia:**
```
watus_jetson/
|-- trt_server/
    |-- __init__.py
    |-- server.py
    |-- engine_wrapper.py
    |-- config.py
```

### Krok 1: Utworzenie Wrappera Engine

**Plik:** `trt_server/engine_wrapper.py`

```python
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import torch

class TrtLlmEngine:
    def __init__(self, engine_dir: str):
        self.runner = ModelRunner.from_dir(engine_dir, rank=0)
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        sampling_config = SamplingConfig(
            end_id=2,
            pad_id=0,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        input_ids = self.runner.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = self.runner.generate(
                input_ids,
                sampling_config=sampling_config,
            )
        
        output_text = self.runner.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return output_text
```

### Krok 2: Utworzenie Serwera FastAPI

**Plik:** `trt_server/server.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import uuid

from engine_wrapper import TrtLlmEngine

app = FastAPI(title="Watus TRT-LLM Server")

ENGINE_DIR = "/path/to/trt_engine"  # ZMIEN NA WLASCIWA SCIEZKE
engine = TrtLlmEngine(ENGINE_DIR)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "local-trt-llm"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.7

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    prompt = "\n".join([
        f"[{m.role.upper()}] {m.content}" 
        for m in request.messages
    ]) + "\n[ASSISTANT]"
    
    output = engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output.split()),
            "total_tokens": len(prompt.split()) + len(output.split())
        }
    )

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "local-trt-llm", "object": "model"}]}

@app.get("/health")
def health():
    return {"status": "ok", "backend": "TensorRT-LLM"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Krok 3: Uruchomienie Serwera

**Uruchomienie:**
```bash
cd watus_jetson/trt_server
python server.py
```

**Oczekiwany wynik:**
```
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Krok 4: Test Serwera

**Uruchomienie (w nowym terminalu):**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Czesc!"}],
    "max_tokens": 50
  }'
```

**Oczekiwany wynik:**
```json
{
  "id": "chatcmpl-xxxxxxxx",
  "choices": [{"message": {"content": "Czesc! Jak moge pomoc?"}}],
  ...
}
```

---

## Integracja z Watus

### Krok 1: Zmiana Konfiguracji warstwa_llm

**Plik do edycji:** `warstwa_llm/src/config.py`

**Zmiany:**
```python
# Stara konfiguracja (zakomentuj):
# CURRENT_MODEL = "gemini:flash"
# LLM_BASE_URL = "http://localhost:11434"

# Nowa konfiguracja:
CURRENT_MODEL = "local-trt-llm"
LLM_BASE_URL = "http://localhost:8080"
```

### Krok 2: Konfiguracja pydantic-ai

**Przyklad uzycia:**
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

local_model = OpenAIModel(
    model_name="local-trt-llm",
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

agent = Agent(
    model=local_model,
    system_prompt="Jestes Watus - pomocny robot.",
)

result = agent.run_sync("Co widzisz?")
print(result.output)
```

---

## Optymalizacja Modeli Wizji

### Krok 1: Konwersja YOLO do TensorRT

**Uruchomienie:**
```python
from ultralytics import YOLO

model = YOLO("yolo12s.pt")

model.export(
    format="engine",
    device=0,
    half=True,
    imgsz=640,
    batch=1,
    workspace=4,
)
```

**Oczekiwany wynik:**
Plik `yolo12s.engine` w tym samym folderze.

### Krok 2: Uzycie Engine w Kodzie

**Zmiana w warstwa_wizji:**
```python
from ultralytics import YOLO

# Stare (zakomentuj):
# model = YOLO("yolo12s.pt")

# Nowe:
model = YOLO("yolo12s.engine")

results = model.predict(frame, device=0)
```

---

## Zarzadzanie Procesami

### Konfiguracja Systemd

**Plik:** `/etc/systemd/system/watus-trt-server.service`

```ini
[Unit]
Description=Watus TensorRT-LLM Server
After=network.target

[Service]
Type=simple
User=watus
WorkingDirectory=/home/watus/watus_jetson/trt_server
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=5
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

**Aktywacja:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable watus-trt-server
sudo systemctl start watus-trt-server
```

**Sprawdzenie statusu:**
```bash
sudo systemctl status watus-trt-server
```

### Monitoring GPU

**Uruchomienie:**
```bash
tegrastats
```

**Oczekiwany wynik:**
```
RAM 8123/31927MB | CPU [12%,8%,5%] | GPU 45% | ...
```

---

## Checklist Migracji

| Krok | Opis | Status |
|------|------|--------|
| 1 | Zainstaluj TensorRT-LLM na Jetsonie | [ ] |
| 2 | Pobierz model bazowy (Mistral 7B lub Llama 2 7B) | [ ] |
| 3 | Zbuduj .engine dla LLM | [ ] |
| 4 | Uruchom serwer OpenAI-compatible | [ ] |
| 5 | Zmien warstwa_llm na nowy endpoint | [ ] |
| 6 | Przetestuj end-to-end (audio -> LLM -> audio) | [ ] |
| 7 | Sklonuj model YOLO do .engine | [ ] |
| 8 | Podmien w warstwa_wizji/main.py sciezke do .engine | [ ] |
| 9 | Skonfiguruj systemd service | [ ] |
| 10 | Uruchom caly system i zmierz FPS/latency | [ ] |

---

## Rozwiazywanie Problemow

### Problem: "Out of memory" podczas budowania engine

**Przyczyna:** Za duzy batch size lub model

**Rozwiazanie:**
1. Zmniejsz `--max_batch_size` do 2 lub 1
2. Uzyj kwantyzacji INT8: `--dtype int8`
3. Zamknij inne aplikacje GPU

### Problem: Serwer nie odpowiada

**Diagnostyka:**
```bash
journalctl -u watus-trt-server -f
```

**Rozwiazania:**
1. Sprawdz czy port 8080 jest wolny
2. Sprawdz czy engine zaladowal sie poprawnie
3. Sprawdz logi bledu

### Problem: pydantic-ai nie laczy sie z serwerem

**Przyczyna:** Zly URL lub firewall

**Rozwiazania:**
1. Sprawdz czy base_url konczy sie na `/v1`
2. Otworz port: `sudo ufw allow 8080/tcp`
3. Sprawdz czy serwer dziala: `curl http://localhost:8080/health`

### Problem: Wolna inferencja mimo TensorRT

**Przyczyna:** Model nie uzywa GPU

**Diagnostyka:**
```bash
tegrastats
```

**Rozwiazania:**
1. Sprawdz czy GPU% > 0 podczas inferencji
2. Sprawdz CUDA_VISIBLE_DEVICES
3. Przebuduj engine z poprawnymi flagami
