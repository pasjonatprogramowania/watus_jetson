# Optymalizacja Watus Jetson dla Jetson AGX Orin

## Wstęp

Ten dokument opisuje pełną migrację z Ollamy na natywne uruchamianie modeli LLM bezpośrednio na GPU Jetson AGX Orin z wykorzystaniem TensorRT-LLM. Celem jest maksymalna wydajność przy zachowaniu kompatybilności z API OpenAI.

**Dlaczego to robimy?**
- Ollama na Jetsonie działa na CPU lub przez emulację, tracimy 80%+ mocy GPU.
- TensorRT-LLM to silnik NVIDIA zoptymalizowany dla architektur Ampere/Ada (w tym Orin).
- `.engine` pliki są skompilowane specjalnie pod GPU Orina, dając 3-10x przyspieszenie.

---

## Część 1: Architektura Docelowa

```
┌─────────────────────────────────────────────────────────────────┐
│                        Jetson AGX Orin                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐     ┌──────────────────────────────────────┐ │
│  │  Watus Audio  │────▶│  OpenAI-Compatible API Server        │ │
│  │  (Python)     │     │  (TensorRT-LLM + FastAPI)            │ │
│  └───────────────┘     │                                      │ │
│                        │  POST /v1/chat/completions           │ │
│  ┌───────────────┐     │  POST /v1/completions                │ │
│  │  Watus LLM    │────▶│                                      │ │
│  │  (pydantic-ai)│     │  ┌──────────────────────────────────┐    │ │
│  └───────────────┘     │  │  TensorRT-LLM Engine         │    │ │
│                        │  │  (llama-7b.engine)           │    │ │
│                        │  │  GPU: 2048 CUDA + 64 Tensor  │    │ │
│                        │  └──────────────────────────────┘    │ │
│                        └──────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────┐     ┌──────────────────────────────────────┐ │
│  │  Watus Wizja  │────▶│  YOLO TensorRT (.engine)             │ │
│  │  (OpenCV)     │     │  + Klasyfikatory (.onnx/.engine)     │ │
│  └───────────────┘     └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Część 2: Przygotowanie Środowiska Jetson

### Zadanie 2.1: Weryfikacja JetPack
```bash
# Sprawdź wersję JetPack (wymagana >= 5.1)
cat /etc/nv_tegra_release
# lub
sudo apt-cache show nvidia-jetpack | grep Version

# Upewnij się, że masz zainstalowane:
# - CUDA 11.4+
# - cuDNN 8.6+
# - TensorRT 8.5+

# Sprawdź CUDA:
nvcc --version

# Sprawdź TensorRT:
dpkg -l | grep tensorrt
```

### Zadanie 2.2: Instalacja TensorRT-LLM
TensorRT-LLM to oficjalna biblioteka NVIDIA do uruchamiania LLM na TensorRT.

```bash
# Klonowanie repozytorium
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Dla Jetson ARM64 - kompilacja ze źródeł (może trwać 1-2h)
# UWAGA: Oficjalne wheel'e są dla x86_64, więc na Jetsonie musimy budować
pip install -r requirements.txt
python setup.py install

# Alternatywa: Kontener Docker (prostsze)
# NVIDIA ma gotowe obrazy NGC dla Jetson:
docker pull nvcr.io/nvidia/tensorrt:23.08-py3
```

### Zadanie 2.3: Pobranie Modelu Bazowego
Polecam Mistral 7B lub Llama 2 7B - optymalne dla 32GB RAM Orina.

```bash
# Użyj Hugging Face CLI
pip install huggingface-cli
huggingface-cli login  # Zaloguj się tokenem

# Pobierz model (przykład: Mistral 7B Instruct)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ./models/mistral-7b-instruct

# LUB dla oszczędności RAM - wersja GPTQ 4-bit:
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
    --local-dir ./models/mistral-7b-gptq
```

---

## Część 3: Konwersja Modelu do TensorRT

### Zadanie 3.1: Eksport do ONNX (opcjonalnie, do debugowania)
ONNX jest formatem pośrednim. TensorRT-LLM może od razu budować engine z checkpointów PyTorch, ale ONNX jest przydatny do inspekcji.

```bash
# Konwersja Hugging Face -> ONNX
python -m transformers.onnx \
    --model=./models/mistral-7b-instruct \
    --feature=causal-lm \
    ./models/mistral-7b.onnx

# Weryfikacja ONNX
python -c "import onnx; m = onnx.load('./models/mistral-7b.onnx'); onnx.checker.check_model(m)"
```

### Zadanie 3.2: Budowanie TensorRT Engine (Kluczowy Krok)
TensorRT-LLM buduje zoptymalizowany silnik `.engine` dla konkretnego GPU.

```bash
cd TensorRT-LLM/examples/llama  # lub mistral, zależnie od modelu

# Krok 1: Konwersja wag do formatu TensorRT-LLM
python convert_checkpoint.py \
    --model_dir ../../models/mistral-7b-instruct \
    --output_dir ./trt_ckpt \
    --dtype float16 \
    --tp_size 1  # Tensor Parallelism (1 dla jednego GPU)

# Krok 2: Budowanie Engine
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 4 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_beam_width 1

# Wynik: ./trt_engine/config.json + rank0.engine
```

**Uwagi dla Jetson AGX Orin:**
- `--max_batch_size 4` - Orin ma 32GB RAM, więc możemy obsłużyć 4 równoległe zapytania.
- `--dtype float16` lub `--dtype int8` dla kwantyzacji (mniejszy model, szybszy).
- Budowanie trwa 30-60 minut dla 7B modelu.

### Zadanie 3.3: Weryfikacja Engine
```bash
cd TensorRT-LLM/examples/llama

python run.py \
    --engine_dir ./trt_engine \
    --max_output_len 100 \
    --input_text "Cześć, jestem Watus. Kim jestem?"
```

---

## Część 4: Serwer OpenAI-Compatible API

Teraz musimy opakować TensorRT-LLM w serwer HTTP zgodny z API OpenAI, żeby `pydantic-ai` i inne frameworki mogły z niego korzystać bez zmian.

### Zadanie 4.1: Struktura Projektu Serwera
```
watus_jetson/
├── trt_server/
│   ├── __init__.py
│   ├── server.py          # FastAPI app
│   ├── engine_wrapper.py  # Wrapper dla TRT-LLM
│   └── config.py
```

### Zadanie 4.2: Implementacja Wrappera TensorRT-LLM
**Plik: `trt_server/engine_wrapper.py`**

```python
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
from tensorrt_llm.bindings import GptModelConfig
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
            end_id=2,  # EOS token ID (zależy od modelu)
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
        
        output_text = self.runner.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
```

### Zadanie 4.3: Implementacja Serwera FastAPI (OpenAI-Compatible)
**Plik: `trt_server/server.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import uuid

from engine_wrapper import TrtLlmEngine

app = FastAPI(title="Watus TRT-LLM Server (OpenAI Compatible)")

# Inicjalizacja silnika przy starcie
ENGINE_DIR = "/path/to/trt_engine"
engine = TrtLlmEngine(ENGINE_DIR)


# ===== Modele Pydantic (zgodne z OpenAI API) =====

class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-trt-llm"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False  # TODO: Streaming

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


# ===== Endpoint /v1/chat/completions =====

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest):
    # Budowanie promptu z historii wiadomości
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"[SYSTEM] {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"[USER] {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"[ASSISTANT] {msg.content}")
    
    prompt = "\n".join(prompt_parts) + "\n[ASSISTANT]"
    
    # Generowanie odpowiedzi
    output = engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    
    # Formatowanie odpowiedzi zgodnie z OpenAI API
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": len(prompt.split()),  # Uproszczone
            "completion_tokens": len(output.split()),
            "total_tokens": len(prompt.split()) + len(output.split()),
        }
    )


# ===== Endpoint /v1/models (lista modeli) =====

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "local-trt-llm",
                "object": "model",
                "created": 1700000000,
                "owned_by": "watus"
            }
        ]
    }


# ===== Health Check =====

@app.get("/health")
def health():
    return {"status": "ok", "backend": "TensorRT-LLM"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Zadanie 4.4: Uruchomienie Serwera
```bash
cd watus_jetson/trt_server
python server.py

# Serwer będzie dostępny pod:
# http://localhost:8080/v1/chat/completions
```

---

## Część 5: Integracja z Watus (pydantic-ai)

### Zadanie 5.1: Zmiana Endpointu w `warstwa_llm`
**Plik: `warstwa_llm/src/__init__.py` lub `config.py`**

```python
# Stara konfiguracja (Ollama):
# CURRENT_MODEL = "gemini:flash"
# LLM_BASE_URL = "http://localhost:11434"

# Nowa konfiguracja (TensorRT-LLM):
CURRENT_MODEL = "local-trt-llm"
LLM_BASE_URL = "http://localhost:8080"
```

### Zadanie 5.2: Konfiguracja pydantic-ai dla Custom Endpointu
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Utworzenie modelu wskazującego na lokalny serwer
local_model = OpenAIModel(
    model_name="local-trt-llm",
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # Nasz serwer nie wymaga klucza
)

agent = Agent(
    model=local_model,
    system_prompt="Jesteś pomocnym asystentem robota Watus.",
)

# Użycie bez zmian:
result = agent.run_sync("Co widzisz?")
print(result.output)
```

---

## Część 6: Optymalizacja Modeli Wizji (YOLO, Klasyfikatory)

### Zadanie 6.1: Konwersja YOLO do TensorRT
Ultralytics YOLO ma wbudowane wsparcie dla TensorRT!

```python
from ultralytics import YOLO

# Załaduj model PyTorch
model = YOLO("yolo12s.pt")

# Eksport do TensorRT (tworzy plik .engine)
model.export(
    format="engine",
    device=0,  # GPU
    half=True,  # FP16 dla Jetson
    imgsz=640,
    batch=1,
    workspace=4,  # GB RAM dla TRT
)

# Wynik: yolo12s.engine
```

### Zadanie 6.2: Użycie .engine w kodzie
```python
from ultralytics import YOLO

# Załaduj skompilowany silnik
model = YOLO("yolo12s.engine")

# Inferencja (bez zmian w API)
results = model.predict(frame, device=0)
```

### Zadanie 6.3: Konwersja Klasyfikatorów (gender, age, emotion)
Jeśli używasz modeli Hugging Face/PyTorch:

```bash
# Eksport PyTorch -> ONNX
python -c "
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained('your-classifier')
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'classifier.onnx', opset_version=13)
"

# ONNX -> TensorRT Engine
trtexec \
    --onnx=classifier.onnx \
    --saveEngine=classifier.engine \
    --fp16 \
    --workspace=2048
```

---

## Część 7: Zarządzanie Procesami na Jetsonie

### Zadanie 7.1: Systemd Service dla TRT Server
**Plik: `/etc/systemd/system/watus-trt-server.service`**

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

```bash
sudo systemctl daemon-reload
sudo systemctl enable watus-trt-server
sudo systemctl start watus-trt-server
```

### Zadanie 7.2: Monitoring GPU
```bash
# Tegra stats (dedykowane dla Jetson)
tegrastats

# Przykładowy output:
# RAM 8123/31927MB | CPU [12%,8%,5%,3%,2%,1%] | GPU 45% | ...
```

---

## Część 8: Checklist Migracji

| Krok | Status | Opis |
|------|--------|------|
| 1 | [ ] | Zainstaluj TensorRT-LLM na Jetsonie |
| 2 | [ ] | Pobierz model bazowy (Mistral 7B lub Llama 2 7B) |
| 3 | [ ] | Zbuduj `.engine` dla LLM |
| 4 | [ ] | Uruchom serwer OpenAI-compatible |
| 5 | [ ] | Zmień `warstwa_llm` na nowy endpoint |
| 6 | [ ] | Przetestuj end-to-end (audio -> LLM -> audio) |
| 7 | [ ] | Sklonuj model YOLO do `.engine` |
| 8 | [ ] | Podmień w `warstwa_wizji/main.py` ścieżkę do `.engine` |
| 9 | [ ] | Skonfiguruj systemd service |
| 10 | [ ] | Uruchom cały system i zmierz FPS/latency |

---

## Część 9: Oczekiwane Przyspieszenie

| Komponent | Przed (Ollama/PyTorch) | Po (TensorRT) | Przyspieszenie |
|-----------|------------------------|---------------|----------------|
| LLM (7B) First Token | ~2000ms | ~200ms | **10x** |
| LLM (7B) Token/s | ~5 tok/s | ~30 tok/s | **6x** |
| YOLO Inference | ~80ms/frame | ~15ms/frame | **5x** |
| Klasyfikatory | ~100ms | ~20ms | **5x** |

---

## Część 10: Troubleshooting

### Problem: "Out of memory" podczas budowania engine
- **Rozwiązanie:** Zmniejsz `--max_batch_size` lub użyj kwantyzacji INT8.

### Problem: Serwer nie odpowiada
- **Sprawdź logi:** `journalctl -u watus-trt-server -f`
- **Sprawdź GPU:** `tegrastats` - czy GPU jest w użyciu?

### Problem: pydantic-ai nie łączy się z serwerem
- **Sprawdź URL:** Czy `base_url` kończy się na `/v1`?
- **Sprawdź firewall:** `sudo ufw allow 8080/tcp`
