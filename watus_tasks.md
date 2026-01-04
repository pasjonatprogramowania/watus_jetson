# Plan Unifikacji Systemu WATUS Jetson - Instrukcja Implementacji

## Spis Tresci
1. [Wprowadzenie](#wprowadzenie)
2. [Modul 0: Fundamenty (watus_core)](#modul-0-fundamenty-watus_core)
3. [Modul 1: Lidar (Migracja na ZMQ)](#modul-1-lidar-migracja-na-zmq)
4. [Modul 2: Wizja (Migracja na ZMQ)](#modul-2-wizja-migracja-na-zmq)
5. [Modul 3: Audio (Unifikacja)](#modul-3-audio-unifikacja)
6. [Modul 4: Consolidator (Mozg Operacyjny)](#modul-4-consolidator-mozg-operacyjny)
7. [Modul 5: Warstwa LLM (Brain)](#modul-5-warstwa-llm-brain)
8. [Modul 6: Zarzadzanie Procesami](#modul-6-zarzadzanie-procesami)
9. [Protokol Komunikacji ZMQ](#protokol-komunikacji-zmq)
10. [Checklist Implementacji](#checklist-implementacji)

---

## Wprowadzenie

### Opis Problemu
Obecny system WATUS Jetson dziala, ale jest "posklejany" z roznych kawalkow:
- Moduly komunikuja sie przez pliki JSON na dysku (wolne, zawodne)
- Kazdy modul ma wlasna konfiguracje w roznych miejscach
- Brak centralnego loggera - wszedzie `print()`
- Trudne debugowanie i utrzymanie

### Cel Unifikacji
Stworzenie jednolitej architektury z:
- Centralnym modulem konfiguracji (watus_core)
- Komunikacja przez ZeroMQ (szybka, niezawodna)
- Wspolny logger
- Latwe zarzadzanie procesami

### Architektura Docelowa

```
+------------------------------------------------------------------+
|                         watus_core/                               |
|  +------------+  +------------+  +------------+                   |
|  | config.py  |  | logger.py  |  | comm.py    |                   |
|  | (.env)     |  | (logging)  |  | (ZMQ)      |                   |
|  +------------+  +------------+  +------------+                   |
+------------------------------------------------------------------+
        |                |                |
        v                v                v
+-------------+  +-------------+  +-------------+  +-------------+
|   lidar/    |  | warstwa_    |  | warstwa_    |  | warstwa_    |
|             |  | wizji/      |  | audio/      |  | llm/        |
+-------------+  +-------------+  +-------------+  +-------------+
        |                |                               |
        v                v                               v
+------------------------------------------------------------------+
|                      consolidator/                                |
|              (laczy dane, publikuje world.state)                  |
+------------------------------------------------------------------+
```

---

## Modul 0: Fundamenty (watus_core)

### Cel
Utworzenie wspolnego modulu z konfiguacja, loggerem i komunikacja ZMQ.

### Krok 0.1: Struktura Katalogow

**Czynnosci:**
1. Utworz folder `watus_core` w glownym katalogu projektu:
```bash
mkdir c:\Users\pawel\Documents\GitHub\watus_jetson\watus_core
```

2. Utworz pusty plik `__init__.py`:
```bash
echo. > c:\Users\pawel\Documents\GitHub\watus_jetson\watus_core\__init__.py
```

**Oczekiwany wynik:**
```
watus_jetson/
|-- watus_core/
    |-- __init__.py
```

### Krok 0.2: Centralna Konfiguracja

**Plik do utworzenia:** `watus_core/config.py`

**Zawartosc pliku:**
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Znajdz katalog glowny projektu (tam gdzie .git)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ENV_FILE = PROJECT_ROOT / ".env"

# Zaladuj zmienne srodowiskowe
load_dotenv(ENV_FILE)

class Config:
    """Centralna konfiguracja systemu WATUS."""
    
    # === Porty ZMQ ===
    ZMQ_PORT_LIDAR = int(os.getenv("ZMQ_PORT_LIDAR", "5555"))
    ZMQ_PORT_VISION = int(os.getenv("ZMQ_PORT_VISION", "5556"))
    ZMQ_PORT_AUDIO = int(os.getenv("ZMQ_PORT_AUDIO", "5557"))
    ZMQ_PORT_WORLD = int(os.getenv("ZMQ_PORT_WORLD", "5558"))
    ZMQ_PORT_ACTIONS = int(os.getenv("ZMQ_PORT_ACTIONS", "5559"))
    
    # === Sciezki do modeli ===
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "warstwa_wizji/models/yolo12s.pt")
    WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "models/faster-whisper-medium")
    
    # === Urzadzenia ===
    CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
    LIDAR_PORT = os.getenv("LIDAR_PORT", "/dev/ttyUSB0")
    AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", None)
    AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", None)
    
    # === Flagi debugowania ===
    SAVE_SESSION_LOGS = os.getenv("SAVE_SESSION_LOGS", "false").lower() == "true"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


# Singleton dla latwego dostępu
config = Config()
```

**Weryfikacja:**
Utwórz plik testowy `watus_core/test_config.py`:
```python
from config import config

print(f"ZMQ_PORT_LIDAR: {config.ZMQ_PORT_LIDAR}")
print(f"LIDAR_PORT: {config.LIDAR_PORT}")
print("Config zaladowany pomyslnie!")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\watus_core
python test_config.py
```

**Oczekiwany wynik:**
```
ZMQ_PORT_LIDAR: 5555
LIDAR_PORT: /dev/ttyUSB0
Config zaladowany pomyslnie!
```

### Krok 0.3: Wspolny Logger

**Plik do utworzenia:** `watus_core/logger.py`

**Zawartosc pliku:**
```python
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Katalog logow
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Zwraca skonfigurowany logger.
    
    Uzycie:
        from watus_core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Wiadomosc")
    """
    logger = logging.getLogger(name)
    
    # Unikaj wielokrotnego dodawania handlerow
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler 1: Konsola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler 2: Plik rotowany
    file_handler = RotatingFileHandler(
        LOG_DIR / "watus_system.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
```

**Weryfikacja:**
```python
from logger import get_logger

logger = get_logger("test")
logger.info("Test loggera")
logger.debug("Debug message")
logger.error("Error message")
```

**Oczekiwany wynik:**
- W konsoli: wiadomosci INFO i ERROR
- W pliku `logs/watus_system.log`: wszystkie wiadomosci

### Krok 0.4: Komunikacja ZMQ

**Plik do utworzenia:** `watus_core/comm.py`

**Zawartosc pliku:**
```python
import json
import time
import zmq
from typing import Tuple, Optional, Any

class Publisher:
    """
    Publisher ZMQ do wysylania danych.
    
    Uzycie:
        pub = Publisher(port=5555)
        pub.send("sensor.lidar", {"tracks": [...]})
    """
    
    def __init__(self, port: int, bind: bool = True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        addr = f"tcp://127.0.0.1:{port}"
        if bind:
            self.socket.bind(addr)
        else:
            self.socket.connect(addr)
    
    def send(self, topic: str, data: dict) -> None:
        """Wysyla dane z timestampem."""
        data["ts"] = time.time()
        json_bytes = json.dumps(data).encode("utf-8")
        self.socket.send_multipart([topic.encode("utf-8"), json_bytes])
    
    def close(self) -> None:
        self.socket.close()
        self.context.term()


class Subscriber:
    """
    Subscriber ZMQ do odbierania danych.
    
    Uzycie:
        sub = Subscriber(ports=[5555, 5556])
        topic, data = sub.recv()
    """
    
    def __init__(self, ports: list, topics: list = None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        for port in ports:
            self.socket.connect(f"tcp://127.0.0.1:{port}")
        
        # Subskrybuj tematy (pusty = wszystkie)
        if topics:
            for topic in topics:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        else:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    def recv(self, timeout_ms: int = 100) -> Optional[Tuple[str, dict]]:
        """Odbiera dane z timeoutem. Zwraca None jesli brak danych."""
        if self.socket.poll(timeout_ms):
            topic, json_bytes = self.socket.recv_multipart()
            data = json.loads(json_bytes.decode("utf-8"))
            return topic.decode("utf-8"), data
        return None
    
    def close(self) -> None:
        self.socket.close()
        self.context.term()
```

**Weryfikacja - utworz 2 pliki testowe:**

**Plik 1:** `watus_core/test_publisher.py`
```python
import time
from comm import Publisher

pub = Publisher(port=5555)
print("Publisher uruchomiony na porcie 5555")

for i in range(10):
    pub.send("test.message", {"count": i, "text": f"Wiadomosc {i}"})
    print(f"Wyslano: {i}")
    time.sleep(1)

pub.close()
```

**Plik 2:** `watus_core/test_subscriber.py`
```python
from comm import Subscriber

sub = Subscriber(ports=[5555])
print("Subscriber polaczony, czekam na wiadomosci...")

while True:
    result = sub.recv(timeout_ms=1000)
    if result:
        topic, data = result
        print(f"Odebrano [{topic}]: {data}")
    else:
        print("Brak wiadomosci...")
```

**Uruchomienie (2 terminale):**

Terminal 1:
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\watus_core
python test_subscriber.py
```

Terminal 2:
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\watus_core
python test_publisher.py
```

**Oczekiwany wynik (Terminal 1):**
```
Subscriber polaczony, czekam na wiadomosci...
Odebrano [test.message]: {'count': 0, 'text': 'Wiadomosc 0', 'ts': ...}
Odebrano [test.message]: {'count': 1, 'text': 'Wiadomosc 1', 'ts': ...}
...
```

---

## Modul 1: Lidar (Migracja na ZMQ)

### Cel
Zastapienie zapisu do pliku `lidar.json` publikacja przez ZMQ.

### Krok 1.1: Modyfikacja run_live.py

**Plik do edycji:** `lidar/src/run_live.py`

**Zmiany do wprowadzenia:**

1. Dodaj importy na poczatku pliku:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from watus_core.comm import Publisher
from watus_core.config import config
from watus_core.logger import get_logger

logger = get_logger("lidar")
```

2. W funkcji `main()` dodaj inicjalizacje publishera:
```python
def main():
    pub = Publisher(port=config.ZMQ_PORT_LIDAR)
    logger.info(f"Lidar publisher na porcie {config.ZMQ_PORT_LIDAR}")
    
    # ... reszta kodu ...
```

3. Zamien zapis do pliku na publikacje ZMQ:
```python
# STARY KOD (zakomentuj):
# with open("data/lidar.json", "w") as f:
#     json.dump({"tracks": tracks_data}, f)

# NOWY KOD:
pub.send("sensor.lidar", {"tracks": tracks_data})
logger.debug(f"Opublikowano {len(tracks_data)} trackow")
```

**Weryfikacja:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\lidar
python src/run_live.py
```

**Oczekiwany wynik:**
```
2026-01-04 16:30:00 | INFO | lidar | Lidar publisher na porcie 5555
2026-01-04 16:30:01 | DEBUG | lidar | Opublikowano 2 trackow
...
```

---

## Modul 2: Wizja (Migracja na ZMQ)

### Cel
Zastapienie zapisu do pliku `camera.json` publikacja przez ZMQ.

### Krok 2.1: Modyfikacja cv_agent.py

**Plik do edycji:** `warstwa_wizji/src/cv_agent.py`

**Zmiany do wprowadzenia:**

1. Dodaj importy:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from watus_core.comm import Publisher
from watus_core.config import config
from watus_core.logger import get_logger

logger = get_logger("vision")
```

2. W metodzie `__init__` klasy CVAgent dodaj:
```python
def __init__(self, ...):
    # ... istniejacy kod ...
    
    # Nowy publisher
    self.pub = Publisher(port=config.ZMQ_PORT_VISION)
    logger.info(f"Vision publisher na porcie {config.ZMQ_PORT_VISION}")
```

3. W metodzie `run()` zamien zapis JSON na ZMQ:
```python
# STARY KOD (zakomentuj):
# if self.save_to_json is not None:
#     self.save_to_json("camera.jsonl", detections)

# NOWY KOD:
self.pub.send("sensor.vision", detections)
```

---

## Modul 3: Audio (Unifikacja)

### Cel
Zastapienie lokalnych printow i konfiguracji uzyciem watus_core.

### Krok 3.1: Aktualizacja Konfiguracji

**Plik do edycji:** `warstwa_audio/watus_audio/config.py`

**Zmiany:**
Na poczatku pliku dodaj:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import centralnej konfiguracji
from watus_core.config import config as core_config

# Uzyj portow z centralnej konfiguracji
PUB_ADDR = f"tcp://127.0.0.1:{core_config.ZMQ_PORT_AUDIO}"
```

### Krok 3.2: Zamiana print na logger

**Pliki do edycji:** Wszystkie pliki w `warstwa_audio/watus_audio/`

**Zamiana:**
```python
# STARY KOD:
print("[Watus] Jakas wiadomosc")

# NOWY KOD:
from watus_core.logger import get_logger
logger = get_logger("audio")
logger.info("Jakas wiadomosc")
```

---

## Modul 4: Consolidator (Mozg Operacyjny)

### Cel
Przepisanie consolidatora z "czytacza plikow" na subskrybenta ZMQ.

### Krok 4.1: Utworzenie consolidator_zmq.py

**Plik do utworzenia:** `consolidator/consolidator_zmq.py`

**Zawartosc pliku:**
```python
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from watus_core.comm import Publisher, Subscriber
from watus_core.config import config
from watus_core.logger import get_logger

logger = get_logger("consolidator")

CAMERA_FOV_HALF = 51.0

class WorldModel:
    """Model swiata laczacy dane z sensorow."""
    
    def __init__(self):
        self.latest_lidar_data = None
        self.latest_vision_data = None
        self.last_lidar_ts = 0
        self.last_vision_ts = 0
    
    def update_lidar(self, data: dict) -> None:
        self.latest_lidar_data = data
        self.last_lidar_ts = data.get("ts", time.time())
        logger.debug(f"Lidar update: {len(data.get('tracks', []))} tracks")
    
    def update_vision(self, data: dict) -> None:
        self.latest_vision_data = data
        self.last_vision_ts = data.get("ts", time.time())
        logger.debug(f"Vision update: {len(data.get('objects', []))} objects")
    
    def fuse(self) -> dict:
        """Laczy dane z lidar i vision."""
        combined_tracks = []
        
        if not self.latest_lidar_data:
            return {"tracks": combined_tracks, "ts": time.time()}
        
        for track in self.latest_lidar_data.get("tracks", []):
            entry = {
                "id": track.get("id"),
                "type": track.get("type", "unknown"),
                "last_position": track.get("last_position"),
                "source": "lidar"
            }
            
            # Dopasuj do vision jesli dostepne
            if self.latest_vision_data and self.latest_vision_data.get("objects"):
                cam_obj = self.latest_vision_data["objects"][0]
                angle_camera = cam_obj.get("angle", 0)
                angle_lidar = self._compute_lidar_angle(track)
                
                if abs(angle_camera - angle_lidar) <= CAMERA_FOV_HALF:
                    entry["gender"] = cam_obj.get("type")
                    entry["camera_bbox"] = {
                        "left": cam_obj.get("left"),
                        "top": cam_obj.get("top"),
                        "width": cam_obj.get("width"),
                        "height": cam_obj.get("height")
                    }
                    entry["source"] = "fused"
            
            combined_tracks.append(entry)
        
        return {"tracks": combined_tracks, "ts": time.time()}
    
    def _compute_lidar_angle(self, track: dict) -> float:
        pos = track.get("last_position", [0, 1])
        return math.degrees(math.atan2(pos[0], pos[1]))


def main():
    logger.info("Uruchamiam Consolidator ZMQ")
    
    # Subskrybent dla Lidar i Vision
    sub = Subscriber(
        ports=[config.ZMQ_PORT_LIDAR, config.ZMQ_PORT_VISION]
    )
    
    # Publisher dla World State
    pub = Publisher(port=config.ZMQ_PORT_WORLD)
    
    model = WorldModel()
    last_fuse_time = 0
    FUSE_INTERVAL_MS = 100
    
    logger.info(f"Subskrybuje porty: {config.ZMQ_PORT_LIDAR}, {config.ZMQ_PORT_VISION}")
    logger.info(f"Publikuje na porcie: {config.ZMQ_PORT_WORLD}")
    
    try:
        while True:
            result = sub.recv(timeout_ms=10)
            
            if result:
                topic, data = result
                if topic == "sensor.lidar":
                    model.update_lidar(data)
                elif topic == "sensor.vision":
                    model.update_vision(data)
            
            # Fuzja co FUSE_INTERVAL_MS
            current_time = time.time() * 1000
            if current_time - last_fuse_time >= FUSE_INTERVAL_MS:
                state = model.fuse()
                pub.send("world.state", state)
                last_fuse_time = current_time
    
    except KeyboardInterrupt:
        logger.info("Zamykam Consolidator")
    finally:
        sub.close()
        pub.close()


if __name__ == "__main__":
    main()
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\consolidator
python consolidator_zmq.py
```

**Oczekiwany wynik:**
```
2026-01-04 16:30:00 | INFO | consolidator | Uruchamiam Consolidator ZMQ
2026-01-04 16:30:00 | INFO | consolidator | Subskrybuje porty: 5555, 5556
2026-01-04 16:30:00 | INFO | consolidator | Publikuje na porcie: 5558
```

---

## Modul 5: Warstwa LLM (Brain)

### Cel
Wzbogacenie LLM o kontekst z world.state.

### Krok 5.1: Subskrypcja World State

**Plik do edycji:** `warstwa_llm/src/main.py`

**Zmiany:**
1. Dodaj import i watek subskrybenta:
```python
import threading
from watus_core.comm import Subscriber
from watus_core.config import config

CURRENT_WORLD_CONTEXT = {}
CONTEXT_LOCK = threading.Lock()

def world_context_thread():
    sub = Subscriber(ports=[config.ZMQ_PORT_WORLD])
    while True:
        result = sub.recv(timeout_ms=100)
        if result:
            _, data = result
            with CONTEXT_LOCK:
                global CURRENT_WORLD_CONTEXT
                CURRENT_WORLD_CONTEXT = data

# Uruchom watek przy starcie
threading.Thread(target=world_context_thread, daemon=True).start()
```

### Krok 5.2: Wzbogacanie Promptu

W funkcji `process_question`:
```python
def format_world_context() -> str:
    with CONTEXT_LOCK:
        tracks = CURRENT_WORLD_CONTEXT.get("tracks", [])
    
    if not tracks:
        return "Nie widze zadnych obiektow."
    
    descriptions = []
    for t in tracks:
        pos = t.get("last_position", [0, 0])
        dist = math.sqrt(pos[0]**2 + pos[1]**2)
        desc = f"{t.get('type', 'obiekt')} ({dist:.1f}m)"
        if t.get("gender"):
            desc += f", {t['gender']}"
        descriptions.append(desc)
    
    return "Widze: " + ", ".join(descriptions)

# W process_question:
context = format_world_context()
enriched_prompt = f"{context}\n\nUzytkownik: {user_question}"
```

---

## Modul 6: Zarzadzanie Procesami

### Krok 6.1: Konfiguracja Ecosystem

**Plik do utworzenia:** `ecosystem.json`

```json
[
  {
    "name": "lidar",
    "cmd": "python lidar/src/run_live.py",
    "cwd": "."
  },
  {
    "name": "vision",
    "cmd": "python warstwa_wizji/main.py",
    "cwd": "."
  },
  {
    "name": "audio",
    "cmd": "python warstwa_audio/run_watus.py",
    "cwd": "."
  },
  {
    "name": "consolidator",
    "cmd": "python consolidator/consolidator_zmq.py",
    "cwd": "."
  },
  {
    "name": "brain",
    "cmd": "python warstwa_llm/src/main.py",
    "cwd": "."
  }
]
```

### Krok 6.2: Skrypt Uruchamiajacy

**Plik do utworzenia:** `run_system.py`

```python
import json
import subprocess
import signal
import sys
import time
from pathlib import Path

from watus_core.logger import get_logger

logger = get_logger("ecosystem")

processes = {}

def load_config():
    with open("ecosystem.json", "r") as f:
        return json.load(f)

def start_process(config: dict) -> subprocess.Popen:
    logger.info(f"Uruchamiam: {config['name']}")
    return subprocess.Popen(
        config["cmd"].split(),
        cwd=config.get("cwd", "."),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def signal_handler(sig, frame):
    logger.info("Otrzymano sygnal zakonczenia")
    for name, proc in processes.items():
        logger.info(f"Zamykam: {name}")
        proc.terminate()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    configs = load_config()
    
    for config in configs:
        processes[config["name"]] = start_process(config)
        time.sleep(1)  # Daj czas na start
    
    logger.info(f"Uruchomiono {len(processes)} procesow")
    
    # Monitoring
    while True:
        for name, proc in list(processes.items()):
            ret = proc.poll()
            if ret is not None:
                logger.warning(f"Proces {name} zakonczyl sie kodem {ret}")
                # Restart
                for cfg in configs:
                    if cfg["name"] == name:
                        time.sleep(5)
                        processes[name] = start_process(cfg)
                        break
        time.sleep(1)

if __name__ == "__main__":
    main()
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson
python run_system.py
```

**Oczekiwany wynik:**
```
2026-01-04 16:30:00 | INFO | ecosystem | Uruchamiam: lidar
2026-01-04 16:30:01 | INFO | ecosystem | Uruchamiam: vision
2026-01-04 16:30:02 | INFO | ecosystem | Uruchamiam: audio
2026-01-04 16:30:03 | INFO | ecosystem | Uruchamiam: consolidator
2026-01-04 16:30:04 | INFO | ecosystem | Uruchamiam: brain
2026-01-04 16:30:05 | INFO | ecosystem | Uruchomiono 5 procesow
```

---

## Protokol Komunikacji ZMQ

### Tematy i Format Danych

| Temat | Nadawca | Odbiorca | Format |
|-------|---------|----------|--------|
| `sensor.lidar` | Lidar | Consolidator | `{"tracks": [...], "ts": ...}` |
| `sensor.vision` | Wizja | Consolidator | `{"objects": [...], "ts": ...}` |
| `world.state` | Consolidator | Brain (LLM) | `{"tracks": [...], "ts": ...}` |
| `user.voice_cmd` | Audio | Brain (LLM) | `{"text": "...", "speaker": "..."}` |
| `robot.action` | Brain (LLM) | Audio/Motor | `{"action": "speak", "payload": "..."}` |

### Przykladowe Dane

**sensor.lidar:**
```json
{
  "tracks": [
    {"id": "abc123", "last_position": [2.0, 3.0], "type": "human"}
  ],
  "ts": 1704380400.123
}
```

**sensor.vision:**
```json
{
  "objects": [
    {"type": "person", "angle": 15.5, "left": 100, "top": 50, "width": 150, "height": 300}
  ],
  "countOfPeople": 1,
  "ts": 1704380400.125
}
```

**world.state:**
```json
{
  "tracks": [
    {"id": "abc123", "type": "human", "source": "fused", "gender": "male"}
  ],
  "ts": 1704380400.200
}
```

---

## Checklist Implementacji

| Krok | Opis | Status |
|------|------|--------|
| 0.1 | Utworz folder watus_core | [ ] |
| 0.2 | Utworz config.py | [ ] |
| 0.3 | Utworz logger.py | [ ] |
| 0.4 | Utworz comm.py | [ ] |
| 0.5 | Przetestuj ZMQ pub/sub | [ ] |
| 1.1 | Zmodyfikuj lidar/run_live.py | [ ] |
| 2.1 | Zmodyfikuj warstwa_wizji/cv_agent.py | [ ] |
| 3.1 | Zunifikuj warstwa_audio/config.py | [ ] |
| 3.2 | Zamien print na logger w audio | [ ] |
| 4.1 | Utworz consolidator_zmq.py | [ ] |
| 5.1 | Dodaj subskrypcje world.state do LLM | [ ] |
| 5.2 | Wzbogac prompt o context | [ ] |
| 6.1 | Utworz ecosystem.json | [ ] |
| 6.2 | Utworz run_system.py | [ ] |
| 6.3 | Przetestuj caly system | [ ] |
