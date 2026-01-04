# Instrukcja Walidacji Systemu WATUS Jetson

## Spis Tresci
1. [Wymagania Wstepne](#wymagania-wstepne)
2. [Walidacja Calego Systemu](#walidacja-calego-systemu)
3. [Walidacja Modulu Lidar](#walidacja-modulu-lidar)
4. [Walidacja Modulu Warstwa Wizji](#walidacja-modulu-warstwa-wizji)
5. [Walidacja Modulu Warstwa Audio](#walidacja-modulu-warstwa-audio)
6. [Walidacja Modulu Warstwa LLM](#walidacja-modulu-warstwa-llm)
7. [Walidacja Modulu Consolidator](#walidacja-modulu-consolidator)

---

## Wymagania Wstepne

### Srodowisko Python
Przed rozpoczeciem walidacji upewnij sie, ze masz zainstalowane wymagane zaleznosci.

**Krok 1: Sprawdz wersje Python**
```bash
python --version
```
Oczekiwany wynik: Python 3.10.x lub nowszy

**Krok 2: Zainstaluj glowne zaleznosci**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson
pip install -r requirements.txt
```

### Sprawdzenie Dostepnosci GPU (opcjonalne, ale zalecane)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
Oczekiwany wynik dla GPU: `CUDA available: True`
Oczekiwany wynik dla CPU: `CUDA available: False` (system bedzie wolniejszy)

---

## Walidacja Calego Systemu

### Test 1: Sprawdzenie struktury projektu

**Plik do uruchomienia:** Brak (manualna weryfikacja)

**Czynnosci:**
1. Otworz folder `c:\Users\pawel\Documents\GitHub\watus_jetson`
2. Sprawdz czy istnieja nastepujace foldery:
   - `warstwa_wizji/`
   - `warstwa_audio/`
   - `warstwa_llm/`
   - `lidar/`
   - `consolidator/`

**Oczekiwany wynik:** Wszystkie 5 folderow jest obecnych.

### Test 2: Import glownych modulow

**Plik do utworzenia:** `test_imports.py` w glownym katalogu

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, ".")

print("Test 1: Import warstwa_wizji...")
try:
    from warstwa_wizji.src.cv_agent import CVAgent
    print("  OK - CVAgent zaimportowany")
except Exception as e:
    print(f"  BLAD: {e}")

print("Test 2: Import lidar...")
try:
    from lidar.src.lidar.tracking import HumanTracker
    print("  OK - HumanTracker zaimportowany")
except Exception as e:
    print(f"  BLAD: {e}")

print("Test 3: Import consolidator...")
try:
    from consolidator.consolidator import compute_lidar_angle
    print("  OK - compute_lidar_angle zaimportowany")
except Exception as e:
    print(f"  BLAD: {e}")

print("\nWszystkie testy importu zakonczone.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson
python test_imports.py
```

**Oczekiwany wynik:**
```
Test 1: Import warstwa_wizji...
  OK - CVAgent zaimportowany
Test 2: Import lidar...
  OK - HumanTracker zaimportowany
Test 3: Import consolidator...
  OK - compute_lidar_angle zaimportowany

Wszystkie testy importu zakonczone.
```

---

## Walidacja Modulu Lidar

### Lokalizacja modulu
```
lidar/
|-- src/
    |-- config.py
    |-- check_lidar.py
    |-- lidar/
        |-- tracking.py
        |-- segmentation.py
```

### Test 1: Sprawdzenie konfiguracji

**Plik do uruchomienia:** Brak (manualna weryfikacja)

**Czynnosci:**
1. Otworz plik `lidar/src/config.py`
2. Sprawdz wartosc `LIDAR_PORT`:
   - Dla Windows: powinno byc `"COM3"` lub podobne
   - Dla Linux/Jetson: powinno byc `"/dev/ttyUSB0"`

**Oczekiwany wynik:** Wartosc odpowiada Twojemu systemowi.

### Test 2: Test polaczenia z LiDAR (wymaga sprzetowego LiDAR)

**Plik do uruchomienia:** `lidar/src/check_lidar.py`

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\lidar
python src/check_lidar.py
```

**Oczekiwany wynik (gdy LiDAR podlaczony):**
```
LiDAR connected on COM3
Received X points
```

**Oczekiwany wynik (gdy LiDAR niepodlaczony):**
```
Error: Could not open port COM3
```
To jest poprawny wynik jesli nie masz LiDAR podlaczonego.

### Test 3: Test algorytmu sledzenia (bez sprzetowego LiDAR)

**Plik do utworzenia:** `lidar/test_tracking.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "src")

from lidar.tracking import HumanTracker
from lidar.segmentation import Segment, BeamCategory

# Utworz tracker
tracker = HumanTracker()
print("HumanTracker utworzony pomyslnie")

# Utworz testowy segment (symulacja detekcji czlowieka)
test_segment = Segment(
    start_idx=0,
    end_idx=10,
    center_x=2.0,
    center_y=3.0,
    length=0.5,
    base_category=BeamCategory.HUMAN
)

# Testowa aktualizacja trackera
tracker.update_tracker([test_segment], dt_s=0.1)

# Sprawdz czy powstal jakis track
all_tracks = tracker.get_all_tracks()
print(f"Liczba trackow: {len(all_tracks)}")

if len(all_tracks) > 0:
    track = all_tracks[0]
    print(f"Track ID: {track.id}")
    print(f"Pozycja: ({track.x:.2f}, {track.y:.2f})")
    print(f"Stan: {track.state}")
    print("TEST PASSED")
else:
    print("BLAD: Brak trackow po aktualizacji")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\lidar
python test_tracking.py
```

**Oczekiwany wynik:**
```
HumanTracker utworzony pomyslnie
Liczba trackow: 1
Track ID: xxxxxxxx
Pozycja: (2.00, 3.00)
Stan: init
TEST PASSED
```

---

## Walidacja Modulu Warstwa Wizji

### Lokalizacja modulu
```
warstwa_wizji/
|-- main.py
|-- src/
    |-- cv_agent.py
    |-- cv_utils/
    |-- img_classifiers/
    |-- model_trainer/
|-- models/              (wymagane do utworzenia)
```

### Test 1: Sprawdzenie struktury folderow

**Czynnosci:**
1. Sprawdz czy istnieje folder `warstwa_wizji/models/`
2. Jesli nie istnieje, utworz go:
```bash
mkdir c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_wizji\models
```

### Test 2: Import CVAgent

**Plik do utworzenia:** `warstwa_wizji/test_cvagent.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "src")

print("Test importu CVAgent...")
try:
    from cv_agent import CVAgent
    print("  OK - CVAgent zaimportowany pomyslnie")
except ImportError as e:
    print(f"  BLAD importu: {e}")
except Exception as e:
    print(f"  BLAD: {e}")

print("\nTest importu cv_utils...")
try:
    from cv_utils import calc_obj_angle, calc_brightness
    print("  OK - Funkcje pomocnicze zaimportowane")
except Exception as e:
    print(f"  BLAD: {e}")

print("\nTest zakonczony.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_wizji
python test_cvagent.py
```

**Oczekiwany wynik:**
```
Test importu CVAgent...
  OK - CVAgent zaimportowany pomyslnie

Test importu cv_utils...
  OK - Funkcje pomocnicze zaimportowane

Test zakonczony.
```

### Test 3: Test obliczen geometrycznych

**Plik do utworzenia:** `warstwa_wizji/test_geometry.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "src")

from cv_utils.angle import calc_obj_angle
from cv_utils.brightness import calc_brightness, suggest_mode
import numpy as np

print("Test 1: Obliczanie kata obiektu...")
# Obiekt w srodku obrazu powinien miec kat ~0
angle = calc_obj_angle((320, 240), (340, 260), imgsz=640, fov_deg=102)
print(f"  Kat dla obiektu w srodku: {angle:.2f} stopni")
if abs(angle) < 10:
    print("  OK - Kat bliski 0 dla srodka obrazu")
else:
    print("  OSTRZEZENIE - Nieoczekiwany kat")

print("\nTest 2: Obliczanie jasnosci...")
# Utworz testowy obraz (szary)
test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
brightness = calc_brightness(test_frame)
print(f"  Jasnosc szarego obrazu: {brightness:.2f}")
if 0.4 < brightness < 0.6:
    print("  OK - Jasnosc w oczekiwanym zakresie")
else:
    print("  OSTRZEZENIE - Nieoczekiwana jasnosc")

print("\nTest 3: Sugerowany tryb...")
mode_light = suggest_mode(0.8, "light")
mode_dark = suggest_mode(0.2, "light")
print(f"  Tryb dla jasnosci 0.8: {mode_light}")
print(f"  Tryb dla jasnosci 0.2: {mode_dark}")

print("\nWszystkie testy zakonczone.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_wizji
python test_geometry.py
```

**Oczekiwany wynik:**
```
Test 1: Obliczanie kata obiektu...
  Kat dla obiektu w srodku: X.XX stopni
  OK - Kat bliski 0 dla srodka obrazu

Test 2: Obliczanie jasnosci...
  Jasnosc szarego obrazu: 0.50
  OK - Jasnosc w oczekiwanym zakresie

Test 3: Sugerowany tryb...
  Tryb dla jasnosci 0.8: light
  Tryb dla jasnosci 0.2: dark

Wszystkie testy zakonczone.
```

### Test 4: Uruchomienie CVAgent z kamera (wymaga kamery)

**UWAGA:** Ten test wymaga podlaczonej kamery USB lub strumienia wideo.

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_wizji
python main.py
```

**Oczekiwany wynik:**
- Otworzy sie okno z podgladem kamery
- Na ekranie pojawia sie ramki wokol wykrytych obiektow
- W konsoli wyswietla sie informacja o FPS
- Nacisnij 'q' aby zamknac

**Mozliwe bledy:**
- "Nie moge otworzyc kamery" - sprawdz podlaczenie kamery
- "Brak modelu YOLO" - pobierz model do folderu models/

---

## Walidacja Modulu Warstwa Audio

### Lokalizacja modulu
```
warstwa_audio/
|-- .env                 (wymagane do utworzenia)
|-- .env.example
|-- run_watus.py
|-- watus_audio/
    |-- config.py
    |-- stt.py
    |-- tts.py
```

### Test 1: Przygotowanie pliku .env

**Czynnosci:**
1. Skopiuj plik .env.example do .env:
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_audio
copy .env.example .env
```

2. Edytuj plik .env i uzupelnij wymagane wartosci:
   - `GEMINI_API_KEY=twoj_klucz` (jesli uzywasz Gemini TTS)
   - `WHISPER_DEVICE_TYPE=cpu` lub `gpu`

### Test 2: Sprawdzenie dostepnych urzadzen audio

**Uruchomienie:**
```bash
python -m sounddevice
```

**Oczekiwany wynik:**
Lista urzadzen audio, np:
```
   0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
   1 Microphone (Realtek Audio), MME (2 in, 0 out)
   2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
   3 Speakers (Realtek Audio), MME (0 in, 2 out)
```

Zapamietaj numery urzadzen wejsciowego (mikrofon) i wyjsciowego (glosniki).

### Test 3: Test konfiguracji

**Plik do utworzenia:** `warstwa_audio/test_config.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "watus_audio")

from config import (
    TTS_PROVIDER, STT_PROVIDER, 
    WHISPER_MODEL_NAME, WHISPER_DEVICE,
    SAMPLE_RATE, WAKE_WORDS
)

print("=== Konfiguracja Warstwa Audio ===")
print(f"TTS Provider: {TTS_PROVIDER}")
print(f"STT Provider: {STT_PROVIDER}")
print(f"Whisper Model: {WHISPER_MODEL_NAME}")
print(f"Whisper Device: {WHISPER_DEVICE}")
print(f"Sample Rate: {SAMPLE_RATE}")
print(f"Wake Words: {WAKE_WORDS}")
print("\nKonfiguracja zaladowana pomyslnie.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_audio
python test_config.py
```

**Oczekiwany wynik:**
```
=== Konfiguracja Warstwa Audio ===
TTS Provider: gemini
STT Provider: local
Whisper Model: models/faster-whisper-medium lub Systran/...
Whisper Device: cpu
Sample Rate: 16000
Wake Words: ['hej watusiu', 'hej watuszu', ...]

Konfiguracja zaladowana pomyslnie.
```

### Test 4: Test TTS (synteza mowy)

**UWAGA:** Ten test wymaga glosnikow i poprawnej konfiguracji TTS.

**Plik do utworzenia:** `warstwa_audio/test_tts.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "watus_audio")

from tts import synthesize_speech_and_play

print("Test syntezy mowy...")
print("Odtwarzam tekst: 'Test systemu Watus'")

try:
    synthesize_speech_and_play("Test systemu Watus", None)
    print("TTS zakonczony pomyslnie")
except Exception as e:
    print(f"BLAD TTS: {e}")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_audio
python test_tts.py
```

**Oczekiwany wynik:**
- Uslyszysz glos mowiacy "Test systemu Watus"
- W konsoli: "TTS zakonczony pomyslnie"

---

## Walidacja Modulu Warstwa LLM

### Lokalizacja modulu
```
warstwa_llm/
|-- .env                 (wymagane do utworzenia)
|-- .env.example
|-- src/
    |-- config.py
    |-- api.py
    |-- emma.py
    |-- logic/
        |-- vectordb.py
```

### Test 1: Przygotowanie pliku .env

**Czynnosci:**
1. Skopiuj plik .env.example do .env:
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_llm
copy .env.example .env
```

2. Edytuj plik .env i uzupelnij:
   - `GEMINI_API_KEY=twoj_klucz`
   - `GEMINI_MODEL=gemini-1.5-flash` (lub inny)

### Test 2: Test konfiguracji

**Plik do utworzenia:** `warstwa_llm/test_config.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "src")

from config import (
    GEMINI_MODEL, CHROMADB_PATH,
    PROJECT_ROOT, DATA_DIR
)

print("=== Konfiguracja Warstwa LLM ===")
print(f"Gemini Model: {GEMINI_MODEL}")
print(f"ChromaDB Path: {CHROMADB_PATH}")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Dir: {DATA_DIR}")

# Sprawdz czy API key jest ustawiony
import os
api_key = os.getenv("GEMINI_API_KEY", "")
if api_key:
    print(f"API Key: ***{api_key[-4:]}")
else:
    print("OSTRZEZENIE: GEMINI_API_KEY nie ustawiony!")

print("\nKonfiguracja zaladowana pomyslnie.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_llm
python test_config.py
```

**Oczekiwany wynik:**
```
=== Konfiguracja Warstwa LLM ===
Gemini Model: gemini-1.5-flash
ChromaDB Path: ...\chroma_db
Project Root: ...\warstwa_llm
Data Dir: ...\warstwa_llm\data
API Key: ***xxxx

Konfiguracja zaladowana pomyslnie.
```

### Test 3: Test ChromaDB

**Plik do utworzenia:** `warstwa_llm/test_chromadb.py`

**Zawartosc pliku:**
```python
import sys
sys.path.insert(0, "src")

from logic.vectordb import initialize_vector_db

print("Test inicjalizacji ChromaDB...")
try:
    client, collection = initialize_vector_db()
    if collection:
        print(f"  OK - Kolekcja: {collection.name}")
        print(f"  Liczba dokumentow: {collection.count()}")
    else:
        print("  BLAD - Nie udalo sie utworzyc kolekcji")
except Exception as e:
    print(f"  BLAD: {e}")

print("\nTest ChromaDB zakonczony.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_llm
python test_chromadb.py
```

**Oczekiwany wynik:**
```
Test inicjalizacji ChromaDB...
  OK - Kolekcja: knowledge_base
  Liczba dokumentow: 0

Test ChromaDB zakonczony.
```

### Test 4: Uruchomienie serwera API

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\warstwa_llm
python src/main.py
```

**Oczekiwany wynik:**
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Test endpointu health (w nowym terminalu):**
```bash
curl http://127.0.0.1:8000/api1/health
```

**Oczekiwany wynik:**
```json
{"status": "healthy"}
```

Zatrzymaj serwer: Ctrl+C

---

## Walidacja Modulu Consolidator

### Lokalizacja modulu
```
consolidator/
|-- consolidator.py
|-- consolidator.json    (generowany)
```

### Test 1: Test funkcji obliczania kata

**Plik do utworzenia:** `consolidator/test_consolidator.py`

**Zawartosc pliku:**
```python
from consolidator import compute_lidar_angle

print("Test funkcji compute_lidar_angle...")

# Test 1: Obiekt na wprost (kat 0)
track1 = {"last_position": [0.0, 5.0]}  # x=0, y=5m do przodu
angle1 = compute_lidar_angle(track1)
print(f"  Obiekt na wprost (0, 5): {angle1:.2f} stopni")
if abs(angle1) < 1:
    print("  OK - Kat bliski 0")
else:
    print("  BLAD - Oczekiwano ~0 stopni")

# Test 2: Obiekt po lewej (kat dodatni)
track2 = {"last_position": [3.0, 5.0]}  # x=3m w lewo, y=5m
angle2 = compute_lidar_angle(track2)
print(f"  Obiekt po lewej (3, 5): {angle2:.2f} stopni")
if angle2 > 0:
    print("  OK - Kat dodatni (lewo)")
else:
    print("  BLAD - Oczekiwano kata dodatniego")

# Test 3: Obiekt po prawej (kat ujemny)
track3 = {"last_position": [-3.0, 5.0]}  # x=-3m w prawo, y=5m
angle3 = compute_lidar_angle(track3)
print(f"  Obiekt po prawej (-3, 5): {angle3:.2f} stopni")
if angle3 < 0:
    print("  OK - Kat ujemny (prawo)")
else:
    print("  BLAD - Oczekiwano kata ujemnego")

print("\nWszystkie testy consolidator zakonczone.")
```

**Uruchomienie:**
```bash
cd c:\Users\pawel\Documents\GitHub\watus_jetson\consolidator
python test_consolidator.py
```

**Oczekiwany wynik:**
```
Test funkcji compute_lidar_angle...
  Obiekt na wprost (0, 5): 0.00 stopni
  OK - Kat bliski 0
  Obiekt po lewej (3, 5): 30.96 stopni
  OK - Kat dodatni (lewo)
  Obiekt po prawej (-3, 5): -30.96 stopni
  OK - Kat ujemny (prawo)

Wszystkie testy consolidator zakonczone.
```

---

## Podsumowanie Walidacji

Po zakonczeniu wszystkich testow, stworz raport:

| Modul | Test | Status |
|-------|------|--------|
| Lidar | Import tracking | OK / BLAD |
| Lidar | Test algorytmu | OK / BLAD |
| Wizja | Import CVAgent | OK / BLAD |
| Wizja | Test geometrii | OK / BLAD |
| Audio | Konfiguracja | OK / BLAD |
| Audio | TTS | OK / BLAD |
| LLM | Konfiguracja | OK / BLAD |
| LLM | ChromaDB | OK / BLAD |
| LLM | API Server | OK / BLAD |
| Consolidator | Obliczanie kata | OK / BLAD |

Jesli wszystkie testy przeszly pomyslnie, system jest gotowy do pracy.

---

## Rozwiazywanie Typowych Problemow

### Blad: ModuleNotFoundError
**Przyczyna:** Brak zaleznosci lub zla sciezka
**Rozwiazanie:** 
```bash
pip install -r requirements.txt
```

### Blad: No module named 'torch'
**Przyczyna:** PyTorch nie zainstalowany
**Rozwiazanie:**
```bash
pip install torch torchvision
```

### Blad: CUDA not available
**Przyczyna:** Brak sterownikow NVIDIA lub PyTorch CPU
**Rozwiazanie:** Zainstaluj PyTorch z obsluga CUDA ze strony pytorch.org

### Blad: API Key not set
**Przyczyna:** Brak klucza API w pliku .env
**Rozwiazanie:** Dodaj klucz do odpowiedniego pliku .env

### Blad: Port already in use
**Przyczyna:** Inny proces uzywa portu
**Rozwiazanie:** Znajdz i zamknij proces lub zmien port w konfiguracji
