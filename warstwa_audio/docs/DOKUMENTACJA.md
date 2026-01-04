# Dokumentacja Modulu Warstwa Audio

## Wprowadzenie

Modul warstwa_audio stanowi komponent systemu WATUS odpowiedzialny za interakcje glosowa z uzytkownikiem. Obejmuje rozpoznawanie mowy (Speech-to-Text, STT), synteze mowy (Text-to-Speech, TTS), wykrywanie aktywnosci glosowej (Voice Activity Detection, VAD) oraz weryfikacje tozsamosci mowcy. Modul jest zaprojektowany do pracy w czasie rzeczywistym z niskimi opoznieniami.

---

## Architektura Systemu

### Glowne Komponenty

**SpeechToTextProcessingEngine (stt.py)** - Silnik rozpoznawania mowy obslugujacy:
- Wykrywanie mowy w strumieniu audio (WebRTC VAD)
- Transkrypcje za pomoca Faster Whisper
- Logike lidera i weryfikacje mowcy

**Synteza Mowy (tts.py)** - Obsluguje trzy backendy TTS:
- Piper (lokalny, ONNX)
- Gemini (zdalny, Google API)
- XTTS-v2 (lokalny, Coqui)

**ZMQMessageBus (bus.py)** - Szyna komunikatow ZMQ do wymiany danych miedzy procesami.

**SystemState (state.py)** - Zarzadzanie stanem systemu i blokowanie wejscia podczas odpowiedzi.

**Config (config.py)** - Centralna konfiguracja ladowana z pliku .env.

### Hierarchia Wywolan

```
watus_main.py
  -> main()
     -> SpeechToTextProcessingEngine.__init__()
        -> _initialize_local_whisper_model()
        -> create_speaker_verifier()
     -> start_listening_loop()
        -> _vad_is_speech()
        -> _process_recorded_speech_segment()
           -> _transcribe_audio_segment()
           -> bus.publish_leader_utterance()
     -> tts_worker_thread()
        -> synthesize_speech_and_play()
           -> synthesize_speech_piper() / synthesize_speech_gemini() / synthesize_speech_xtts()
```

---

## Konfiguracja

### Plik .env

Wszystkie ustawienia sa ladowane z pliku `.env` w katalogu glownym modulu. Przykladowy plik `.env.example` zawiera szablony wszystkich dostepnych zmiennych.

### Zmienne Ogolne

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| KMP_DUPLICATE_LIB_OK | TRUE | Zapobieganie bledom Intel MKL |
| OMP_NUM_THREADS | 1 | Liczba watkow OpenMP |
| MKL_NUM_THREADS | 1 | Liczba watkow MKL |

### Komunikacja ZMQ

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| ZMQ_PUB_ADDR | tcp://127.0.0.1:7780 | Adres gniazda publikujacego |
| ZMQ_SUB_ADDR | tcp://127.0.0.1:7781 | Adres gniazda subskrybujacego |

### Konfiguracja STT (Speech-to-Text)

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| STT_PROVIDER | local | Dostawca: "local" (Whisper) lub "groq" |
| WHISPER_SIZE | medium | Rozmiar modelu: tiny, base, small, medium, large, large-v3 |
| WHISPER_DEVICE_TYPE | cpu | Urzadzenie: "cpu" lub "gpu" |
| WHISPER_NUM_WORKERS | 1 | Liczba workerow |
| GROQ_API_KEY | - | Klucz API Groq (jesli STT_PROVIDER=groq) |

Automatyczne mapowanie:
- Jesli WHISPER_DEVICE_TYPE=gpu: WHISPER_DEVICE=cuda, WHISPER_COMPUTE=float16
- Jesli WHISPER_DEVICE_TYPE=cpu: WHISPER_DEVICE=cpu, WHISPER_COMPUTE=int8

### Konfiguracja TTS (Text-to-Speech)

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| TTS_PROVIDER | gemini | Dostawca: "piper", "gemini" lub "xtts" |
| PIPER_MODEL_PATH | models/piper/... | Sciezka do modelu ONNX Piper |
| PIPER_SAMPLE_RATE | 22050 | Czestotliwosc probkowania Piper |
| PIPER_BIN | - | Sciezka do binarki Piper (fallback) |
| GEMINI_API_KEY | - | Klucz API Google Gemini |
| GEMINI_MODEL | gemini-2.0-flash-exp | Model Gemini |
| GEMINI_VOICE | Callirrhoe | Glos Gemini |
| XTTS_SPEAKER_WAV | models/xtts/ref.wav | Plik referencyjny glosu XTTS |
| XTTS_LANGUAGE | pl | Jezyk XTTS |

### Konfiguracja VAD (Voice Activity Detection)

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| WATUS_SR | 16000 | Czestotliwosc probkowania audio |
| WATUS_BLOCKSIZE | 320 | Rozmiar bloku (20ms przy 16kHz) |
| WATUS_VAD_MODE | 1 | Tryb VAD (0-3, wyzszy = czulszy) |
| WATUS_VAD_MIN_MS | 150 | Minimalna dlugosc mowy (ms) |
| WATUS_SIL_MS_END | 450 | Czas ciszy konczacy wypowiedz (ms) |
| ASR_MIN_DBFS | -34 | Minimalny poziom glosnosci (dBFS) |

### Parametry Detekcji Mowy

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| WATUS_PREBUFFER_FRAMES | 15 | Ramki pre-buffer przed detekcja |
| WATUS_START_MIN_FRAMES | 4 | Minimalna liczba ramek do startu |
| WATUS_START_MIN_DBFS | -30 | Minimalna glosnosc do startu |
| WATUS_MIN_MS_BEFORE_ENDPOINT | 500 | Min. czas przed zakonczeniem |
| MAX_UTT_MS | 6500 | Maksymalna dlugosc wypowiedzi (ms) |
| WATUS_GAP_TOL_MS | 450 | Tolerowana przerwa w mowie (ms) |

### Weryfikacja Mowcy

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| SPEAKER_VERIFY | 1 | Wlacz weryfikacje mowcy |
| WAKE_WORDS | hej watusiu,... | Lista slow wybudzajacych |
| SPEAKER_THRESHOLD | 0.64 | Prog podobienstwa glosu (0-1) |
| SPEAKER_STICKY_SEC | 3600 | Czas pamietania lidera (s) |
| SPEAKER_REQUIRE_MATCH | 1 | Wymagaj dopasowania mowcy |

### Urzadzenia Audio

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| WATUS_INPUT_DEVICE | - | ID mikrofonu |
| WATUS_OUTPUT_DEVICE | - | ID glosnikow |

Lista dostepnych urzadzen: `python -m sounddevice`

---

## Struktura Folderow

Oczekiwana struktura katalogu warstwa_audio:

```
warstwa_audio/
|-- .env                        # Konfiguracja (WYMAGANE DO UTWORZENIA)
|-- .env.example                # Szablon konfiguracji
|-- .gitignore                  
|-- README.md                   
|-- requirements.txt            
|
|-- camera_runner.py            # Integracja z kamera
|-- run_reporter.py             # Uruchamianie reportera
|-- run_watus.py                # Glowny punkt wejscia
|-- test_ollama.py              # Testy Ollama
|
|-- docs/                       # Dokumentacja
|   |-- DOKUMENTACJA.md         # Ten plik
|
|-- models/                     # Modele ML (WYMAGANE DO UTWORZENIA)
|   |-- faster-whisper-medium/  # Model Whisper (lub inny rozmiar)
|   |-- piper/                  
|   |   |-- voices/
|   |       |-- pl_PL-jarvis_wg_glos-medium.onnx
|   |-- xtts/                   
|       |-- ref.wav             # Referencyjny glos dla XTTS
|
|-- data/                       # Dane wyjsciowe (WYMAGANE DO UTWORZENIA)
|   |-- watus_audio/
|       |-- dialog.jsonl        # Historia dialogu
|       |-- camera.jsonl        # Dane z kamery
|       |-- responses.jsonl     # Odpowiedzi systemu
|       |-- meldunki.jsonl      # Meldunki
|
|-- scenarios_text/             # Scenariusze tekstowe
|   |-- active.jsonl            # Aktywny scenariusz
|
|-- tests/                      # Testy jednostkowe
|
|-- watus_audio/                # Glowny pakiet Python
    |-- __init__.py
    |-- audio_utils.py          # Narzedzia audio
    |-- bus.py                  # Szyna ZMQ
    |-- common.py               # Funkcje wspolne
    |-- config.py               # Konfiguracja
    |-- led.py                  # Sterowanie LED
    |-- reporter_camera.py      # Reporter kamery
    |-- reporter_llm.py         # Reporter LLM
    |-- reporter_main.py        # Glowny reporter
    |-- state.py                # Stan systemu
    |-- stt.py                  # Rozpoznawanie mowy
    |-- tts.py                  # Synteza mowy
    |-- verifier.py             # Weryfikacja mowcy
    |-- watus_main.py           # Glowna petla
```

### Wymagane Foldery do Utworzenia

1. **Plik .env** - Skopiuj .env.example i uzupelnij wymagane klucze API
2. **models/faster-whisper-{size}/** - Model Whisper (pobierany automatycznie przy pierwszym uruchomieniu)
3. **models/piper/voices/** - Model Piper TTS (jesli TTS_PROVIDER=piper)
4. **data/watus_audio/** - Folder na pliki wyjsciowe

---

## Logika Lidera

System implementuje koncepcje "lidera" - glownego uzytkownika z ktorym prowadzi rozmowe.

### Rejestracja Lidera

1. Uzytkownik wypowiada slowo wybudzajace (np. "hej watusiu")
2. System rejestruje probke glosu jako "lider"
3. Kolejne wypowiedzi sa weryfikowane pod katem zgodnosci z zarejestrowanym glosem

### Weryfikacja Mowcy

Przy kazdej wypowiedzi system:
1. Oblicza embedding glosu
2. Porownuje z zarejestrowanym embedingiem lidera
3. Jesli podobienstwo >= SPEAKER_THRESHOLD, rozpoznaje jako lidera
4. Wypowiedzi nie-lidera sa ignorowane lub zapisywane jako "unknown"

### Tryb Sticky

Po rozpoznaniu lidera, prog jest obnizany (SPEAKER_STICKY_THRESHOLD) przez czas SPEAKER_STICKY_SEC, co ulatwia utrzymanie identyfikacji.

---

## Format Wyjsciowy

### dialog.jsonl

Kazda linia to obiekt JSON z informacja o wypowiedzi:

```json
{
  "type": "leader_utterance",
  "session_id": "abc123",
  "group_id": "leader_1704380400000",
  "speaker_id": "leader",
  "is_leader": true,
  "turn_ids": [1704380400000],
  "text_full": "Jaka jest dzisiejsza pogoda?",
  "category": "wypowiedz",
  "reply_hint": true,
  "ts_start": 1704380399.5,
  "ts_end": 1704380401.2,
  "dbfs": -25.3,
  "verify": {"score": 0.78, "is_leader": true},
  "emit_reason": "endpoint",
  "ts": 1704380401.5
}
```

---

## Obslugiwane Backendy TTS

### Piper (lokalny)

Zalety:
- Brak wymagan internetowych
- Niskie opoznienia
- Darmowy

Wymagania:
- Model ONNX dla jezyka polskiego
- Zainstalowana biblioteka piper lub binarka piper-tts

### Gemini (zdalny)

Zalety:
- Wysoka jakosc glosu
- Brak lokalnych wymogow obliczeniowych

Wymagania:
- Klucz API Google Gemini (GEMINI_API_KEY)
- Polaczenie internetowe
- Mozliwe opoznienia sieciowe

### XTTS-v2 (lokalny)

Zalety:
- Voice cloning - mozliwosc naladowania dowolnego glosu
- Wysoka jakosc
- Wielojezycznosc

Wymagania:
- Plik referencyjny glosu (XTTS_SPEAKER_WAV)
- GPU z CUDA (zalecane)
- Znaczne wymagania RAM/VRAM

---

## Komunikacja ZMQ

System wykorzystuje ZeroMQ do komunikacji miedzy-procesowej.

### Architektura

- **Publisher** (PUB_ADDR) - Publikuje wypowiedzi lidera
- **Subscriber** (SUB_ADDR) - Subskrybuje odpowiedzi z LLM

### Tematy (Topics)

- `dialog.leader` - Wypowiedzi rozpoznane jako lider
- `dialog.unknown` - Wypowiedzi nierozpoznane

---

## Rozwiazywanie Problemow

### Brak dzwieku z mikrofonu

1. Sprawdz dostepne urzadzenia: `python -m sounddevice`
2. Ustaw WATUS_INPUT_DEVICE na poprawne ID
3. Sprawdz czy mikrofon nie jest wyciszony w systemie

### Whisper nie laduje sie

1. Sprawdz dostepnosc CUDA: `torch.cuda.is_available()`
2. Dla CPU ustaw WHISPER_DEVICE_TYPE=cpu
3. Sprawdz czy masz wystarczajaco RAM dla wybranego rozmiaru modelu

### Gemini TTS nie dziala

1. Sprawdz poprawnosc GEMINI_API_KEY
2. Zweryfikuj polaczenie internetowe
3. Sprawdz logi pod katem bledow API (401, 429, etc.)

### Piper TTS nie dziala

1. Sprawdz czy plik modelu ONNX istnieje
2. Zweryfikuj sciezke PIPER_MODEL_PATH
3. Dla metody binarnej sprawdz czy PIPER_BIN jest ustawione

### System nie reaguje na glos

1. Sprawdz czy ASR_MIN_DBFS nie jest zbyt wysoki
2. Zmniejsz WATUS_VAD_MODE dla wiekszej czulosci
3. Sprawdz czy slowo wybudzajace jest poprawnie wymawiane

---

## Ograniczenia

1. **Jeden lider** - System obsluguje jednego lidera naraz
2. **Jezyk polski** - Domyslna konfiguracja jest dla jezyka polskiego
3. **Opoznienia sieciowe** - Backend Gemini moze wprowadzac opoznienia
4. **Wymagania sprzetowe** - XTTS wymaga GPU dla rozsadnej wydajnosci
5. **VAD** - WebRTC VAD moze miec problemy z niestandardowymi odglosami
