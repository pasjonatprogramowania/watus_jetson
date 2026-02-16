# Instrukcja Konfiguracji Projektu

## 1. Przygotowanie Środowiska

Projekt korzysta z **Python 3.10**.

### Zarządzanie wersjami Pythona

Jeśli masz zainstalowanych wiele wersji Pythona, zaleca się użycie [pyenv](https://github.com/pyenv/pyenv).

**Instalacja i ustawienie wersji:**

```bash
pyenv install 3.10
pyenv local 3.10

```

**Weryfikacja:**

```bash
python --version

```

*Powinna wyświetlić się wersja 3.10.x.*

---

## 2. Środowisko Wirtualne (venv)

**Tworzenie venv:**

```bash
python -m venv ./venv-win

```

**Aktywacja:**

* **Windows:**

```bash
./venv-win/Scripts/activate

```

* **Linux/macOS:**

```bash
source ./venv-win/bin/activate

```

---

## 3. Instalacja Zależności

Zaleca się użycie biblioteki **uv**, która znacząco przyspiesza instalację. Jeśli nie chcesz jej używać, pomiń pierwszy krok i usuń przedrostek `uv` z komendy instalacyjnej.

```bash
# Opcjonalne: instalacja uv
pip install uv

# Instalacja bibliotek z pliku requirements.txt
uv pip install -r requirements.txt

```

---

## 4. Konfiguracja Zmiennych Środowiskowych (.env)

Projekt korzysta z **jednego, skonsolidowanego pliku `.env`** w głównym katalogu projektu. Wszystkie moduły (warstwa_llm, warstwa_audio, warstwa_wizji) wczytują zmienne z tego samego pliku.

### Tworzenie pliku `.env`

1. Skopiuj plik przykładowy:

```bash
cp .env.example .env
```

2. Otwórz plik `.env` i uzupełnij wartości kluczy API oraz konfiguracji zgodnie z komentarzami w pliku.

### Najważniejsze zmienne


| Zmienna             | Opis                                             | Wymagana   |
| ------------------- | ------------------------------------------------ | ---------- |
| `GEMINI_API_KEY`    | Klucz API Google Gemini (TTS + LLM)              | ✅ Tak     |
| `TTS_PROVIDER`      | Silnik TTS:`piper`, `gemini`, `inworld`          | ✅ Tak     |
| `STT_PROVIDER`      | Silnik STT:`local` (Whisper)                     | ✅ Tak     |
| `WHISPER_MODEL`     | Ścieżka/nazwa modelu Whisper                   | ✅ Tak     |
| `WHISPER_DEVICE`    | Urządzenie:`cpu` lub `cuda`                     | ✅ Tak     |
| `LLM_HTTP_URL`      | Adres serwera LLM                                | ✅ Tak     |
| `INWORLD_API_KEY`   | Klucz API Inworld (jeśli`TTS_PROVIDER=inworld`) | Opcjonalna |
| `OPENAI_API_KEY`    | Klucz API OpenAI (jeśli używany)               | Opcjonalna |
| `ANTHROPIC_API_KEY` | Klucz API Anthropic (jeśli używany)            | Opcjonalna |

> **Uwaga:** Pełna lista zmiennych wraz z opisami znajduje się w pliku `.env.example`.

---

## 5. Konfiguracja TTS (Text-To-Speech)

### Opcja A: Model Lokalny (Piper)

1. Pobierz **Piper**: [Releases v2023.11.14-2](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2).
2. Pobierz **Głos**: [WitoldG/polish_piper_models](https://huggingface.co/WitoldG/polish_piper_models/tree/main).

   * Wymagane pliki: `jarvis.onnx` oraz `jarvis.onnx.json`.
3. **Struktura folderów:**

   * Stwórz folder `models/`.
   * Rozpakuj tam Pipera.
   * Pliki `.onnx` umieść w podfolderze `voices/`.

**Docelowa ścieżka:**
`models/piper/voices/pl_PL-jarvis_wg_glos-medium.onnx`

4. W pliku `.env` ustaw:

```
TTS_PROVIDER=piper
PIPER_MODEL_PATH=models/piper/voices/pl_PL-jarvis_wg_glos-medium.onnx
```

---

### Opcja B: Modele w Chmurze

#### Gemini TTS

1. Wygeneruj klucz API: [Google AI Studio](https://aistudio.google.com/app/api-keys).
2. W pliku `.env` w katalogu głównym ustaw:

```
TTS_PROVIDER=gemini
GEMINI_API_KEY=twój_klucz_api
GEMINI_VOICE=Achird
```

#### Inworld TTS

1. Załóż konto: [inworld.ai/tts](https://inworld.ai/tts).
2. Wygeneruj klucze API w zakładce [API Keys](https://platform.inworld.ai/).
3. W pliku `.env` w katalogu głównym ustaw:

```
TTS_PROVIDER=inworld
INWORLD_API_KEY=twój_klucz_base64
INWORLD_MODEL=inworld-tts-1.5-max
INWORLD_VOICE=Ashley
```

---

## 6. Konfiguracja STT — Faster Whisper

Projekt wykorzystuje [faster-whisper](https://github.com/SYSTRAN/faster-whisper) jako lokalny silnik rozpoznawania mowy (STT).

### Instalacja

Pakiet `faster-whisper` jest zawarty w `requirements.txt`. Jeśli chcesz zainstalować go ręcznie:

```bash
pip install faster-whisper
```

> **GPU (CUDA):** Aby korzystać z akceleracji GPU, wymagane jest zainstalowanie CUDA Toolkit (≥11.8) oraz cuDNN. Na systemach z NVIDIA GPU (np. Jetson) upewnij się, że masz zainstalowane odpowiednie sterowniki.

### Pobieranie modelu

Modele Whisper można albo pobrać ręcznie, albo pozwolić bibliotece pobrać je automatycznie do cache.

#### Opcja A: Automatyczne pobieranie (z HuggingFace)

Ustaw w `.env` krótką nazwę modelu — biblioteka pobierze go automatycznie przy pierwszym uruchomieniu:

```
WHISPER_MODEL=medium
WHISPER_DEVICE=cpu
WHISPER_COMPUTE=int8
```

Dostępne rozmiary: `tiny`, `base`, `small`, `medium`, `large`, `large-v3`.

#### Opcja B: Model lokalny (offline)

1. Pobierz model ręcznie z [HuggingFace](https://huggingface.co/Systran):
   * Przykład: [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium)
2. Umieść go w katalogu `models/whisper/`:

large:
https://huggingface.co/Systran/faster-whisper-large-v3
   ```
   models/whisper/faster-whisper-medium/
   ├── config.json
   ├── model.bin
   ├── tokenizer.json
   └── vocabulary.txt
   ```
3. W pliku `.env` ustaw pełną ścieżkę:
   ```
   WHISPER_MODEL=models/whisper/faster-whisper-medium
   ```

### Wybór urządzenia i precyzji


| Urządzenie | `WHISPER_DEVICE` | `WHISPER_COMPUTE` | Uwagi                             |
| ----------- | ---------------- | ----------------- | --------------------------------- |
| CPU         | `cpu`            | `int8`            | Wolniejsze, ale działa wszędzie |
| GPU NVIDIA  | `cuda`           | `float16`         | Wymaga CUDA, znacznie szybsze     |

---

## 7. Konfiguracja LLM

### Gemini

1. Wygeneruj klucz API: [Google AI Studio](https://aistudio.google.com/app/api-keys).
2. W pliku `.env` w katalogu głównym ustaw:

```
GEMINI_API_KEY=twój_klucz_api
GEMINI_MODEL=gemini-flash-lite-latest
```

### OpenAI (opcjonalnie)

1. Wygeneruj klucz API: [OpenAI Platform](https://platform.openai.com/api-keys).
2. W pliku `.env` ustaw:

```
OPENAI_API_KEY=twój_klucz
OPENAI_MODEL=gpt-4o
```

### Anthropic (opcjonalnie)

1. Wygeneruj klucz API: [Anthropic Console](https://console.anthropic.com/).
2. W pliku `.env` ustaw:

```
ANTHROPIC_API_KEY=twój_klucz
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Kod mozna włączyć za pomocą skryptów w folderze scripts, musisz tylko zmienic z "venv-win" na twoją nazwe venva jesli masz inną

Struktóre potrzebnych folderów mozesz stworzyć wlączając ```1make_empty_folders.py``` 
