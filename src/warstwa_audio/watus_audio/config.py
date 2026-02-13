import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych z głównego pliku .env projektu
_PROJECT_ROOT_ENV = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_PROJECT_ROOT_ENV, override=True)

def _env_int(key: str, default: int) -> int:
    """Pobiera zmienną środowiskową jako int. Traktuje pusty string jako brak wartości."""
    val = os.environ.get(key)
    if not val:  # None or ""
        return default
    return int(val)

def _env_float(key: str, default: float) -> float:
    """Pobiera zmienną środowiskową jako float. Traktuje pusty string jako brak wartości."""
    val = os.environ.get(key)
    if not val:  # None or ""
        return default
    return float(val)

# === Ustawienia Ogólne ===
# Zapobieganie błędom bibliotek Intel MKL/OpenMP
KMP_DUPLICATE_LIB_OK = os.environ.get("KMP_DUPLICATE_LIB_OK") or "TRUE"
OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS") or "1"
MKL_NUM_THREADS = os.environ.get("MKL_NUM_THREADS") or "1"
CT2_SKIP_CONVERTERS = os.environ.get("CT2_SKIP_CONVERTERS") or "1"

# === Komunikacja ZMQ ===
# Adresy gniazd ZMQ do komunikacji między procesami
PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")

# === Synteza Mowy (TTS) ===
# Wybór dostawcy TTS.
# Możliwe wartości:
#   - 'piper': Lokalny silnik, wymaga modelu ONNX.
#   - 'gemini': Zdalny API Google Gemini.
#   - 'inworld': Zdalny API Inworld AI (https://inworld.ai/tts).
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "gemini").lower()

# Konfiguracja Piper TTS
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "models/piper/voices/pl_PL-jarvis_wg_glos-medium.onnx")
PIPER_SR = _env_int("PIPER_SAMPLE_RATE", 22050)
PIPER_BIN = os.environ.get("PIPER_BIN")
PIPER_CONFIG = os.environ.get("PIPER_CONFIG")

# Konfiguracja Gemini TTS (Live API)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Callirrhoe")
GEMINI_LIVE_RECEIVE_SAMPLE_RATE = _env_int("GEMINI_LIVE_RECEIVE_SAMPLE_RATE", 24000)

# Konfiguracja XTTS-v2
XTTS_MODEL_PATH = os.environ.get("XTTS_MODEL_PATH") # Opcjonalnie, jeśli manualnie
XTTS_SPEAKER_WAV = os.environ.get("XTTS_SPEAKER_WAV", "models/xtts/ref.wav")
XTTS_LANGUAGE = os.environ.get("XTTS_LANGUAGE", "pl")

# Konfiguracja Inworld TTS
# Klucz API w formacie Base64 (pobierz z: https://platform.inworld.ai/)
INWORLD_API_KEY = os.environ.get("INWORLD_API_KEY")
# Model Inworld TTS.
# Możliwe wartości: "inworld-tts-1", "inworld-tts-1-max", "inworld-tts-1.5-mini", "inworld-tts-1.5-max"
INWORLD_MODEL = os.environ.get("INWORLD_MODEL", "inworld-tts-1.5-max")
# Głos Inworld. Przykłady: "Ashley", "Dennis", itp.
INWORLD_VOICE = os.environ.get("INWORLD_VOICE", "Ashley")
# Częstotliwość próbkowania audio dla Inworld (Hz). Domyślnie 48000.
INWORLD_SAMPLE_RATE = _env_int("INWORLD_SAMPLE_RATE", 48000)
# Prędkość mówienia Inworld (0.5 - 1.5). Domyślnie 1.0.
INWORLD_SPEED = _env_float("INWORLD_SPEED", 1.0)

# === Rozpoznawanie Mowy (STT) ===
# Wybór dostawcy STT.
# Możliwe wartości:
#   - 'local': Lokalny model Whisper (faster-whisper).
#   - 'groq': Zdalne API Groq (obecnie nieużywane/eksperymentalne).
STT_PROVIDER = os.environ.get("STT_PROVIDER", "local").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "whisper-large-v3")

def _normalize_fw_model(name: str) -> str:
    """
    Normalizuje nazwę modelu Faster Whisper.
    Jeśli podano krótką nazwę (np. 'small'), zamienia ją na pełną ścieżkę repozytorium.
    
    Argumenty:
        name (str): Nazwa modelu.
        
    Zwraca:
        str: Pełna nazwa modelu (np. 'guillaumekln/faster-whisper-small').
        
    Hierarchia wywołań:
        (używana przy inicjalizacji zmiennych globalnych w config.py)
    """
    name = (name or "").strip()
    short = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"}
    if "/" not in name and name.lower() in short:
        return f"guillaumekln/faster-whisper-{name.lower()}"
    return name

# === Konfiguracja Whisper (Uproszczona) ===
# Rozmiar modelu Whisper.
# Możliwe wartości: 'tiny', 'base', 'small', 'medium', 'large', 'large-v3'.
_whisper_size = os.environ.get("WHISPER_SIZE", "medium").lower()

# Typ urządzenia obliczeniowego.
# Możliwe wartości: 'cpu', 'gpu' (zamieniane na 'cuda' w kodzie).
_whisper_device_type = os.environ.get("WHISPER_DEVICE_TYPE", "cpu").lower()

# Automatyczne ustalenie ścieżki do modelu
# Najpierw sprawdzamy lokalny katalog models/faster-whisper-{size}
_local_model_path = f"models/faster-whisper-{_whisper_size}"
if os.path.exists(_local_model_path):
    WHISPER_MODEL_NAME = _local_model_path
else:
    # Fallback do nazwy repozytorium (pobierze do cache)
    WHISPER_MODEL_NAME = f"Systran/faster-whisper-{_whisper_size}"

# Automatyczne ustalenie parametrów compute/device
if _whisper_device_type == "gpu":
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE = "float16"
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE = "int8"

# Nadpisanie (opcjonalne) przez stare zmienne, jeśli ktoś ich użył ręcznie w .env
if os.environ.get("WHISPER_MODEL"):
    WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL")
if os.environ.get("WHISPER_DEVICE"):
    WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE")
if os.environ.get("WHISPER_COMPUTE"):
    WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE")


WHISPER_NUM_WORKERS = _env_int("WHISPER_NUM_WORKERS", 1)
CPU_THREADS = _env_int("WATUS_CPU_THREADS", os.cpu_count() or 4)

# === Audio i VAD ===
SAMPLE_RATE = _env_int("WATUS_SR", 16000)
BLOCK_SIZE = _env_int("WATUS_BLOCKSIZE", int(round(SAMPLE_RATE * 0.02)))
VAD_MODE = _env_int("WATUS_VAD_MODE", 1)
VAD_MIN_MS = _env_int("WATUS_VAD_MIN_MS", 150)
SIL_MS_END = _env_int("WATUS_SIL_MS_END", 450)
ASR_MIN_DBFS = _env_float("ASR_MIN_DBFS", -34.0)

# Parametry detekcji mowy
PREBUFFER_FRAMES = _env_int("WATUS_PREBUFFER_FRAMES", 15)
START_MIN_FRAMES = _env_int("WATUS_START_MIN_FRAMES", 4)
START_MIN_DBFS = _env_float("WATUS_START_MIN_DBFS", ASR_MIN_DBFS + 4.0)
MIN_MS_BEFORE_ENDPOINT = _env_int("WATUS_MIN_MS_BEFORE_ENDPOINT", 500)
END_AT_DBFS_DROP = _env_float("END_AT_DBFS_DROP", 0.0)
EMIT_COOLDOWN_MS = _env_int("EMIT_COOLDOWN_MS", 300)
MAX_UTT_MS = _env_int("MAX_UTT_MS", 6500)
GAP_TOL_MS = _env_int("WATUS_GAP_TOL_MS", 450)

# Urządzenia audio
# ID urządzeń można sprawdzić za pomocą `python -m sounddevice`.
IN_DEV_ENV = os.environ.get("WATUS_INPUT_DEVICE")  # ID mikrofonu
OUT_DEV_ENV = os.environ.get("WATUS_OUTPUT_DEVICE") # ID głośników

DIALOG_PATH = os.environ.get("DIALOG_PATH", "data/watus_audio/dialog.jsonl")

# === Weryfikacja Mówcy ===
SPEAKER_VERIFY = _env_int("SPEAKER_VERIFY", 1)
WAKE_WORDS = [w.strip() for w in
              (os.environ.get("WAKE_WORDS") or "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu").split(",") if
              w.strip()]
# Próg podobieństwa głosu (0.0 - 1.0). Wyższy = trudniej zaakceptować.
SPEAKER_THRESHOLD = _env_float("SPEAKER_THRESHOLD", 0.64)
# Próg "lepki" - ułatwia utrzymanie identyfikacji, gdy użytkownik już raz został rozpoznany.
SPEAKER_STICKY_THRESHOLD = _env_float("SPEAKER_STICKY_THRESHOLD", SPEAKER_THRESHOLD)
SPEAKER_GRACE = _env_float("SPEAKER_GRACE", 0.12)
SPEAKER_STICKY_SEC = _env_float("SPEAKER_STICKY_SEC", _env_float("SPEAKER_STICKY_S", 3600.0))
SPEAKER_MIN_ENROLL_SCORE = _env_float("SPEAKER_MIN_ENROLL_SCORE", 0.55)
SPEAKER_MIN_DBFS = _env_float("SPEAKER_MIN_DBFS", -40.0)
SPEAKER_MAX_DBFS = _env_float("SPEAKER_MAX_DBFS", -5.0)
SPEAKER_BACK_THRESHOLD = _env_float("SPEAKER_BACK_THRESHOLD", 0.56)
SPEAKER_REQUIRE_MATCH = _env_int("SPEAKER_REQUIRE_MATCH", 1)

# === Zachowanie ===
WAIT_REPLY_S = _env_float("WAIT_REPLY_S", 0.6)

# === Reporter ===
LLM_HTTP_URL = (os.environ.get("LLM_HTTP_URL") or "").strip()
HTTP_TIMEOUT = _env_float("HTTP_TIMEOUT", _env_float("LLM_HTTP_TIMEOUT", 30.0))
SCENARIOS_DIR = os.environ.get("WATUS_SCENARIOS_DIR", "./scenarios_text")
SCENARIO_ACTIVE_PATH = os.environ.get("SCENARIO_ACTIVE_PATH", os.path.join(SCENARIOS_DIR, "active.jsonl"))
CAMERA_NAME  = os.environ.get("CAMERA_NAME", "cam_front")
CAMERA_JSONL = os.environ.get("CAMERA_JSONL", "data/watus_audio/camera.jsonl") # Domyślnie plik lokalny
LOG_DIR   = os.environ.get("LOG_DIR", "./")
RESP_FILE = os.path.join(LOG_DIR, "data/watus_audio/responses.jsonl")
MELD_FILE = os.path.join(LOG_DIR, "data/watus_audio/meldunki.jsonl")
CAM_WINDOW_SEC = _env_float("CAMERA_WINDOW_SEC", 2.5)
