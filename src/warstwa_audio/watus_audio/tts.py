import os
import sys
import io
import time
import json
import base64
import subprocess
import tempfile
import sounddevice as sd
import soundfile as sf
from .common import log_message
from . import config

# Sprawdzenie dostępności requests (dla Inworld TTS)
try:
    import requests as _requests_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    log_message("[Watus][TTS] requests nie dostępne - pip install requests (wymagane dla Inworld TTS)")

# Sprawdzenie dostępności Piper
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    log_message("[Watus][TTS] Piper Python API nie dostępne, używam binary metod")

# Sprawdzenie dostępności Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log_message("[Watus][TTS] Google Gemini TTS nie dostępne - pip install google-genai")

LOADED_PIPER_VOICE_MODEL = None

def _prepare_env_for_piper_binary(piper_bin_path: str) -> dict:
    """
    Przygotowuje zmienne środowiskowe dla binarnego Pipera (biblioteki).
    
    Hierarchia wywołań:
        tts.py -> synthesize_speech_piper() -> _prepare_env_for_piper_binary()
    """
    env_vars = os.environ.copy()
    bin_dir = os.path.dirname(piper_bin_path) if piper_bin_path else ""
    phonemize_lib_path = os.path.join(bin_dir, "piper-phonemize", "lib")
    extra_library_paths = []
    if os.path.isdir(bin_dir): extra_library_paths.append(bin_dir)
    if os.path.isdir(phonemize_lib_path): extra_library_paths.append(phonemize_lib_path)
    if not extra_library_paths: return env_vars

    if sys.platform == "darwin":
        path_key = "DYLD_LIBRARY_PATH"
    elif sys.platform.startswith("linux"):
        path_key = "LD_LIBRARY_PATH"
    else:
        path_key = "PATH"
    current_path_value = env_vars.get(path_key, "")
    separator = (":" if path_key != "PATH" else ";")
    env_vars[path_key] = (separator.join([*extra_library_paths, current_path_value]) if current_path_value else separator.join(extra_library_paths))
    return env_vars


def _initialize_piper_voice_model():
    """
    Ładuje model Piper do pamięci (jeśli używamy Python API).
    
    Hierarchia wywołań:
        tts.py -> synthesize_speech_piper() -> _initialize_piper_voice_model()
    """
    global LOADED_PIPER_VOICE_MODEL
    if PIPER_AVAILABLE:
        if not LOADED_PIPER_VOICE_MODEL:  # Initialize if not already loaded
            try:
                model_file_path = config.PIPER_MODEL_PATH
                if not os.path.isfile(model_file_path):
                    log_message(f"[Watus][TTS] Brak modelu Piper: {model_file_path}")
                    return False
                LOADED_PIPER_VOICE_MODEL = PiperVoice.load(model_file_path)
                log_message(f"[Watus][TTS] Piper voice załadowany z: {model_file_path}")
                return True
            except Exception as e:
                log_message(f"[Watus][TTS] Błąd ładowania Piper voice: {e}")
                return False
        else:
            # Already loaded
            return True
    return False


def synthesize_speech_piper(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Piper TTS i odtwarza ją.
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy.
        audio_output_device_index (int): Indeks urządzenia wyjściowego audio.
        
    Hierarchia wywołań:
        tts.py -> synthesize_speech_and_play() -> synthesize_speech_piper()
    """
    if not text_to_synthesize or not text_to_synthesize.strip(): return

    # Próba użycia nowego Python API
    if PIPER_AVAILABLE and _initialize_piper_voice_model():
        try:
            # Generate speech using new Python API (returns AudioChunk iterator)
            import numpy as np
            from piper import AudioChunk

            synthesized_audio_bytes = []
            for audio_stream_chunk in LOADED_PIPER_VOICE_MODEL.synthesize(text_to_synthesize):
                if isinstance(audio_stream_chunk, AudioChunk):
                    chunk_data_float32 = audio_stream_chunk.audio_int16_array.astype(np.float32) / 32768.0
                    synthesized_audio_bytes.append(chunk_data_float32)
                else:
                    log_message(f"[Watus][TTS] Nieoczekiwany typ chunk: {type(audio_stream_chunk)}")

            if synthesized_audio_bytes:
                # Combine all audio chunks
                concatenated_audio_samples = np.concatenate(synthesized_audio_bytes)
                sample_rate_hz = LOADED_PIPER_VOICE_MODEL.config.sample_rate
                sd.play(concatenated_audio_samples, sample_rate_hz, device=audio_output_device_index, blocking=True)
                return
            else:
                log_message(f"[Watus][TTS] Brak danych audio z nowego API")
        except Exception as e:
            log_message(f"[Watus][TTS] Błąd Python API, próbuję binary fallback: {e}")

    # Fallback to binary method (legacy)
    if not config.PIPER_BIN or not os.path.isfile(config.PIPER_BIN):
        log_message(f"[Watus][TTS] Uwaga: brak/niepoprawny PIPER_BIN: {config.PIPER_BIN}")
        return
    if not config.PIPER_MODEL_PATH or not os.path.isfile(config.PIPER_MODEL_PATH):
        log_message(f"[Watus][TTS] Brak/niepoprawny PIPER_MODEL_PATH: {config.PIPER_MODEL_PATH}")
        return

    try:
        if sys.platform == "darwin":
            bin_dir = os.path.dirname(config.PIPER_BIN)
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", bin_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

    piper_config_args = ["--config", config.PIPER_CONFIG] if config.PIPER_CONFIG and os.path.isfile(config.PIPER_CONFIG) else []
    process_env = _prepare_env_for_piper_binary(config.PIPER_BIN)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temporary_wav_file_path = tmp_file.name
    start_time = time.time()
    try:
        command_args = [config.PIPER_BIN, "--model", config.PIPER_MODEL_PATH, *piper_config_args, "--output_file", temporary_wav_file_path]
        if config.PIPER_SR: command_args += ["--sample_rate", str(config.PIPER_SR)]
        subprocess.run(command_args, input=(text_to_synthesize or "").encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=process_env)
        audio_data_array, sample_rate_hz = sf.read(temporary_wav_file_path, dtype="float32")
        sd.play(audio_data_array, sample_rate_hz, device=audio_output_device_index, blocking=True)
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        log_message(f"[Watus][TTS] Piper błąd (proc): {error_message}")
    except Exception as e:
        log_message(f"[Watus][TTS] Odtwarzanie nieudane: {e}")
    finally:
        try:
            os.unlink(temporary_wav_file_path)
        except Exception:
            pass
        log_message(f"[Perf] TTS_play_ms={int((time.time() - start_time) * 1000)}")


import asyncio
import numpy as np


async def _synthesize_speech_gemini_live(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Gemini Live API (WebSocket) i odtwarza ją.
    
    Używa `client.aio.live.connect()` do nawiązania sesji WebSocket,
    wysyła tekst przez `send_client_content()`, odbiera chunki audio PCM
    i odtwarza je przez sounddevice.
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy.
        audio_output_device_index (int): Indeks urządzenia wyjściowego.
        
    Hierarchia wywołań:
        tts.py -> synthesize_speech_gemini() -> _synthesize_speech_gemini_live()
    """
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=config.GEMINI_VOICE
                )
            )
        ),
    )

    start_time = time.time()
    audio_chunks = []

    async with client.aio.live.connect(
        model=config.GEMINI_MODEL, config=live_config
    ) as session:
        # Wyślij tekst do syntezy
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part(text=text_to_synthesize)],
            ),
            turn_complete=True,
        )

        # Odbierz audio z sesji
        turn = session.receive()
        async for response in turn:
            if data := response.data:
                audio_chunks.append(data)
            elif text := response.text:
                log_message(f"[Watus][TTS] Gemini Live text response: {text}")

    if audio_chunks:
        # Złącz chunki raw PCM (16-bit signed int, mono, 24kHz)
        raw_pcm = b"".join(audio_chunks)
        audio_array = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        sample_rate = config.GEMINI_LIVE_RECEIVE_SAMPLE_RATE
        sd.play(audio_array, sample_rate, device=audio_output_device_index, blocking=True)

        log_message(f"[Perf] Gemini_Live_TTS_play_ms={int((time.time() - start_time) * 1000)}")
    else:
        log_message("[Watus][TTS] No audio data from Gemini Live API")


def synthesize_speech_gemini(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Gemini Live API i odtwarza ją.
    Wrapper synchroniczny dla async _synthesize_speech_gemini_live().
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy.
        audio_output_device_index (int): Indeks urządzenia wyjściowego.
        
    Hierarchia wywołań:
        tts.py -> synthesize_speech_and_play() -> synthesize_speech_gemini()
    """
    if not text_to_synthesize or not text_to_synthesize.strip():
        return

    if not GEMINI_AVAILABLE:
        log_message("[Watus][TTS] Gemini TTS not available - pip install google-genai")
        return

    if not config.GEMINI_API_KEY:
        log_message("[Watus][TTS] GEMINI_API_KEY not set")
        return

    try:
        log_message(f"[Watus][TTS] Generating Gemini Live TTS for: {text_to_synthesize[:50]}...")
        asyncio.run(_synthesize_speech_gemini_live(text_to_synthesize, audio_output_device_index))
    except Exception as e:
        log_message(f"[Watus][TTS] Gemini Live TTS error: {e}")


LOADED_XTTS_MODEL = None

def _initialize_xtts_model():
    """
    Ładuje model XTTS do pamięci.
    
    Hierarchia wywołań:
        tts.py -> synthesize_speech_xtts() -> _initialize_xtts_model()
    """
    global LOADED_XTTS_MODEL
    if LOADED_XTTS_MODEL: return True
    
    try:
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"[Watus][TTS] Loading XTTS model on {device}...")
        
        # Ładujemy model XTTS v2
        LOADED_XTTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        log_message("[Watus][TTS] XTTS model loaded.")
        return True
    except Exception as e:
        log_message(f"[Watus][TTS] Failed to load XTTS: {e}")
        return False

def synthesize_speech_xtts(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Coqui XTTS-v2.
    
    Hierarchia wywołań:
        tts.py -> synthesize_speech_and_play() -> synthesize_speech_xtts()
    """
    if not text_to_synthesize or not text_to_synthesize.strip(): return
    
    if not _initialize_xtts_model():
        log_message("[Watus][TTS] XTTS initialization failed.")
        return

    speaker_wav = config.XTTS_SPEAKER_WAV
    if not os.path.isfile(speaker_wav):
        log_message(f"[Watus][TTS] Brak pliku referencyjnego głosu: {speaker_wav}")
        return

    try:
        start_time = time.time()
        # Generujemy do pliku tymczasowego (API TTS często preferuje pliki)
        # Można też użyć .tts() i dostać wav w pamięci, ale .tts_to_file jest prostsze w obsłudze formatów
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Synteza
        LOADED_XTTS_MODEL.tts_to_file(
            text=text_to_synthesize,
            speaker_wav=speaker_wav,
            language=config.XTTS_LANGUAGE,
            file_path=output_path
        )
        
        # Odtwarzanie
        audio_data, sr = sf.read(output_path, dtype="float32")
        sd.play(audio_data, sr, device=audio_output_device_index, blocking=True)
        
        log_message(f"[Perf] XTTS_play_ms={int((time.time() - start_time) * 1000)}")
        
        try:
            os.unlink(output_path)
        except:
            pass
            
    except Exception as e:
        log_message(f"[Watus][TTS] XTTS error: {e}")


def synthesize_speech_inworld(text_to_synthesize: str, audio_output_device_index):
    """
    Generuje mowę za pomocą Inworld TTS API (streaming) i odtwarza ją.
    
    Używa endpointu streamingowego: POST https://api.inworld.ai/tts/v1/voice:stream
    Format audio: LINEAR16 (PCM 16-bit), konfigurowalny sample rate.
    
    Argumenty:
        text_to_synthesize (str): Tekst do syntezy (max 2000 znaków).
        audio_output_device_index (int): Indeks urządzenia wyjściowego audio.
        
    Hierarchia wywołań:
        tts.py -> synthesize_speech_and_play() -> synthesize_speech_inworld()
    """
    if not text_to_synthesize or not text_to_synthesize.strip():
        return

    if not REQUESTS_AVAILABLE:
        log_message("[Watus][TTS] Inworld TTS wymaga biblioteki 'requests' - pip install requests")
        return

    if not config.INWORLD_API_KEY:
        log_message("[Watus][TTS] INWORLD_API_KEY nie ustawiony! Pobierz z: https://platform.inworld.ai/")
        return

    try:
        log_message(f"[Watus][TTS] Generating Inworld TTS for: {text_to_synthesize[:50]}...")
        start_time = time.time()

        url = "https://api.inworld.ai/tts/v1/voice:stream"
        headers = {
            "Authorization": f"Basic {config.INWORLD_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text_to_synthesize,
            "voiceId": config.INWORLD_VOICE,
            "modelId": config.INWORLD_MODEL,
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": config.INWORLD_SAMPLE_RATE,
            },
        }

        # Dodaj prędkość mówienia jeśli inna niż domyślna
        if config.INWORLD_SPEED != 1.0:
            payload["talkingSpeed"] = config.INWORLD_SPEED

        response = _requests_lib.post(url, json=payload, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Zbieranie chunków audio ze streamu
        raw_audio_data = io.BytesIO()
        first_chunk_time = None

        for line in response.iter_lines():
            if not line:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.time()

            try:
                chunk = json.loads(line)
                audio_content_b64 = chunk.get("result", {}).get("audioContent", "")
                if not audio_content_b64:
                    continue
                audio_chunk_bytes = base64.b64decode(audio_content_b64)

                # Pomiń nagłówek WAV w chunkach (44 bajty)
                if len(audio_chunk_bytes) > 44:
                    raw_audio_data.write(audio_chunk_bytes[44:])
                else:
                    raw_audio_data.write(audio_chunk_bytes)
            except (json.JSONDecodeError, KeyError) as e:
                log_message(f"[Watus][TTS] Inworld chunk parse error: {e}")
                continue

        audio_bytes = raw_audio_data.getvalue()
        if not audio_bytes:
            log_message("[Watus][TTS] Inworld: brak danych audio w odpowiedzi")
            return

        # Zbuduj pełny plik WAV z zebranych surowych danych PCM
        import wave
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bajty
            wf.setframerate(config.INWORLD_SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        wav_buffer.seek(0)
        audio_samples_array, sample_rate_hz = sf.read(wav_buffer, dtype='float32')

        sd.play(audio_samples_array, sample_rate_hz, device=audio_output_device_index, blocking=True)

        ttfb = f", TTFB={int((first_chunk_time - start_time) * 1000)}ms" if first_chunk_time else ""
        log_message(f"[Perf] Inworld_TTS_play_ms={int((time.time() - start_time) * 1000)}{ttfb}")

    except _requests_lib.exceptions.HTTPError as e:
        log_message(f"[Watus][TTS] Inworld HTTP error: {e.response.status_code} - {e.response.text[:200]}")
    except _requests_lib.exceptions.ConnectionError:
        log_message("[Watus][TTS] Inworld: brak połączenia z API")
    except _requests_lib.exceptions.Timeout:
        log_message("[Watus][TTS] Inworld: timeout połączenia")
    except Exception as e:
        log_message(f"[Watus][TTS] Inworld TTS error: {e}")


def synthesize_speech_and_play(text_to_synthesize: str, audio_output_device_index):
    """
    Uniwersalna funkcja TTS - wybiera Piper, Gemini, XTTS lub Inworld w zależności od konfiguracji.
    
    Argumenty:
        text_to_synthesize (str): Tekst do wypowiedzenia.
        audio_output_device_index (int): Indeks urządzenia wyjściowego.
        
    Hierarchia wywołań:
        watus_main.py -> tts_worker_thread() -> synthesize_speech_and_play()
    """
    provider = config.TTS_PROVIDER
    
    if provider == "gemini":
        log_message("[Watus][TTS] Using Gemini TTS")
        synthesize_speech_gemini(text_to_synthesize, audio_output_device_index)
    elif provider == "inworld":
        log_message("[Watus][TTS] Using Inworld TTS")
        synthesize_speech_inworld(text_to_synthesize, audio_output_device_index)
    elif provider == "xtts":
        log_message("[Watus][TTS] Using XTTS-v2")
        synthesize_speech_xtts(text_to_synthesize, audio_output_device_index)
    else:
        log_message("[Watus][TTS] Using Piper TTS")
        synthesize_speech_piper(text_to_synthesize, audio_output_device_index)
