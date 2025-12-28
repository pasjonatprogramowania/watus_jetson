import sounddevice as sd
from .common import log_message
from . import config

def print_available_audio_devices():
    """
    Wypisuje listę dostępnych urządzeń audio do logów.
    
    Hierarchia wywołań:
        watus_main.py -> main() -> print_available_audio_devices()
    """
    try:
        devs = sd.query_devices()
        log_message("[Watus] Audio devices:")
        for i, d in enumerate(devs):
            log_message(f"  [{i}] {d['name']} in:{d.get('max_input_channels', 0)} out:{d.get('max_output_channels', 0)}")
    except Exception as e:
        log_message(f"[Watus] Nie można wypisać urządzeń: {e}")


def _find_device_index_by_name_fragment(fragment: str, want_output=False):
    """
    Szuka indeksu urządzenia audio po fragmencie nazwy.
    """
    frag = (fragment or "").lower().strip()
    for i, d in enumerate(sd.query_devices()):
        if frag and frag in d['name'].lower():
            if want_output and d.get("max_output_channels", 0) > 0: return i
            if not want_output and d.get("max_input_channels", 0) > 0: return i
    return None


def _resolve_device_identifier(v, want_output=False):
    """
    Rozwiązuje identyfikator urządzenia (indeks lub nazwa) na indeks liczbowy.
    """
    if not v: return None
    try:
        return int(v)
    except ValueError:
        return _find_device_index_by_name_fragment(v, want_output=want_output)


def _auto_select_default_input_device():
    """
    Automatycznie wybiera domyślne urządzenie wejściowe.
    Preferuje 'pulse' lub 'default' aby unikać problematycznych urządzeń APE na Jetson.
    """
    devices = sd.query_devices()
    
    # Preferuj urządzenie 'pulse' (PulseAudio) - najlepiej współpracuje z USB audio
    for i, d in enumerate(devices):
        if 'pulse' in d['name'].lower() and d.get("max_input_channels", 0) > 0:
            return i
    
    # Następnie spróbuj 'default'
    for i, d in enumerate(devices):
        if d['name'].lower() == 'default' and d.get("max_input_channels", 0) > 0:
            return i
    
    # Fallback do systemowego default
    try:
        di = sd.default.device
        if isinstance(di, (list, tuple)) and di[0] is not None: return di[0]
    except Exception:
        pass
    
    # Ostateczność: pierwsze urządzenie z wejściem, ale pomijamy APE
    for i, d in enumerate(devices):
        name_lower = d['name'].lower()
        # Pomijamy urządzenia APE (Audio Processing Engine) - często nie działają jako standardowe mikrofony
        if 'ape' in name_lower:
            continue
        if d.get("max_input_channels", 0) > 0:
            return i
    
    # Absolutna ostateczność - cokolwiek z wejściem
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return i
    return None


def _auto_select_default_output_device():
    """
    Automatycznie wybiera domyślne urządzenie wyjściowe.
    """
    try:
        di = sd.default.device
        if isinstance(di, (list, tuple)) and di[1] is not None: return di[1]
    except Exception:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0: return i
    return None

def get_default_input_device_index():
    """
    Zwraca indeks wybranego urządzenia wejściowego (mikrofonu).
    Bierze pod uwagę konfigurację lub wybiera automatycznie.
    
    Hierarchia wywołań:
        watus_main.py -> main() -> get_default_input_device_index()
    """
    return _resolve_device_identifier(config.IN_DEV_ENV, want_output=False) or _auto_select_default_input_device()

def get_default_output_device_index():
    """
    Zwraca indeks wybranego urządzenia wyjściowego (głośników).
    Bierze pod uwagę konfigurację lub wybiera automatycznie.
    
    Hierarchia wywołań:
        watus_main.py -> main() -> get_default_output_device_index()
    """
    return _resolve_device_identifier(config.OUT_DEV_ENV, want_output=True) or _auto_select_default_output_device()
