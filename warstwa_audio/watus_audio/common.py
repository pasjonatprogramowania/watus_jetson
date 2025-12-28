import json
from . import config

def log_message(message: str):
    """
    Wypisuje wiadomość do standardowego wyjścia (konsoli) z wymuszonym flushowaniem bufora.
    
    Argumenty:
        message (str): Treść wiadomości do wypisania.
    
    Zwraca:
        None
    
    Hierarchia wywołań:
        Wywoływana w wielu miejscach projektu:
        - watus_main.py -> main() -> log_message()
        - watus_main.py -> indicate_listen_state() -> log_message()
        - watus_main.py -> indicate_think_state() -> log_message()
        - watus_main.py -> indicate_speak_state() -> log_message()
        - watus_main.py -> indicate_idle_state() -> log_message()
        - watus_main.py -> tts_worker_thread() -> log_message()
        - stt.py -> SpeechToTextProcessingEngine -> log_message()
        - bus.py -> ZMQMessageBus -> log_message()
        - audio_utils.py -> print_available_audio_devices() -> log_message()
        - reporter_main.py -> start_scenario_watch_loop() -> log_message()
        - reporter_main.py -> start_main_loop() -> log_message()
        - reporter_camera.py -> start_camera_tail_loop() -> log_message()
        - reporter_llm.py -> send_query_to_llm() -> log_message()
        - tts.py -> synthesize_speech_* -> log_message()
        - verifier.py -> create_speaker_verifier() -> log_message()
    """
    print(message, flush=True)

def append_line_to_dialog_history(dialog_object: dict, file_path=config.DIALOG_PATH):
    """
    Dopisuje obiekt dialogu (jako linię JSON) do pliku historii dialogów.
    
    Argumenty:
        dialog_object (dict): Obiekt reprezentujący wpis w dialogu (np. wypowiedź użytkownika).
        file_path (str): Ścieżka do pliku, domyślnie pobierana z konfiguracji (config.DIALOG_PATH).
    
    Zwraca:
        None
    
    Hierarchia wywołań:
        - stt.py -> SpeechToTextProcessingEngine._process_recorded_speech_segment() -> append_line_to_dialog_history()
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dialog_object, ensure_ascii=False) + "\n")
    except Exception as e:
        log_message(f"[Watus] Failed to write dialog line: {e}")

def write_object_to_jsonl_file(file_path: str, data_object: dict):
    """
    Zapisuje dowolny obiekt jako linię JSON do wskazanego pliku (tryb append).
    
    Argumenty:
        file_path (str): Ścieżka do pliku docelowego.
        data_object (dict): Obiekt danych do zapisania.
    
    Zwraca:
        None
    
    Hierarchia wywołań:
        - reporter_main.py -> start_main_loop() -> write_object_to_jsonl_file()
        - reporter_llm.py -> send_query_to_llm() -> write_object_to_jsonl_file()
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_object, ensure_ascii=False) + "\n")
    except Exception:
        pass
