"""
Skrypt startowy: Warstwa Audio - Watus (główny agent głosowy)
Uruchamia nasłuchiwanie mikrofonu, STT, TTS i komunikację z LLM.
"""
import os
import sys
import subprocess

# === KONFIGURACJA ===
VENV_NAME = "venv-win"  # Nazwa folderu venv w głównym katalogu projektu
# ====================

# Oblicz ścieżki
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Ustal ścieżkę do interpretera Python w venv (Windows vs Linux)
if sys.platform == "win32":
    PYTHON_PATH = os.path.join(PROJECT_ROOT, VENV_NAME, "Scripts", "python.exe")
else:
    PYTHON_PATH = os.path.join(PROJECT_ROOT, VENV_NAME, "bin", "python")

COMPONENT_DIR = os.path.join(PROJECT_ROOT, "warstwa_audio")

def main():
    if not os.path.isfile(PYTHON_PATH):
        print(f"[BŁĄD] Nie znaleziono Pythona w venv: {PYTHON_PATH}")
        print(f"       Upewnij się, że folder '{VENV_NAME}' istnieje w: {PROJECT_ROOT}")
        sys.exit(1)

    print(f"[INFO] Uruchamianie: Watus (agent głosowy)")
    print(f"[INFO] Python: {PYTHON_PATH}")
    print(f"[INFO] Katalog roboczy: {COMPONENT_DIR}")

    try:
        subprocess.run(
            [PYTHON_PATH, "run_watus.py"],
            cwd=COMPONENT_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[BŁĄD] Proces zakończył się kodem: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Zatrzymano przez użytkownika.")

if __name__ == "__main__":
    main()
