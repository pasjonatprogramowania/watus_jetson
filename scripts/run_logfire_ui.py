import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Uruchamia Logfire Web UI.
    Logfire to domyślne narzędzie do głębokiego profilowania logów ustrukturyzowanych, uzywane mocno z Pydantic-AI.
    """
    
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    if os.name == 'nt':  # Windows
        logfire_exe = os.path.join(project_root, "venv-win", "Scripts", "logfire.exe")
    else:  # Linux/Mac
        logfire_exe = os.path.join(project_root, "venv", "bin", "logfire")

    if not os.path.exists(logfire_exe):
        print(f"[ERROR] Nie znaleziono pliku Logfire: {logfire_exe}")
        print("Upewnij się, że logfire jest zainstalowane.")
        sys.exit(1)

    print(f"[INFO] Uruchamianie Logfire Local Web UI...")
    print("-" * 50)
    
    try:
        subprocess.run([logfire_exe, "app"], cwd=project_root)
    except KeyboardInterrupt:
        print("\n[INFO] Zatrzymano przeglądarkę Logfire.")
    except Exception as e:
        print(f"[ERROR] Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()
