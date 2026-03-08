import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Uruchamia Wbudowany Webowy Interfejs (Inspector) dla platformy Google ADK.
    Narzędzie to pozwala na podgląd agentów, śledzenie logiki (Trace Tab) i historii zdarzeń.
    Wymaga aby terminal używał wirtualnego środowiska.
    """
    
    # Znajdź główny folder watus_jetson
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    agents_dir = project_root / "agents"
    
    # Określ ścieżkę do wirtualnego środowiska
    if os.name == 'nt':  # Windows
        adk_exe = os.path.join(project_root, "venv-win", "Scripts", "adk.exe")
    else:  # Linux/Mac
        adk_exe = os.path.join(project_root, "venv", "bin", "adk")

    if not os.path.exists(adk_exe):
        print(f"[ERROR] Nie znaleziono pliku ADK: {adk_exe}")
        print("Upewnij się, że google-adk jest zainstalowane (pip install google-adk).")
        sys.exit(1)

    if not agents_dir.exists():
        print(f"[ERROR] Brak folderu agentów: {agents_dir}")
        sys.exit(1)

    port = 8001
    print(f"[INFO] Uruchamianie Google ADK Web Interface na porcie {port}...")
    print(f"[INFO] Folder agentów: {agents_dir}")
    print(f"[INFO] Po starcie, otwórz http://127.0.0.1:{port} w przeglądarce.")
    print("-" * 50)
    
    try:
        # Uruchamia ADK web z CWD ustawionym na folder agents/
        # Dzięki temu w Inspectorze pojawią się TYLKO zarejestrowane agenty
        subprocess.run([adk_exe, "web", "--port", str(port)], cwd=str(agents_dir))
    except KeyboardInterrupt:
        print("\n[INFO] Zatrzymano przeglądarkę ADK Web.")
    except Exception as e:
        print(f"[ERROR] Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()

