"""
Punkt wejściowy warstwy wizji.

Uruchamia CVAgent - główny agent wizyjny odpowiedzialny za
detekcję obiektów, śledzenie i integrację z Lidarem.

Użycie:
    python main.py
"""

import os
import sys
from pathlib import Path

# Dodaj katalog główny do PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HOME", "/media/jetson/sd/hg")

# Ładowanie zmiennych środowiskowych z głównego pliku .env projektu
from dotenv import load_dotenv
_PROJECT_ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_PROJECT_ROOT_ENV, override=True)

from src.cv_agent import CVAgent


def main():
    """
    Główna funkcja uruchamiająca agenta wizyjnego.
    
    Tworzy instancję CVAgent i uruchamia główną pętlę
    przetwarzania wideo z zapisem do pliku i integracją z Lidarem.
    """
    agent = CVAgent()
    agent.run(
        save_video=True, 
        show_window=True, 
        consolidate_with_lidar=True
    )


if __name__ == "__main__":
    main()
