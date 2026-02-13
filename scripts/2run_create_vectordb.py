"""
Skrypt startowy: Tworzenie / aktualizacja bazy wektorowej ChromaDB.
Przetwarza pliki JSONL z folderu data/ i dodaje je do bazy wektorowej.
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

COMPONENT_DIR = os.path.join(PROJECT_ROOT, "src", "warstwa_llm")

# Kod Python do wykonania wewnątrz venv z poprawnym sys.path
VECTORDB_SCRIPT = '''
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dodaj warstwa_llm do sys.path
sys.path.insert(0, os.getcwd())

from src.logic.vectordb import initialize_vector_db, batch_process, add_to_vector_db
from src.config import DATA_FOLDER, CHROMADB_PATH

print("=" * 60)
print("  TWORZENIE BAZY WEKTOROWEJ ChromaDB")
print("=" * 60)
print()
print(f"[INFO] Folder danych:   {DATA_FOLDER}")
print(f"[INFO] Ścieżka ChromaDB: {CHROMADB_PATH}")
print()

# 1. Inicjalizacja bazy
print("[1/3] Inicjalizacja bazy wektorowej...")
client, collection = initialize_vector_db()
if not collection:
    print("[BŁĄD] Nie udało się zainicjalizować bazy wektorowej!")
    sys.exit(1)
print(f"[OK]  Baza zainicjalizowana. Kolekcja: {collection.name}")
print(f"      Istniejące dokumenty: {collection.count()}")
print()

# 2. Przetwarzanie plików
print(f"[2/3] Przetwarzanie plików JSONL z: {DATA_FOLDER}")
results = batch_process(DATA_FOLDER)
if not results:
    print("[INFO] Brak nowych danych do przetworzenia.")
    print("[INFO] Upewnij się, że w folderze data/ znajdują się pliki .jsonl")
    sys.exit(0)
print(f"[OK]  Przetworzono {len(results)} dokumentów.")
print()

# 3. Dodawanie do bazy
print(f"[3/3] Dodawanie {len(results)} dokumentów do bazy wektorowej...")
add_to_vector_db(collection, results)
print(f"[OK]  Baza wektorowa zaktualizowana. Łącznie dokumentów: {collection.count()}")
print()
print("=" * 60)
print("  GOTOWE!")
print("=" * 60)
'''


def main():
    if not os.path.isfile(PYTHON_PATH):
        print(f"[BŁĄD] Nie znaleziono Pythona w venv: {PYTHON_PATH}")
        print(f"       Upewnij się, że folder '{VENV_NAME}' istnieje w: {PROJECT_ROOT}")
        sys.exit(1)

    print(f"[INFO] Uruchamianie: Tworzenie bazy wektorowej")
    print(f"[INFO] Python: {PYTHON_PATH}")
    print(f"[INFO] Katalog roboczy: {COMPONENT_DIR}")
    print()

    try:
        subprocess.run(
            [PYTHON_PATH, "-c", VECTORDB_SCRIPT],
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
