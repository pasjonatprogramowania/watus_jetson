"""
Skrypt tworzący wymaganą strukturę folderów projektu.
Tworzy puste katalogi i pliki placeholder, jeśli jeszcze nie istnieją.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


# === Foldery do utworzenia ===
FOLDERS = [
    # Modele
    os.path.join(PROJECT_ROOT, "models", "whisper"),
    os.path.join(PROJECT_ROOT, "models", "ECAPA"),
    os.path.join(PROJECT_ROOT, "models", "piper", "voices"),

    # Dane
    os.path.join(PROJECT_ROOT, "data", "watus_audio"),

]

# === Puste pliki do utworzenia (jeśli nie istnieją) ===
EMPTY_FILES = [
    os.path.join(PROJECT_ROOT, "data", "warstwa_llm", "questions.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "watus_audio", "dialog.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "watus_audio", "camera.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "watus_audio", "responses.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "watus_audio", "meldunki.jsonl"),
]


def main():
    print(f"[INFO] Katalog główny projektu: {PROJECT_ROOT}")
    print()

    created_dirs = 0
    skipped_dirs = 0

    for folder in FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            rel = os.path.relpath(folder, PROJECT_ROOT)
            print(f"  [+] Utworzono folder:  {rel}")
            created_dirs += 1
        else:
            skipped_dirs += 1

    created_files = 0
    skipped_files = 0

    for filepath in EMPTY_FILES:
        if not os.path.exists(filepath):
            # Upewnij się, że folder istnieje
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                pass  # pusty plik
            rel = os.path.relpath(filepath, PROJECT_ROOT)
            print(f"  [+] Utworzono plik:    {rel}")
            created_files += 1
        else:
            skipped_files += 1

    print()
    print(f"[INFO] Podsumowanie:")
    print(f"       Foldery:  {created_dirs} utworzonych, {skipped_dirs} już istniejących")
    print(f"       Pliki:    {created_files} utworzonych, {skipped_files} już istniejących")
    print(f"[INFO] Gotowe!")


if __name__ == "__main__":
    main()
