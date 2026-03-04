import os
import re
import json
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import (
    MEM0_KNOWLEDGE_PATH, DATA_FOLDER, DOCUMENT_METADATA_SYSTEM_PROMPT,
    TOPIC, KEYWORDS, MENTIONED_NAMES, CATEGORIES,
    DOCUMENTS, METADATAS, DISTANCES, KNOWLEDGE_COLLECTION_NAME
)
from ..types import DocumentMetadata, ProcessingResult
from .llm import run_agent_with_logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".jsonl"]

# Opóźnienie między zapytaniami API (sekundy).
# Darmowy tier Gemini: 10 req/min → min. 6s między zapytaniami.
API_REQUEST_DELAY = 7.0
MAX_RETRIES = 5

# ──────────────────────────────────────────────
# Singleton instancji mem0 Memory (baza wiedzy)
# ──────────────────────────────────────────────
_knowledge_memory_instance = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def _get_knowledge_memory():
    """
    Leniwa inicjalizacja instancji pamięci mem0 dla bazy wiedzy (RAG).

    Używa kolekcji Qdrant 'watus_knowledge' (oddzielnej od EMMA 'watus_memory').
    Konfiguracja: Gemini LLM + Gemini Embedder + Qdrant vector store.

    Zwraca:
        Memory: Zainicjalizowana instancja mem0 Memory.

    Hierarchia wywołań:
        logic/vectordb.py -> initialize_vector_db() -> _get_knowledge_memory()
        logic/vectordb.py -> return_collection() -> _get_knowledge_memory()
    """
    global _knowledge_memory_instance
    if _knowledge_memory_instance is None:
        from mem0 import Memory

        # Upewnij się, że GOOGLE_API_KEY jest ustawiony (wymagany przez Gemini provider)
        google_api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

        # Wyłącz telemetrię mem0, żeby zapobiec błędom blokady plików Qdrant na Windowsie
        os.environ["MEM0_TELEMETRY"] = "false"

        knowledge_path = str(_PROJECT_ROOT / MEM0_KNOWLEDGE_PATH)
        os.makedirs(knowledge_path, exist_ok=True)

        config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash-lite",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/embedding-001",
                    "embedding_dims": 768,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": KNOWLEDGE_COLLECTION_NAME,
                    "path": knowledge_path,
                    "embedding_model_dims": 768,
                }
            }
        }

        try:
            # Obejście problemu Qdrant lock (WinError 32) przez telemetrię mem0
            import mem0.memory.main
            original_create = mem0.memory.main.VectorStoreFactory.create
            class MockTelemetryDB:
                def insert(self, *a, **k): pass
                def search(self, *a, **k): return []
                def update(self, *a, **k): pass
                def get(self, *a, **k): return None
                def list(self, *a, **k): return []
                def delete(self, *a, **k): pass
            def patched_create(provider, cfg):
                if getattr(cfg, 'collection_name', None) == 'mem0migrations':
                    return MockTelemetryDB()
                return original_create(provider, cfg)
            mem0.memory.main.VectorStoreFactory.create = patched_create

            _knowledge_memory_instance = mem0.memory.main.Memory.from_config(config)
            logger.info(f"mem0 Knowledge Memory zainicjalizowana (kolekcja: {KNOWLEDGE_COLLECTION_NAME}).")
        except Exception as e:
            logger.error(f"Błąd inicjalizacji mem0 Knowledge Memory: {e}")
            raise

    return _knowledge_memory_instance


def generate_metadata(conversation_content: str) -> DocumentMetadata:
    """
    Generuje metadane dla treści rozmowy używając agenta LLM.
    Zawiera mechanizm retry z exponential backoff dla błędów rate limit (429).

    Argumenty:
        conversation_content (str): Treść rozmowy w formacie tekstowym/JSON.

    Zwraca:
        DocumentMetadata: Wygenerowane metadane.

    Hierarchia wywołań:
        src.logic.vectordb.py -> generate_metadata() -> src.logic.llm.run_agent_with_logging()
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            metadata, _ = run_agent_with_logging(
                conversation_content, "metadata_agent", DOCUMENT_METADATA_SYSTEM_PROMPT, DocumentMetadata
            )
            return metadata
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                retry_match = re.search(r'retry\s*(?:in|Delay["\s:]*)\s*["\'"]?(\d+)', error_str, re.IGNORECASE)
                wait_time = int(retry_match.group(1)) + 5 if retry_match else 30 * attempt
                logger.warning(
                    f"Rate limit (429) — próba {attempt}/{MAX_RETRIES}. "
                    f"Czekam {wait_time}s..."
                )
                print(f"  [RATE LIMIT] Próba {attempt}/{MAX_RETRIES}. Czekam {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Przekroczono {MAX_RETRIES} prób z powodu rate limit.")


def process_file(file_path: str) -> List[ProcessingResult]:
    """
    Przetwarza pojedynczy plik rozmowy i zwraca listę wyników.

    Odczytuje plik linia po linii (format JSONL), generuje metadane dla każdego wpisu
    i tworzy obiekty `ProcessingResult`. Zawiera opóźnienie między zapytaniami API.

    Argumenty:
        file_path (str): Ścieżka do pliku do przetworzenia.

    Zwraca:
        List[ProcessingResult]: Lista przetworzonych dokumentów wraz z metadanymi i ID.

    Hierarchia wywołań:
        src.logic.vectordb.py -> process_file() -> generate_metadata()
    """
    logger.info(f"Przetwarzanie pliku: {file_path}")

    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        total = len(lines)
        for line_num, line in enumerate(lines, 1):
            try:
                document_content = json.loads(line)
                print(f"  [{line_num}/{total}] Generowanie metadanych...")
                metadata = generate_metadata(json.dumps(document_content))
                result = ProcessingResult(
                    filename=Path(file_path).name,
                    document_content=document_content,
                    metadata=metadata,
                    processing_id=str(uuid.uuid4())
                )
                results.append(result)

                # Opóźnienie między zapytaniami, aby nie przekroczyć rate limitu
                if line_num < total:
                    time.sleep(API_REQUEST_DELAY)

            except json.JSONDecodeError as e:
                logger.warning(f"Pomijanie nieprawidłowego JSON w linii {line_num} w {file_path}: {e}")
            except Exception as e:
                logger.error(f"Błąd przetwarzania linii {line_num} w {file_path}: {e}")

        logger.info(f"Pomyślnie przetworzono {len(results)} rozmów z {file_path}")
        return results
    except Exception as e:
        logger.error(f"Błąd przetwarzania pliku {file_path}: {e}")
        return []


def batch_process(folder: str = DATA_FOLDER):
    """
    Przetwarza wszystkie obsługiwane pliki w zadanym folderze.

    Przeszukuje folder w poszukiwaniu plików z rozszerzeniami zdefiniowanymi w `SUPPORTED_EXTENSIONS`
    i uruchamia dla nich `process_file`.

    Argumenty:
        folder (str): Ścieżka do folderu z danymi. Domyślnie `DATA_FOLDER`.

    Zwraca:
        list: Lista wszystkich wyników przetwarzania ze wszystkich plików.

    Hierarchia wywołań:
        src.logic.vectordb.py -> batch_process() -> process_file()
    """
    results = []
    folder_path = Path(folder)

    if not folder_path.exists():
        logger.error(f"Folder {folder} nie istnieje.")
        return results

    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))

    if not files_to_process:
        logger.warning(f"Brak obsługiwanych plików w folderze {folder}")
        return results

    logger.info(f"Znaleziono {len(files_to_process)} plików do przetworzenia")

    for file in files_to_process:
        file_results = process_file(str(file))
        if file_results:
            results.extend(file_results)

    logger.info(f"Pomyślnie przetworzono {len(results)} rozmów z {len(files_to_process)} plików")
    return results


def initialize_vector_db():
    """
    Inicjalizuje instancję mem0 Memory dla bazy wiedzy.

    Zwraca krotkę (None, memory) dla kompatybilności wstecznej
    z poprzednim API ChromaDB (client, collection).

    Zwraca:
        tuple: Para (None, Memory) lub (None, None) w przypadku błędu.

    Hierarchia wywołań:
        src.logic.vectordb.py -> initialize_vector_db() -> _get_knowledge_memory()
    """
    try:
        memory = _get_knowledge_memory()
        logger.info("Baza wiedzy (mem0) zainicjalizowana.")
        return None, memory
    except Exception as e:
        logger.error(f"Błąd inicjalizacji bazy wiedzy (mem0): {str(e)}")
        return None, None


def add_to_vector_db(memory, results):
    """
    Dodaje przetworzone rozmowy do bazy wiedzy mem0.

    Iteruje po obiektach ProcessingResult i dodaje każdy dokument
    do instancji mem0 Memory z odpowiednimi metadanymi.

    Argumenty:
        memory: Instancja mem0 Memory.
        results (list): Lista obiektów `ProcessingResult` do dodania.

    Hierarchia wywołań:
        src.logic.vectordb.py -> add_to_vector_db() -> mem0.Memory.add()
    """
    if not memory:
        logger.error("Brak dostępnej instancji mem0 Memory")
        return

    added_count = 0
    for result in results:
        doc_text = json.dumps(result.document_content, ensure_ascii=False)

        metadata = {
            TOPIC: result.metadata.main_topic,
            KEYWORDS: ", ".join(result.metadata.keywords[:20]),
            CATEGORIES: ", ".join(result.metadata.categories[:5]),
            MENTIONED_NAMES: ", ".join(result.metadata.mentioned_names)
        }

        try:
            memory.add(
                doc_text,
                user_id="knowledge_base",
                metadata=metadata,
            )
            added_count += 1
        except Exception as e:
            logger.error(f"Błąd dodawania dokumentu do mem0: {e}")

    logger.info(f"Dodano {added_count}/{len(results)} dokumentów do bazy wiedzy (mem0)")


def return_collection(collection_name: str = None):
    """
    Zwraca instancję mem0 Memory (baza wiedzy).

    Zastępuje poprzednią wersję zwracającą kolekcję ChromaDB.
    Parametr `collection_name` zachowany dla kompatybilności wstecznej, ale ignorowany
    (nazwa kolekcji jest ustawiona w konfiguracji mem0).

    Argumenty:
        collection_name (str): Ignorowany (zachowany dla kompatybilności).

    Zwraca:
        Memory: Instancja mem0 Memory.

    Hierarchia wywołań:
        src.logic.vectordb.py -> return_collection() -> _get_knowledge_memory()
    """
    return _get_knowledge_memory()


def search_vector_db(memory, query: str, n_results: int = 3):
    """
    Przeszukuje bazę wiedzy mem0 w poszukiwaniu podobnych dokumentów.

    Wykonuje zapytanie semantyczne via mem0.search() i mapuje wyniki
    na format kompatybilny z poprzednim API ChromaDB.

    Argumenty:
        memory: Instancja mem0 Memory.
        query (str): Tekst zapytania.
        n_results (int): Liczba wyników do zwrócenia. Domyślnie 3.

    Zwraca:
        dict: Wyniki zapytania w formacie ChromaDB-kompatybilnym:
              {documents: [[str, ...]], metadatas: [[dict, ...]], distances: [[float, ...]]}
              lub None w przypadku błędu.

    Hierarchia wywołań:
        src.logic.vectordb.py -> search_vector_db() -> mem0.Memory.search()
    """
    if not memory:
        logger.error("Brak dostępnej instancji mem0 Memory")
        return None

    try:
        raw_results = memory.search(
            query,
            user_id="knowledge_base",
            limit=n_results
        )
        logger.info(f"Wykonano wyszukiwanie w bazie wiedzy (mem0) dla zapytania: {query}")

        # Mapowanie wyników mem0 na format kompatybilny z ChromaDB
        documents = []
        metadatas = []
        distances = []

        memories = raw_results.get("results", [])
        for mem in memories:
            memory_text = mem.get("memory", "")
            documents.append(memory_text)

            # Odtworzenie metadanych z formatu mem0
            meta = mem.get("metadata", {}) or {}
            metadatas.append(meta)

            # mem0 zwraca score (wyższy = lepszy), ChromaDB zwracał distance (niższy = lepszy)
            # Mapujemy score → distance: distance = 1 - score
            score = mem.get("score", 0.0)
            distances.append(round(1.0 - score, 4))

        # Format ChromaDB: listy owinięte w dodatkową listę (batch query support)
        return {
            DOCUMENTS: [documents],
            METADATAS: [metadatas],
            DISTANCES: [distances]
        }

    except Exception as e:
        logger.error(f"Błąd przeszukiwania bazy wiedzy (mem0): {e}")
        return None
