import os
import re
import json
import time
import uuid
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ..config import (
    CHROMADB_PATH, DATA_FOLDER, DOCUMENT_METADATA_SYSTEM_PROMPT,
    TOPIC, KEYWORDS, MENTIONED_NAMES, CATEGORIES,
    DOCUMENTS, METADATAS, DISTANCES
)
from ..types import DocumentMetadata, ProcessingResult
from .llm import run_agent_with_logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"
SUPPORTED_EXTENSIONS = [".jsonl"]

# Opóźnienie między zapytaniami API (sekundy).
# Darmowy tier Gemini: 10 req/min → min. 6s między zapytaniami.
API_REQUEST_DELAY = 7.0
MAX_RETRIES = 5


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
                # Spróbuj wyciągnąć czas oczekiwania z odpowiedzi API
                retry_match = re.search(r'retry\s*(?:in|Delay["\s:]*)\s*["\']?(\d+)', error_str, re.IGNORECASE)
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
    Inicjalizuje klienta ChromaDB i kolekcję.

    Tworzy trwałą instancję klienta ChromaDB w ścieżce `CHROMADB_PATH`
    i pobiera lub tworzy kolekcję o nazwie `COLLECTION_NAME`.

    Zwraca:
        tuple: Para (client, collection) lub (None, None) w przypadku błędu.

    Hierarchia wywołań:
        src.logic.vectordb.py -> initialize_vector_db() -> chromadb.PersistentClient()
    """
    try:
        if not Path(CHROMADB_PATH).exists():
            os.makedirs(CHROMADB_PATH, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMADB_PATH)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=DefaultEmbeddingFunction()
            )
            logger.info(f"Baza wektorowa zainicjalizowana w: {CHROMADB_PATH}")
            return client, collection
        
        # Jeśli istnieje, wciąż zwróć klienta/kolekcję
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=DefaultEmbeddingFunction()
        )
        return client, collection

    except Exception as e:
        logger.error(f"Błąd inicjalizacji bazy wektorowej: {str(e)}")
        return None, None


def add_to_vector_db(collection, results):
    """
    Dodaje przetworzone rozmowy do bazy wektorowej.

    Przygotowuje dokumenty, metadane i ID z obiektów `ProcessingResult`
    i wstawia je do podanej kolekcji ChromaDB.

    Argumenty:
        collection: Obiekt kolekcji ChromaDB.
        results (list): Lista obiektów `ProcessingResult` do dodania.

    Hierarchia wywołań:
        src.logic.vectordb.py -> add_to_vector_db() -> chromadb.Collection.add()
    """
    if not collection:
        logger.error("Brak dostępnej kolekcji bazy wektorowej")
        return

    documents = []
    metadatas = []
    ids = []

    for result in results:
        doc_text = json.dumps(result.document_content, ensure_ascii=False)
        documents.append(doc_text)
        metadata = {
            TOPIC: result.metadata.main_topic,
            KEYWORDS: ", ".join(result.metadata.keywords[:20]),
            CATEGORIES: ", ".join(result.metadata.categories[:5]),
            MENTIONED_NAMES: ", ".join(result.metadata.mentioned_names)
        }
        metadatas.append(metadata)
        ids.append(result.processing_id)

    try:
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Dodano {len(documents)} rozmów do bazy wektorowej")
    except Exception as e:
        logger.error(f"Błąd dodawania do bazy wektorowej: {e}")


def return_collection(collection_name: str = COLLECTION_NAME):
    """
    Zwraca instancję kolekcji ChromaDB.

    Tworzy klienta (jeśli trzeba) i pobiera kolekcję. Przydatne do operacji tylko do odczytu.

    Argumenty:
        collection_name (str): Nazwa kolekcji. Domyślnie `COLLECTION_NAME`.

    Zwraca:
        Collection: Obiekt kolekcji ChromaDB.

    Hierarchia wywołań:
        src.logic.vectordb.py -> return_collection() -> chromadb.PersistentClient()
    """
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return client.get_collection(
        name=collection_name,
        embedding_function=DefaultEmbeddingFunction()
    )


def search_vector_db(collection, query: str, n_results: int = 3):
    """
    Przeszukuje bazę wektorową w poszukiwaniu podobnych rozmów.

    Wykonuje zapytanie semantyczne do ChromaDB.

    Argumenty:
        collection: Obiekt kolekcji ChromaDB.
        query (str): Tekst zapytania.
        n_results (int): Liczba wyników do zwrócenia. Domyślnie 3.

    Zwraca:
        dict: Wyniki zapytania (dokumenty, metadane, odległości) lub None w przypadku błędu.

    Hierarchia wywołań:
        src.logic.vectordb.py -> search_vector_db() -> chromadb.Collection.query()
    """
    if not collection:
        logger.error("Brak dostępnej kolekcji bazy wektorowej")
        return None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        logger.info(f"Wykonano wyszukiwanie wektorowe dla zapytania: {query}")
        return results
    except Exception as e:
        logger.error(f"Błąd przeszukiwania bazy wektorowej: {e}")
        return None
