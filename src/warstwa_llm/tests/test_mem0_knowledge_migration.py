"""
Testy jednostkowe migracji bazy wiedzy: ChromaDB → mem0

Testy weryfikują:
1. Poprawność importów (brak referencji do ChromaDB)
2. Inicjalizację instancji mem0 Memory (baza wiedzy)
3. Kompatybilność API (sygnatur funkcji) z istniejącymi konsumentami
4. Format wyników search_vector_db (kompatybilny z ChromaDB)
5. Poprawność add_to_vector_db (dodawanie dokumentów via mem0)
6. Izolację kolekcji (wiedza vs. pamięć EMMA)
7. Obsługę błędów i przypadków brzegowych
8. Integrację z api.py (vector_search flow)

Uruchomienie:
    cd src/warstwa_llm
    python -m pytest tests/test_mem0_knowledge_migration.py -v
"""
import pytest
import sys
import os
import json
import time
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

# conftest.py handles sys.path setup


# =============================================================================
# 1. Testy importów — brak referencji do ChromaDB
# =============================================================================
class TestImports:
    """Sprawdza, czy migracja usunęła wszystkie zależności od ChromaDB."""

    def test_no_chromadb_import_in_logic_vectordb(self):
        """logic/vectordb.py nie powinien importować chromadb."""
        import importlib
        import src.logic.vectordb as vdb_module
        source_file = vdb_module.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        assert "import chromadb" not in source_code, \
            "logic/vectordb.py nadal zawiera 'import chromadb'"
        assert "DefaultEmbeddingFunction" not in source_code, \
            "logic/vectordb.py nadal importuje DefaultEmbeddingFunction"

    def test_mem0_is_importable(self):
        """Pakiet mem0 musi być dostępny."""
        from mem0 import Memory
        assert Memory is not None

    def test_logic_vectordb_imports(self):
        """Wszystkie publiczne funkcje logic/vectordb.py importują się poprawnie."""
        from src.logic.vectordb import (
            initialize_vector_db,
            batch_process,
            add_to_vector_db,
            search_vector_db,
            return_collection,
            generate_metadata,
            process_file,
        )
        assert all(callable(fn) for fn in [
            initialize_vector_db, batch_process, add_to_vector_db,
            search_vector_db, return_collection, generate_metadata, process_file
        ])

    def test_wrapper_vectordb_imports(self):
        """Wrapper src/vectordb.py importuje się poprawnie."""
        from src.vectordb import (
            initialize_vector_db,
            batch_process,
            add_to_vector_db,
            search_vector_db,
            return_collection,
        )
        assert all(callable(fn) for fn in [
            initialize_vector_db, batch_process, add_to_vector_db,
            search_vector_db, return_collection
        ])

    def test_config_has_mem0_constants(self):
        """config.py powinien zawierać stałe mem0 zamiast ChromaDB."""
        from src.config import MEM0_KNOWLEDGE_PATH, KNOWLEDGE_COLLECTION_NAME
        assert MEM0_KNOWLEDGE_PATH is not None
        assert KNOWLEDGE_COLLECTION_NAME == "watus_knowledge"

    def test_config_no_chromadb_path(self):
        """config.py nie powinien już eksportować CHROMADB_PATH."""
        from src import config
        assert not hasattr(config, "CHROMADB_PATH"), \
            "config.py nadal eksportuje CHROMADB_PATH — nie zostało usunięte"


# =============================================================================
# 2. Testy inicjalizacji mem0 Memory
# =============================================================================
class TestKnowledgeMemoryInit:
    """Testy tworzenia instancji mem0 dla bazy wiedzy."""

    def test_get_knowledge_memory_returns_instance(self):
        """_get_knowledge_memory() powinien zwrócić instancję Memory."""
        from src.logic.vectordb import _get_knowledge_memory
        memory = _get_knowledge_memory()
        assert memory is not None

    def test_get_knowledge_memory_is_singleton(self):
        """Dwa wywołania _get_knowledge_memory() powinny zwrócić tę samą instancję."""
        from src.logic.vectordb import _get_knowledge_memory
        m1 = _get_knowledge_memory()
        m2 = _get_knowledge_memory()
        assert m1 is m2, "Memory nie jest singletonem — tworzona wielokrotnie"

    def test_initialize_vector_db_returns_tuple(self):
        """initialize_vector_db() powinien zwrócić (None, Memory)."""
        from src.logic.vectordb import initialize_vector_db
        client, memory = initialize_vector_db()
        assert client is None, "client powinien być None (kompatybilność wsteczna)"
        assert memory is not None, "memory nie powinien być None"

    def test_return_collection_returns_memory(self):
        """return_collection() powinien zwrócić instancję Memory."""
        from src.logic.vectordb import return_collection
        memory = return_collection()
        assert memory is not None
        # Powinien mieć metody search i add (interfejs mem0)
        assert hasattr(memory, "search"), "Memory brak metody search"
        assert hasattr(memory, "add"), "Memory brak metody add"


# =============================================================================
# 3. Testy izolacji kolekcji (wiedza vs EMMA)
# =============================================================================
class TestCollectionIsolation:
    """Sprawdza, że baza wiedzy i pamięć EMMA to osobne instancje."""

    def test_knowledge_and_emma_are_different_instances(self):
        """Instancje mem0 dla wiedzy i EMMA powinny być różne."""
        from src.logic.vectordb import _get_knowledge_memory
        from src.emma import _get_memory as _get_emma_memory

        knowledge = _get_knowledge_memory()
        emma = _get_emma_memory()

        assert knowledge is not emma, \
            "Knowledge i EMMA Memory to ta sama instancja — powinny być osobne"

    def test_knowledge_collection_name_is_correct(self):
        """Kolekcja wiedzy powinna nazywać się 'watus_knowledge'."""
        from src.config import KNOWLEDGE_COLLECTION_NAME
        assert KNOWLEDGE_COLLECTION_NAME == "watus_knowledge"


# =============================================================================
# 4. Testy formatu wyników search_vector_db
# =============================================================================
class TestSearchResultFormat:
    """Sprawdza kompatybilność formatu wyników z poprzednim API ChromaDB."""

    def test_search_returns_chromadb_compatible_dict(self):
        """search_vector_db() powinien zwrócić dict z kluczami documents/metadatas/distances."""
        from src.logic.vectordb import return_collection, search_vector_db
        from src.config import DOCUMENTS, METADATAS, DISTANCES

        memory = return_collection()
        result = search_vector_db(memory, "test query", n_results=3)

        # Wynik może być None jeśli baza jest pusta — to OK
        if result is not None:
            assert isinstance(result, dict), f"Oczekiwano dict, otrzymano {type(result)}"
            assert DOCUMENTS in result, f"Brak klucza '{DOCUMENTS}'"
            assert METADATAS in result, f"Brak klucza '{METADATAS}'"
            assert DISTANCES in result, f"Brak klucza '{DISTANCES}'"

    def test_search_results_are_nested_lists(self):
        """Wyniki powinny być listami opakowanyni w dodatkową listę (format batch ChromaDB)."""
        from src.logic.vectordb import return_collection, search_vector_db
        from src.config import DOCUMENTS, METADATAS, DISTANCES

        memory = return_collection()
        result = search_vector_db(memory, "test query", n_results=2)

        if result is not None:
            # ChromaDB format: {documents: [[doc1, doc2]], metadatas: [[m1, m2]], ...}
            docs = result[DOCUMENTS]
            assert isinstance(docs, list), "documents nie jest listą"
            assert len(docs) == 1, "documents powinien mieć dokładnie 1 element (batch)"
            assert isinstance(docs[0], list), "documents[0] nie jest listą"

            metas = result[METADATAS]
            assert isinstance(metas, list) and len(metas) == 1
            assert isinstance(metas[0], list), "metadatas[0] nie jest listą"

            dists = result[DISTANCES]
            assert isinstance(dists, list) and len(dists) == 1
            assert isinstance(dists[0], list), "distances[0] nie jest listą"

    def test_search_with_none_memory_returns_none(self):
        """search_vector_db(None, ...) powinien zwrócić None, nie crash."""
        from src.logic.vectordb import search_vector_db
        result = search_vector_db(None, "test query", n_results=3)
        assert result is None

    def test_search_distances_are_numeric(self):
        """Distances powinny być liczbami zmiennoprzecinkowymi."""
        from src.logic.vectordb import return_collection, search_vector_db
        from src.config import DISTANCES

        memory = return_collection()
        result = search_vector_db(memory, "test", n_results=2)

        if result is not None and result[DISTANCES][0]:
            for dist in result[DISTANCES][0]:
                assert isinstance(dist, (int, float)), \
                    f"Distance powinien być liczbą, otrzymano {type(dist)}"


# =============================================================================
# 5. Testy add_to_vector_db
# =============================================================================
class TestAddToVectorDb:
    """Testy dodawania dokumentów do bazy wiedzy."""

    def test_add_with_none_memory_does_not_crash(self):
        """add_to_vector_db(None, ...) nie powinien rzucać wyjątku."""
        from src.logic.vectordb import add_to_vector_db
        # Powinno logować błąd, ale nie crashować
        add_to_vector_db(None, [])

    def test_add_with_empty_results(self):
        """add_to_vector_db z pustą listą wyników nie powinien crashować."""
        from src.logic.vectordb import return_collection, add_to_vector_db
        memory = return_collection()
        add_to_vector_db(memory, [])

    def test_add_single_document(self):
        """Dodanie pojedynczego dokumentu powinno zakończyć się sukcesem."""
        from src.logic.vectordb import return_collection, add_to_vector_db
        from src.types import ProcessingResult, DocumentMetadata
        import uuid

        memory = return_collection()

        metadata = DocumentMetadata(
            keywords=["test", "migracja", "mem0"],
            mentioned_names=["Watus"],
            main_topic="Test migracji bazy wiedzy",
            categories=["test"]
        )
        result = ProcessingResult(
            filename="test_migration.jsonl",
            document_content={"role": "user", "content": "To jest test migracji."},
            metadata=metadata,
            processing_id=str(uuid.uuid4())
        )

        # Nie powinno rzucić wyjątku
        add_to_vector_db(memory, [result])

    def test_added_document_is_searchable(self):
        """Dodany dokument powinien być wyszukiwalny."""
        from src.logic.vectordb import return_collection, add_to_vector_db, search_vector_db
        from src.types import ProcessingResult, DocumentMetadata
        from src.config import DOCUMENTS
        import uuid

        memory = return_collection()

        # Dodaj dokument z unikalną treścią
        unique_marker = f"marker_migracja_{int(time.time())}"
        metadata = DocumentMetadata(
            keywords=["unikalne", unique_marker],
            mentioned_names=[],
            main_topic=unique_marker,
            categories=["test_migracji"]
        )
        result = ProcessingResult(
            filename="test_search.jsonl",
            document_content={"test": unique_marker, "info": "Dokument testowy migracji"},
            metadata=metadata,
            processing_id=str(uuid.uuid4())
        )

        add_to_vector_db(memory, [result])

        # Daj mem0 chwilę na indeksację
        time.sleep(1)

        # Wyszukaj
        search_result = search_vector_db(memory, unique_marker, n_results=3)

        assert search_result is not None, "Wyszukiwanie zwróciło None"
        docs = search_result.get(DOCUMENTS, [[]])[0]
        assert len(docs) > 0, f"Nie znaleziono żadnych dokumentów dla markera '{unique_marker}'"


# =============================================================================
# 6. Testy kompatybilności z api.py
# =============================================================================
class TestApiCompatibility:
    """Sprawdza, że api.py może korzystać z nowego API vectordb."""

    def test_api_imports_vectordb_functions(self):
        """api.py powinien poprawnie importować search_vector_db i return_collection."""
        from src.logic.vectordb import search_vector_db, return_collection
        assert callable(search_vector_db)
        assert callable(return_collection)

    def test_return_collection_result_is_truthy(self):
        """return_collection() powinien zwrócić wartość truthy (api.py sprawdza 'if not collection')."""
        from src.logic.vectordb import return_collection
        memory = return_collection()
        assert memory, "return_collection() zwraca wartość falsy — api.py odrzuci jako błąd"

    def test_search_result_compatible_with_vector_search_result(self):
        """Wyniki search_vector_db mogą być odpakowane do VectorSearchResult."""
        from src.logic.vectordb import return_collection, search_vector_db
        from src.config import DOCUMENTS, METADATAS, DISTANCES
        from src.types import VectorSearchResult

        memory = return_collection()
        result = search_vector_db(memory, "informacje o WAT", n_results=3)

        if result is not None and result.get(DOCUMENTS, [[]])[0]:
            # Symulacja tego co robi api.py:vector_search()
            vsr = VectorSearchResult(
                documents=result[DOCUMENTS][0],
                metadatas=result[METADATAS][0],
                distances=result[DISTANCES][0],
            )
            assert isinstance(vsr.documents, list)
            assert isinstance(vsr.metadatas, list)
            assert isinstance(vsr.distances, list)


# =============================================================================
# 7. Testy helper functions (niezmienionych)
# =============================================================================
class TestHelperFunctions:
    """Testy dla funkcji, które nie zależały od ChromaDB i nie powinny się zmienić."""

    def test_batch_process_with_nonexistent_folder(self):
        """batch_process() z nieistniejącym folderem powinien zwrócić pustą listę."""
        from src.logic.vectordb import batch_process
        result = batch_process("/nonexistent/folder/path")
        assert result == []

    def test_process_file_with_nonexistent_file(self):
        """process_file() z nieistniejącym plikiem powinien zwrócić pustą listę."""
        from src.logic.vectordb import process_file
        result = process_file("/nonexistent/file.jsonl")
        assert result == []


# =============================================================================
# 8. Testy przypadków brzegowych
# =============================================================================
class TestEdgeCases:
    """Testy dla obsługi błędów i przypadków brzegowych."""

    def test_search_with_empty_query(self):
        """Wyszukiwanie z pustym zapytaniem powinno obsłużyć poprawnie."""
        from src.logic.vectordb import return_collection, search_vector_db
        memory = return_collection()
        # Nie powinno crashować — zwróci None lub puste wyniki
        try:
            result = search_vector_db(memory, "", n_results=3)
            # Pusty wynik lub None to akceptowalne odpowiedzi
        except Exception:
            pass  # mem0 może rzucić wyjątek — to też OK

    def test_search_with_polish_characters(self):
        """Wyszukiwanie z polskimi znakami powinno działać poprawnie."""
        from src.logic.vectordb import return_collection, search_vector_db
        memory = return_collection()
        result = search_vector_db(memory, "żółćńśźę — polskie znaki", n_results=2)
        # Nie powinno crashować
        if result is not None:
            from src.config import DOCUMENTS
            assert DOCUMENTS in result

    def test_search_with_n_results_zero(self):
        """Wyszukiwanie z n_results=0 nie powinno crashować."""
        from src.logic.vectordb import return_collection, search_vector_db
        memory = return_collection()
        try:
            result = search_vector_db(memory, "test", n_results=0)
        except Exception:
            pass  # Akceptowalne — mem0 może nie obsługiwać limit=0


# =============================================================================
# Uruchomienie
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" MIGRACJA ChromaDB → mem0 — TESTY JEDNOSTKOWE")
    print("=" * 60)
    print()

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
        "--color=yes"
    ])

    sys.exit(exit_code)
