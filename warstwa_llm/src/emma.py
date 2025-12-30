"""
Moduł Pamięci EMMA - Używa biblioteki mem0 do pamięci konwersacyjnej.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ustaw zmienne środowiskowe dla mem0
# GOOGLE_API_KEY jest wymagany dla Gemini LLM i embeddera
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Zainicjalizuj pamięć mem0 z konfiguracją Gemini
_memory_instance = None

def _get_memory():
    """
    Leniwa inicjalizacja instancji pamięci mem0.
    
    Zapewnia, że obiekt Memory jest tworzony tylko raz i ponownie używany.
    Konfiguruje Gemini jako LLM i Embedder oraz Qdrant jako magazyn wektorów.
    
    Zwraca:
        Memory: Zainicjalizowana instancja pamięci mem0.
        
    Rzuca:
        Exception: Jeśli inicjalizacja się nie powiedzie.
        
    Hierarchia wywołań:
        emma.py -> retrieve_relevant_memories() -> _get_memory()
        emma.py -> consolidate_memory() -> _get_memory()
    """
    global _memory_instance
    if _memory_instance is None:
        from mem0 import Memory
        
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
                    "model": "models/text-embedding-004",
                    "embedding_dims": 768,  # Model Gemini text-embedding-004 używa 768 wymiarów
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "watus_memory",
                    "path": "./qdrant_mem0_data",  # Pamięć lokalna
                    "embedding_model_dims": 768,  # Musi pasować do wymiarów embeddera
                }
            }
        }
        
        try:
            _memory_instance = Memory.from_config(config)
            logger.info("mem0 Memory initialized successfully with Gemini.")
            print("DEBUG: mem0 Memory initialized successfully with Gemini.")
        except Exception as e:
            logger.error(f"Failed to initialize mem0 Memory: {e}")
            print(f"DEBUG: Failed to initialize mem0 Memory: {e}")
            raise
    
    return _memory_instance


def retrieve_relevant_memories(user_id: str, query: str, n_results: int = 3) -> str:
    """
    Pobiera wspomnienia istotne dla zapytania używając mem0.
    
    Argumenty:
        user_id (str): Identyfikator użytkownika (np. ID sesji).
        query (str): Zapytanie wyszukiwania (zazwyczaj ostatnia wiadomość użytkownika).
        n_results (int): Maksymalna liczba wyników do zwrócenia. Domyślnie 3.
    
    Zwraca:
        str: Sformatowany ciąg znaków z kontekstem pamięci lub pusty ciąg, jeśli brak wspomnień.
    
    Hierarchia wywołań:
        warstwa_llm/src/main.py -> process_question() -> retrieve_relevant_memories()
    """
    print(f"DEBUG: mem0: Retrieving memories for user {user_id} with query: {query}")
    logger.info(f"mem0: Retrieving memories for user {user_id} with query: {query}")
    
    try:
        memory = _get_memory()
        results = memory.search(query, user_id=user_id, limit=n_results)
        
        print(f"DEBUG: mem0: Search results: {results}")
        logger.info(f"mem0: Search results: {results}")
        
        if not results or not results.get("results"):
            print("DEBUG: mem0: No memories found.")
            logger.info("mem0: No memories found.")
            return ""
        
        memories = results["results"]
        context_parts = []
        
        for mem in memories:
            memory_text = mem.get("memory", "")
            if memory_text:
                context_parts.append(f"- {memory_text}")
        
        if not context_parts:
            return ""
        
        memory_context = "KONTEKST PAMIĘCI (z poprzednich rozmów z tym użytkownikiem):\n" + "\n".join(context_parts) + "\n"
        print(f"DEBUG: mem0: Returning memory context: {memory_context}")
        logger.info(f"mem0: Returning memory context: {memory_context}")
        return memory_context
        
    except Exception as e:
        logger.error(f"mem0 retrieval failed: {e}")
        print(f"DEBUG: mem0 retrieval failed: {e}")
        return ""


def consolidate_memory(user_id: str, user_input: str, ai_response: str):
    """
    Konsoliduje turę rozmowy w pamięci używając mem0.
    
    Dodaje wiadomość użytkownika i odpowiedź AI do systemu pamięci.
    mem0 automatycznie obsługuje ekstrakcję i przechowywanie istotnych faktów.
    
    Argumenty:
        user_id (str): Identyfikator użytkownika.
        user_input (str): Wiadomość użytkownika.
        ai_response (str): Odpowiedź AI.
        
    Zwraca:
        None
        
    Hierarchia wywołań:
        warstwa_llm/src/main.py -> process_question() -> consolidate_memory()
    """
    print(f"DEBUG: mem0: Consolidating memory for user {user_id}")
    logger.info(f"mem0: Consolidating memory for user {user_id}")
    
    try:
        memory = _get_memory()
        
        # Utwórz wiadomości w formacie oczekiwanym przez mem0
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ]
        
        # Dodaj do pamięci - mem0 automatycznie wyodrębni i zapisze istotne fakty
        result = memory.add(messages, user_id=user_id)
        
        print(f"DEBUG: mem0: Memory consolidation result: {result}")
        logger.info(f"mem0: Memory consolidation result: {result}")
        
    except Exception as e:
        logger.error(f"mem0 consolidation failed: {e}")
        print(f"DEBUG: mem0 consolidation failed: {e}")
