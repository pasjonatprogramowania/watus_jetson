import logging
import uvicorn
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

from .config import (
    BASE_API_HOST, BASE_API_PORT, MAIN_PROCESS_QUESTION, MAIN_HEALTH, MAIN_WEBHOOK,
    ALLOWED, ACTIONS_REQUIRED, SERIOUS, TOOL_REQUIRED, DOCUMENTS, METADATAS, DISTANCES
)
from .types import Question, Answer, VectorSearchResult
from .logic.llm import (
    get_decision_vector, handle_warning_response, handle_funny_response,
    handle_default_response, handle_action_selection
)
from .logic.vectordb import search_vector_db, return_collection
from src.emma import retrieve_relevant_memories, consolidate_memory

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Asystent AI - Proces Przetwarzania - Zoptymalizowany",
    description="API demonstrujące zoptymalizowany schemat blokowy przetwarzania zapytań przez AI z early stopping.",
    version="1.2.0",
)

def vector_search(query: str) -> VectorSearchResult:
    """
    Wyszukuje wektory w bazie danych z ulepszoną obsługą błędów i logowaniem.

    Argumenty:
        query (str): Zapytanie do wyszukania.

    Zwraca:
        VectorSearchResult: Wynik wyszukiwania zawierający dokumenty i metadane.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> vector_search() -> src.logic.vectordb.return_collection()
        warstwa_llm/src/api.py -> vector_search() -> src.logic.vectordb.search_vector_db()
    """
    n_results = 3
    try:
        if n_results <= 0:
            logger.error(f"Nieprawidłowa liczba żądanych wyników: {n_results}")
            raise HTTPException(
                status_code=400,
                detail=f"Liczba wyników musi być dodatnia, otrzymano {n_results}"
            )
        collection = return_collection()
        if not collection:
            logger.error("Błąd inicjalizacji bazy wektorowej")
            raise HTTPException(
                status_code=500,
                detail="Baza wektorowa nie zainicjalizowana. Sprawdź konfigurację."
            )
        if not query or len(query.strip()) == 0:
            logger.error("Podano puste zapytanie wyszukiwania")
            raise HTTPException(
                status_code=400,
                detail="Zapytanie nie może być puste"
            )
        search_results = search_vector_db(collection, query, n_results)
        if not search_results or not search_results.get(DOCUMENTS, []):
            # Możemy tu nie rzucać błędu 404, tylko zwrócić pusty wynik,
            # zależnie od tego czy chcemy "fail hard". Tutaj rzucamy 404 dla zachowania logiki.
            logger.warning(f"Brak wyników dla zapytania: {query}")
            raise HTTPException(
                status_code=404,
                detail="Nie znaleziono pasujących dokumentów w bazie wektorowej"
            )
        
        logger.info(f"Znalezione wyniki: {len(search_results[DOCUMENTS][0])} dokumentów")
        
        # Wyciągamy pierwsze listy (bo query obsługuje batch)
        result = VectorSearchResult(
            documents=search_results[DOCUMENTS][0],
            metadatas=search_results[METADATAS][0],
            distances=search_results[DISTANCES][0]
        )
        logger.info("Wyszukiwanie wektorowe zakończone sukcesem")
        return result

    except HTTPException as http_err:
        logger.error(f"Błąd HTTP w wyszukiwaniu wektorowym: {http_err.detail}")
        raise
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas wyszukiwania wektorowego: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Nieoczekiwany błąd podczas wyszukiwania wektorowego: {str(e)}"
        )


def process_question(content: str) -> Answer:
    """
    Główna funkcja przetwarzająca pytanie użytkownika.

    Analizuje pytanie, podejmuje decyzje o akcjach/narzędziach, integruje pamięć (EMMA),
    wykonuje wyszukiwanie (RAG) i generuje odpowiedź.

    Argumenty:
        content (str): Treść pytania.

    Zwraca:
        Answer: Odpowiedź zawierająca tekst i wektor decyzyjny.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> process_question() -> src.logic.llm.get_decision_vector()
        warstwa_llm/src/api.py -> process_question() -> src.api.vector_search()
        warstwa_llm/src/api.py -> process_question() -> src.emma.retrieve_relevant_memories()
        warstwa_llm/src/api.py -> process_question() -> src.logic.llm.handle_*()
        warstwa_llm/src/api.py -> process_question() -> src.emma.consolidate_memory()
    """
    try:
        decision_vector = get_decision_vector(content)

        decision_data = {
            ALLOWED: decision_vector.is_allowed,
            ACTIONS_REQUIRED: decision_vector.is_actions_required,
            SERIOUS: decision_vector.is_serious,
            TOOL_REQUIRED: decision_vector.is_tool_required
        }

        if not decision_vector.is_allowed:
            response = handle_warning_response(content, decision_data)
            return Answer(answer=response, decisionVector=decision_vector)

        if not decision_vector.is_serious:
            response = handle_funny_response(content, decision_data)
            return Answer(answer=response, decisionVector=decision_vector)

        # Obsługa narzędzi / RAG
        if decision_data[TOOL_REQUIRED] is not None:
            try:
                vector_result = vector_search(content)
                additional_info = f"\nInformacje z bazy wektorowej: {''.join([doc for doc in vector_result.documents])}"
                content += additional_info
            except Exception as e:
                logger.warning(f"Vector search failed (non-critical): {e}")

        # --- EMMA Integration ---
        user_id = "default_user"
        try:
            memory_context = retrieve_relevant_memories(user_id, content)
            if memory_context:
                content += f"\n\n{memory_context}"
        except Exception as e:
            logger.error(f"EMMA retrieval failed: {e}")

        response = handle_default_response(content, decision_data)

        # Memory Consolidation
        try:
            consolidate_memory(user_id, content, response)
        except Exception as e:
            logger.error(f"EMMA consolidation failed: {e}")

        return Answer(answer=response, decisionVector=decision_vector)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing: {str(e)}")


@app.post(MAIN_PROCESS_QUESTION, response_model=Answer)
def process_question_endpoint(question: Question):
    """
    Endpoint do przetwarzania pytań.

    Argumenty:
        question (Question): Obiekt pytania.

    Zwraca:
        Answer: Odpowiedź systemu.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> POST /process_question -> process_question_endpoint() -> src.api.process_question()
    """
    return process_question(question.content)


@app.post("/api1/webhook")
def webhook_api1(payload: Dict[str, Any]):
    """
    Endpoint webhook dla agenta audio.

    Argumenty:
        payload (Dict[str, Any]): Dane wejściowe (może zawierać 'prompt').

    Zwraca:
        dict: Słownik z kluczem 'output'.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> POST /api1/webhook -> webhook_api1() -> src.api.process_question_endpoint()
    """
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Brak promptu")
    question = Question(content=prompt)
    answer = process_question_endpoint(question)
    return {"output": answer.answer}


@app.post(MAIN_WEBHOOK)
def webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint dla zewnętrznych integracji.

    Argumenty:
        payload (Dict[str, Any]): Dane wejściowe.

    Zwraca:
        dict: Słownik z kluczem 'output'.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> POST /webhook -> webhook() -> src.api.process_question_endpoint()
    """
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question_endpoint(question)
    return {"output": answer.answer}


@app.get(MAIN_HEALTH)
def health():
    """
    Endpoint sprawdzania stanu usługi.

    Zwraca:
        dict: Status usługi.

    Hierarchia wywołań:
        warstwa_llm/src/api.py -> GET /health -> health()
    """
    return {"ok": True}
