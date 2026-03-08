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
from .emma import retrieve_relevant_memories, consolidate_memory

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Asystent AI - Proces Przetwarzania - Zoptymalizowany (Google ADK Version)",
    description="API demonstrujące przetwarzanie zapytań przez AI oparte z frameworka Google ADK zamiast Pydantic-AI.",
    version="2.0.0",
)

def vector_search(query: str) -> VectorSearchResult:
    """
    Wyszukuje wektory w bazie danych z ulepszoną obsługą błędów i logowaniem.

    Argumenty:
        query (str): Zapytanie do wyszukania.

    Zwraca:
        VectorSearchResult: Wynik wyszukiwania zawierający dokumenty i metadane.

    Hierarchia wywołań:
        warstwa_llm/adk_src/api.py -> vector_search() -> logic.vectordb.return_collection()
        warstwa_llm/adk_src/api.py -> vector_search() -> logic.vectordb.search_vector_db()
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


def _extract_speaker_id(content: str) -> str:
    """
    Wyciąga speaker_id z opisu zapytania (tag [SPEAKER=...]).
    Fallback do 'default_user' jeśli tag nie istnieje.
    """
    import re
    match = re.search(r'\[SPEAKER=(\S+?)\]', content)
    if match:
        spk = match.group(1)
        if spk and spk != "unknown":
            return spk
    return "default_user"


def process_question(content: str) -> Answer:
    """
    Główna funkcja przetwarzająca pytanie użytkownika z Google ADK LLM Agentem za kulisami.
    """
    original_content = content
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

        # Obsługa narzędzi / RAG (Zawsze wyciągamy kontekst z bazy dla podstawowych pytań)
        try:
            vector_result = vector_search(content)
            extracted_docs = []
            for doc in vector_result.documents:
                if isinstance(doc, dict):
                    extracted_docs.append(doc.get("memory", str(doc)))
                else:
                    extracted_docs.append(str(doc))
                    
            additional_info = f"\nInformacje z bazy wektorowej:\n" + "\n".join(extracted_docs)
            content += additional_info
        except Exception as e:
            logger.warning(f"Vector search failed (non-critical): {e}")

        # --- EMMA Integration (z powiązaniem speaker_id) ---
        user_id = _extract_speaker_id(original_content)
        try:
            memory_context = retrieve_relevant_memories(user_id, original_content)
            if memory_context:
                content += f"\n\n{memory_context}"
        except Exception as e:
            logger.error(f"EMMA retrieval failed: {e}")

        response = handle_default_response(content, decision_data)

        # Memory Consolidation: Zapisujemy czysty input bez kontekstu
        try:
            consolidate_memory(user_id, original_content, response)
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
        warstwa_llm/adk_src/api.py -> POST /process_question -> process_question_endpoint() -> process_question()
    """
    return process_question(question.content)


@app.post("/api1/webhook")
def webhook_api1(payload: Dict[str, Any]):
    """
    Endpoint webhook dla agenta audio.
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
    """
    return {"ok": True}
