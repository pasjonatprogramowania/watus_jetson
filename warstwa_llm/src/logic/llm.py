import json
import time
import logging
from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from ..config import (
    CURRENT_MODEL, DECISION_VECTOR_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPR,
    FUNNY_SYSTEM_PROMPT, CHOOSE_ACTION_SYSTEM_PROMPT, WARNING_SYSTEM_PROMPR,
    DUCKDUCKGO_TOOL, TOOL_REQUIRED
)
from ..types import DecisionVector, Action, WatusActiveState

logger = logging.getLogger(__name__)

def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    """
    Loguje odpowiedź LLM z pomiarem czasu na standardowe wyjście.

    Argumenty:
        user_query (str): Zapytanie użytkownika.
        agent_name (str): Nazwa agenta przetwarzającego zapytanie.
        response (str): Odpowiedź wygenerowana przez agenta.
        response_time (float): Czas przetwarzania w sekundach.

    Hierarchia wywołań:
        wywoływana przez funkcje run_agent_with_logging
    """
    print(f"Pytanie: {user_query}")
    print(f"Agent: {agent_name}")
    print(f"Odpowiedz: {response}")
    print(f"Czas odpowiedzi: {response_time:.3f} sekund")
    print("-" * 50)


def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:
    """
    Uruchamia agenta z logowaniem czasu odpowiedzi.

    Tworzy instancję agenta z podanym modelem i promptem, uruchamia go synchronicznie,
    a następnie loguje wynik i czas wykonania.

    Argumenty:
        content (str): Treść wejściowa dla agenta.
        agent_name (str): Nazwa agenta do celów logowania.
        system_prompt (str): Prompt systemowy definiujący zachowanie agenta.
        output_type (type): Oczekiwany typ danych wyjściowych (Pydantic model lub typ prosty).

    Zwraca:
        tuple: Krotka zawierająca (output, response_time).
               Output jest typu określonego w output_type.

    Hierarchia wywołań:
        src.logic.llm.py -> run_agent_with_logging() -> pydantic_ai.Agent.run_sync()
        src.logic.llm.py -> run_agent_with_logging() -> log_llm_response()
    """
    agent = Agent(
        model=CURRENT_MODEL,
        output_type=output_type,
        system_prompt=system_prompt
    )
    start_time = time.time()
    result = agent.run_sync(content)
    response_time = time.time() - start_time
    output = result.output
    log_llm_response(content, agent_name, str(output), response_time)
    return output, response_time


def get_decision_vector(content: str) -> DecisionVector:
    """
    Sprawdza pytanie pod kątem wszystkich kategorii decyzyjnych w jednym żądaniu.

    Wykorzystuje agenta `decision_vector_agent` do analizy treści i zwraca obiekt `DecisionVector`,
    który zawiera informacje o tym, czy pytanie jest dozwolone, czy wymaga akcji, itp.

    Argumenty:
        content (str): Treść zapytania użytkownika.

    Zwraca:
        DecisionVector: Obiekt zawierający wektor decyzyjny.

    Hierarchia wywołań:
        src.logic.llm.py -> get_decision_vector() -> run_agent_with_logging()
    """
    decision_vector, _ = run_agent_with_logging(
        content, "decision_vector_agent", DECISION_VECTOR_SYSTEM_PROMPT, DecisionVector
    )
    return decision_vector


def handle_default_response(content: str, decision_data: Dict[str, Any]) -> str:
    """
    Obsługuje odpowiedź domyślną (normalną konwersację).

    Jeśli wektor decyzyjny wskazuje na potrzebę użycia narzędzi (np. wyszukiwanie),
    są one dołączane do agenta.

    Argumenty:
        content (str): Treść zapytania użytkownika.
        decision_data (Dict[str, Any]): Dane z wektora decyzyjnego (np. wymagane narzędzia).

    Zwraca:
        str: Odpowiedź tekstowa wygenerowana przez model.

    Hierarchia wywołań:
        src.logic.llm.py -> handle_default_response() -> pydantic_ai.Agent.run_sync()
    """
    tools = []
    if decision_data.get(TOOL_REQUIRED) == DUCKDUCKGO_TOOL:
        tools.append(duckduckgo_search_tool())

    agent = Agent(
        model=CURRENT_MODEL,
        output_type=str,
        system_prompt=DEFAULT_SYSTEM_PROMPR,
        tools=tools
    )
    result = agent.run_sync(content)
    return result.output


def handle_warning_response(content: str, decision_data: Dict[str, bool]) -> str:
    """
    Obsługuje ostrzeżenie dla niedozwolonych pytań.

    Generuje grzeczną, ale stanowczą odmowę odpowiedzi lub prośbę o przeformułowanie pytania,
    zgodnie z polityką bezpieczeństwa.

    Argumenty:
        content (str): Treść zapytania użytkownika.
        decision_data (Dict[str, bool]): Dane decyzyjne (powód odrzucenia).

    Zwraca:
        str: Odpowiedź z ostrzeżeniem.

    Hierarchia wywołań:
        src.logic.llm.py -> handle_warning_response() -> run_agent_with_logging()
    """
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "warning_agent", WARNING_SYSTEM_PROMPR, str
    )
    return response


def handle_funny_response(content: str, decision_data: Dict[str, bool]) -> str:
    """
    Obsługuje żartobliwą odpowiedź dla niepoważnych pytań.

    Jeśli pytanie jest rozpoznane jako żart lub nonsens, agent generuje humorystyczną odpowiedź.

    Argumenty:
        content (str): Treść zapytania użytkownika.
        decision_data (Dict[str, bool]): Dane decyzyjne.

    Zwraca:
        str: Humorystyczna odpowiedź.

    Hierarchia wywołań:
        src.logic.llm.py -> handle_funny_response() -> run_agent_with_logging()
    """
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "funny_agent", FUNNY_SYSTEM_PROMPT, str
    )
    return response


def handle_action_selection(content: str) -> str:
    """
    Obsługuje wybór i wykonanie akcji (np. śledzenie).

    Analizuje intencję użytkownika dotyczącą akcji i aktualizuje stan globalny `WatusActiveState`.

    Argumenty:
        content (str): Treść zapytania użytkownika zawierająca polecenie.

    Zwraca:
        str: Komunikat potwierdzający wykonanie lub zakończenie akcji.

    Hierarchia wywołań:
        src.logic.llm.py -> handle_action_selection() -> run_agent_with_logging()
    """
    selected_action, _ = run_agent_with_logging(
        content, "action_agent", CHOOSE_ACTION_SYSTEM_PROMPT, Action
    )

    if selected_action == Action.follow_action:
        WatusActiveState.following = True
        return "Rozpoczęto śledzenie."
    elif selected_action == Action.end_action:
        WatusActiveState.following = False
        return "Zakończono śledzenie."
    else:
        return "Nieznana akcja."


def check_context(content: str) -> Dict[str, Any]:
    """
    Sprawdza kontekst dla danego pytania.

    (Funkcja placeholder) Obecnie zwraca przykładowy kontekst.

    Argumenty:
        content (str): Treść zapytania.

    Zwraca:
        Dict[str, Any]: Słownik z danymi kontekstowymi.
    """
    return {"context": "Przykładowy kontekst na podstawie zapytania."}
