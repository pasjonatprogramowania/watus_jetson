import json
import time
import logging
from typing import Any, Dict

from google import genai
from google.genai import types

from ..config import (
    CURRENT_MODEL, DECISION_VECTOR_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPR,
    FUNNY_SYSTEM_PROMPT, CHOOSE_ACTION_SYSTEM_PROMPT, WARNING_SYSTEM_PROMPR,
    DUCKDUCKGO_TOOL, TOOL_REQUIRED
)
from ..types import DecisionVector, Action, WatusActiveState

logger = logging.getLogger(__name__)

client = genai.Client()

# Mockowe narzędzie DuckDuckGo
def duckduckgo_search_tool_adk(query: str) -> dict:
    """Wyszukuje informacje w internecie używając DuckDuckGo."""
    logger.info(f"Wykonywanie wyszukiwania dla: {query}")
    return {"status": "success", "results": "Wyniki wyszukiwania..."}

def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    print(f"Pytanie: {user_query}")
    print(f"Agent: {agent_name}")
    print(f"Odpowiedz: {response}")
    print(f"Czas odpowiedzi: {response_time:.3f} sekund")
    print("-" * 50)


def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type = str) -> tuple:
    """
    Uruchamia agenta Google GenAI API z logowaniem.
    Używa ustrukturyzowanego wyjścia JSON za pomocą natywnego response_schema.
    """
    
    generation_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
    )
    
    if output_type != str:
        generation_config.response_mime_type = "application/json"
        
        # Oczekiwany format do podpowiedzi jeśli pydantic schemy nie sa wspierane bezpośrednio, ale genai wspiera
        try:
            generation_config.response_schema = output_type
        except Exception as e:
            logger.warning(f"Could not bind response_schema to native pydantic type {e}, appending instruction")
            generation_config.system_instruction += f"\n\nOdpowiedź MUSI być w formacie poprawnego JSON pasującego do schematu: {output_type.__name__}."
            
    start_time = time.time()
    
    try:
        response = client.models.generate_content(
            model=CURRENT_MODEL,
            contents=content,
            config=generation_config
        )
    except Exception as e:
        logger.error(f"GenAI Error in {agent_name}: {e}")
        raise
        
    response_time = time.time() - start_time
    raw_output = response.text

    output = raw_output
    if output_type != str:
        try:
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()
            parsed_dict = json.loads(clean_json)
            output = output_type(**parsed_dict)
        except Exception as e:
            logger.error(f"Failed to parse GenAI output into {output_type.__name__}: {e}")
            raise ValueError(f"Oczekiwano formatu JSON dla klasy {output_type.__name__}.")
            
    log_llm_response(content, agent_name, str(output), response_time)
    return output, response_time


def get_decision_vector(content: str) -> DecisionVector:
    """
    Sprawdza pytanie pod kątem wszystkich kategorii decyzyjnych w jednym żądaniu via genai.
    """
    decision_vector, _ = run_agent_with_logging(
        content, "decision_vector_agent", DECISION_VECTOR_SYSTEM_PROMPT, DecisionVector
    )
    return decision_vector


def handle_default_response(content: str, decision_data: Dict[str, Any]) -> str:
    """
    Obsługuje odpowiedź domyślną (normalną konwersację) via genai.
    """
    tools = []
    if decision_data.get(TOOL_REQUIRED) == DUCKDUCKGO_TOOL:
        # Rejestrowanie narzędzia dla client API (uproszczone z ADK)
        tools.append(duckduckgo_search_tool_adk)
        
    generation_config = types.GenerateContentConfig(
        system_instruction=DEFAULT_SYSTEM_PROMPR,
        temperature=0.7,
        tools=tools if tools else None 
    )
    
    response = client.models.generate_content(
        model=CURRENT_MODEL,
        contents=content,
        config=generation_config
    )
    
    # Simple tool call handling simulation - the new genai client automatically does function calling loop
    # If the response contains text, we return it, if it's a function call, we should execute it but for now we'll fetch its text.
    return response.text


def handle_warning_response(content: str, decision_data: Dict[str, bool]) -> str:
    """
    Obsługuje ostrzeżenie dla niedozwolonych pytań via genai.
    """
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "warning_agent", WARNING_SYSTEM_PROMPR, str
    )
    return response


def handle_funny_response(content: str, decision_data: Dict[str, bool]) -> str:
    """
    Obsługuje żartobliwą odpowiedź dla niepoważnych pytań via genai.
    """
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "funny_agent", FUNNY_SYSTEM_PROMPT, str
    )
    return response


def handle_action_selection(content: str) -> str:
    """
    Obsługuje wybór i wykonanie akcji via genai.
    """
    from pydantic import BaseModel
    
    class ActionResponse(BaseModel):
        action_name: str
        
    custom_prompt = CHOOSE_ACTION_SYSTEM_PROMPT

    selected_action_obj, _ = run_agent_with_logging(
        content, "action_agent", custom_prompt, ActionResponse
    )
    
    try:
        selected_action = Action(selected_action_obj.action_name)
    except ValueError:
        selected_action = Action.follow_action

    if selected_action == Action.follow_action:
        WatusActiveState.following = True
        return "Rozpoczęto śledzenie."
    elif selected_action == Action.end_action:
        WatusActiveState.following = False
        return "Zakończono śledzenie."
    else:
        return "Nieznana akcja."


def check_context(content: str) -> Dict[str, Any]:
    return {"context": "Przykładowy kontekst na podstawie zapytania."}
