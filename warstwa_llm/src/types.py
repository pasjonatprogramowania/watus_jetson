from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .config import DUCKDUCKGO_TOOL, WATOZNAWCA_TOOL

class Action(str, Enum):
    """
    Enum reprezentujący możliwe akcje do podjęcia przez system.

    Wartości:
        follow_action: Rozpoczęcie śledzenia obiektu/osoby.
        end_action: Zakończenie śledzenia.

    Hierarchia/Użycie:
        Używany w src.logic.llm.handle_action_selection()
    """
    follow_action = "sledzenie"
    end_action = "koniec_sledzenia"


class WatusActiveState(BaseModel):
    """
    Model stanu aktywnego systemu Watus.

    Atrybuty:
        following (bool): Czy system aktualnie wykonuje akcję śledzenia.

    Hierarchia/Użycie:
        Używany w src.logic.llm.handle_action_selection() do śledzenia stanu.
    """
    following: bool


class DecisionVector(BaseModel):
    """
    [Dozwolone?, Czy_działanie?, Poważne?, Potrzeba_info?]
    
    Model danych decyzji podejmowanej przez agenta decyzyjnego.
    Określa sposób dalszego przetwarzania zapytania użytkownika.
    
    Atrybuty:
        is_allowed (bool): Czy zapytanie jest zgodne z polityką bezpieczeństwa.
        is_actions_required (bool): Czy zapytanie wymaga podjęcia akcji (np. sterowanie robotem).
        is_serious (bool): Czy zapytanie jest poważne (czy wymaga poważnej odpowiedzi).
        is_tool_required (str): Wymagane narzędzie ("google", "watoznawca", lub None).

    Hierarchia/Użycie:
        Zwracany przez src.logic.llm.get_decision_vector()
        Używany w src.api.process_question()
    """
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: str | None = Field(DUCKDUCKGO_TOOL, description=f"Tool needed: {DUCKDUCKGO_TOOL} for web search, {WATOZNAWCA_TOOL} for WAT knowledge.")


class Question(BaseModel):
    """
    Model reprezentujący pytanie użytkownika.

    Atrybuty:
        content (str): Treść pytania.

    Hierarchia/Użycie:
        Używany w endpointach API (src.api.process_question_endpoint)
        Używany w src.logic.llm
    """
    content: str


class VectorSearchRequest(BaseModel):
    """
    Model żądania wyszukiwania wektorowego.

    Atrybuty:
        query (str): Treść zapytania do wyszukania.
        n_results (int): Liczba żądanych wyników (domyślnie 3).

    Hierarchia/Użycie:
        Model pomocniczy dla operacji wyszukiwania wektorowego.
    """
    query: str
    n_results: int = 3


class VectorSearchResult(BaseModel):
    """
    Model wyniku wyszukiwania wektorowego.

    Atrybuty:
        documents (List[str]): Lista treści znalezionych dokumentów.
        metadatas (List[Dict[str, Any]]): Lista metadanych dla znalezionych dokumentów.
        distances (List[float]): Lista odległości (podobieństwa) dla wyników.

    Hierarchia/Użycie:
        Zwracany przez src.api.vector_search()
    """
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class Answer(BaseModel):
    """
    Model odpowiedzi systemu.

    Atrybuty:
        answer (str): Tekst odpowiedzi dla użytkownika.
        decisionVector (DecisionVector): Wektor decyzyjny określający kontekst odpowiedzi.

    Hierarchia/Użycie:
        Zwracany przez src.api.process_question()
        Zwracany przez endpoint src.api.process_question_endpoint()
    """
    answer: str
    decisionVector: DecisionVector


class DocumentMetadata(BaseModel):
    """
    Model metadanych dokumentu (rozmowy).

    Atrybuty:
        keywords (List[str]): Lista słów kluczowych.
        mentioned_names (List[str]): Lista imion rozmówców.
        main_topic (str): Główny temat rozmowy.
        categories (List[str]): Kategorie rozmowy.

    Hierarchia/Użycie:
        Generowany przez src.logic.vectordb.generate_metadata()
    """
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")


class ProcessingResult(BaseModel):
    """
    Model wyniku przetwarzania pliku/rozmowy.

    Atrybuty:
        filename (str): Nazwa pliku źródłowego.
        document_content (Dict[str, Any]): Zawartość dokumentu (rozmowy).
        metadata (DocumentMetadata): Wygenerowane metadane.
        processing_id (str): Unikalny identyfikator przetwarzania.

    Hierarchia/Użycie:
        Generowany przez src.logic.vectordb.process_file()
        Używany przez src.logic.vectordb.add_to_vector_db()
    """
    filename: str
    document_content: Dict[str, Any]
    metadata: DocumentMetadata
    processing_id: str
