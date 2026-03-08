from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .config import DUCKDUCKGO_TOOL, WATOZNAWCA_TOOL

class Action(str, Enum):
    follow_action = "sledzenie"
    end_action = "koniec_sledzenia"


class WatusActiveState(BaseModel):
    following: bool


class DecisionVector(BaseModel):
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: str | None = Field(DUCKDUCKGO_TOOL, description=f"Tool needed: {DUCKDUCKGO_TOOL} for web search, {WATOZNAWCA_TOOL} for WAT knowledge.")


class Question(BaseModel):
    content: str


class VectorSearchRequest(BaseModel):
    query: str
    n_results: int = 3


class VectorSearchResult(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class Answer(BaseModel):
    answer: str
    decisionVector: DecisionVector


class DocumentMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")


class ProcessingResult(BaseModel):
    filename: str
    document_content: Dict[str, Any]
    metadata: DocumentMetadata
    processing_id: str
