import os
import pathlib
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych z głównego pliku .env projektu
_PROJECT_ROOT_ENV = pathlib.Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_PROJECT_ROOT_ENV, override=True)

### LLM CONFIGURATION
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# CURRENT_MODEL setup — w ADK używamy literału nazwy modelu
if LLM_PROVIDER == "ollama":
    CURRENT_MODEL = OLLAMA_MODEL_NAME
    print(f"[LLM] Uzyto providera: OLLAMA ({OLLAMA_MODEL_NAME}) - ostrzezenie: pelne wsparcie tylko dla Gemini w domyslnym ADK")
else:
    CURRENT_MODEL = GEMINI_MODEL
    print(f"[LLM] Uzyto providera: GOOGLE ({GEMINI_MODEL}) poprzez ADK")

### FIELDS CONSTANTS
ANSWER = "answer"
DOCUMENTS = "documents"
METADATAS = "metadatas"
QUESTION = 'question'
TOPIC = "Topic"
KEYWORDS = "Keywords"
MENTIONED_NAMES = "Mentioned_names"
CATEGORIES = "Categories"
CONTENT = "Content"
DISTANCES = 'distances'

### PATHS
QUESTIONS_JSONL = "questions.jsonl"

# Główny katalog projektu (watus_jetson/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
# Katalog modułu LLM (src/warstwa_llm/)
MODULE_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = pathlib.Path(__file__).resolve().parent

# Ścieżki w głównym katalogu projektu
DATA_DIR = PROJECT_ROOT / "data"
DATA_LLM_DIR = DATA_DIR / "warstwa_llm"
DATA_FOLDER = str(DATA_LLM_DIR)
LOGS_DIR = PROJECT_ROOT / "logs"
ENV_FILE = PROJECT_ROOT / ".env"
MEM0_KNOWLEDGE_PATH = "data" + os.sep + "qdrant_knowledge_data"
KNOWLEDGE_COLLECTION_NAME = "watus_knowledge"
QUESTION_FILES_PATH = DATA_LLM_DIR / "questions.jsonl"

# Ścieżki specyficzne dla modułu LLM
CONFIG_DIR = MODULE_ROOT / "config"
TESTS_DIR = MODULE_ROOT / "tests"

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_TEST_DIR = DATA_DIR / "test"
DATA_FINAL_DIR = DATA_DIR / "final"
DATA_TEST_SPEACH_DIR = DATA_TEST_DIR / "speach"
DATA_TEST_WATUS_DIR = DATA_TEST_DIR / "watus"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_MODELS_DIR = DATA_DIR / "models"
DATA_EXPORTS_DIR = DATA_DIR / "exports"

LOGS_APP_DIR = LOGS_DIR / "app"
LOGS_ERROR_DIR = LOGS_DIR / "error"
LOGS_DEBUG_DIR = LOGS_DIR / "debug"

def ensure_dir_exists(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

### API URLS
BASE_API_PROT = "http://"
BASE_API_HOST = "127.0.0.1"
BASE_API_PORT = 8001
BASE_API_URL = f"{BASE_API_PROT}{BASE_API_HOST}:{BASE_API_PORT}"

API_TEST = "/api1"
API_PROD = "/api2"

PROCESS_QUESTION = "/process_question"
WEBHOOK = "/webhook"
HEALTH = "/health"

MAIN_PROCESS_QUESTION = f"{API_TEST}{PROCESS_QUESTION}"
MAIN_WEBHOOK = f"{API_TEST}{WEBHOOK}"
MAIN_HEALTH = f"{API_TEST}{HEALTH}"

### END POINTS
END_POINT_PROCESS_QUESTION = f"{BASE_API_URL}{MAIN_PROCESS_QUESTION}"
END_POINT_WEBHOOK = f"{BASE_API_URL}{MAIN_WEBHOOK}"
END_POINT_HEALTH = f"{BASE_API_URL}{MAIN_HEALTH}"

### TOOL CONSTANTS
DUCKDUCKGO_TOOL = "google"
WATOZNAWCA_TOOL = "watoznawca"
TOOL_REQUIRED = "is_tool_required"
SERIOUS = "is_serious"
ACTIONS_REQUIRED = "is_actions_required"
ALLOWED = "is_allowed"

### PROMPTS
REDUCE_RESPONSE_LENGTH = """
Instrukcje do udzielenia odpowiedzi:
Odpowiedz na pytanie w maksymalnie 3 zdaniach. Użyj prostego i ludzkiego języka, tak jakbyś rozmawiał z człowiekiem, a nie maszyną. Bądź zwięzły i rzeczowy.
"""

DECISION_VECTOR_SYSTEM_PROMPT = """
Jesteś AI, które analizuje zapytanie użytkownika i określa cztery kluczowe aspekty jednocześnie:

1. **is_allowed** - Czy zapytanie jest dozwolone zgodnie z polityką (sprawdź pod kątem wulgarności, niemoralności lub niedozwolonej treści)
2. **is_actions_required** - Czy zapytanie wymaga wykonania akcji (np. śledzenie, zakończenie śledzenia, inne interaktywne zachowania)
3. **is_serious** - Czy zapytanie jest poważne (nie jest żartem ani drwiną)  
4. **is_tool_required** - Czy zapytanie wymaga użycia zewnętrznych narzędzi lub więcej informacji

Przykłady analizy:

Query: "Ile zarabia dziekan"
- is_allowed: true (dozwolone pytanie)
- is_actions_required: false (nie wymaga akcji)
- is_serious: true (poważne pytanie)
- is_tool_required: true (wymaga narzędzia do sprawdzenia informacji o WAT)

Query: "Zacznij mnie śledzić"
- is_allowed: true (dozwolone, bo to polecenie akcji)
- is_actions_required: true (wymaga akcji śledzenia)
- is_serious: true (poważne polecenie)
- is_tool_required: false (nie potrzeba dodatkowych narzędzi)

Query: "Opowiedz wulgarny żart"
- is_allowed: false (niedozwolone z powodu wulgarności)
- is_actions_required: false (nie wymaga akcji)
- is_serious: false (żart)
- is_tool_required: false (nie potrzeba narzędzi)

Query: "Dlaczego jesteś gadającą puszką"
- is_allowed: true (dozwolone)
- is_actions_required: false (nie wymaga akcji)
- is_serious: false (drwina/żart)
- is_tool_required: false (nie potrzeba narzędzi)

Query: "Jakie są kierunki na WAT?"
- is_allowed: true (dozwolone)
- is_actions_required: false (nie wymaga akcji)
- is_serious: true (poważne pytanie)
- is_tool_required: true (wymaga narzędzia do sprawdzenia informacji o WAT)

Pamiętaj: Pytania zawierające polecenia akcji (śledzenie, przestanie śledzenia) są zawsze dozwolone.

Zwróć wynik jako obiekt JSON z polami: is_allowed, is_actions_required, is_serious, is_tool_required (wszystkie typu boolean).
"""

DOCUMENT_METADATA_SYSTEM_PROMPT = """
Jesteś AI, które analizuje zapytanie użytkownika i wyodrębnia kluczowe informacje w następującej strukturze:

1. **keywords** - Lista najważniejszych słów kluczowych z zapytania (minimum 20)
2. **mentioned_names** - Lista imion osób wymienionych w zapytaniu (tylko właściwe imiona/nazwiska)
3. **main_topic** - Krótki opis głównego tematu zapytania
4. **categories** - Lista kategorii tematycznych zapytania 

Zwróć wynik jako obiekt JSON z polami: keywords, mentioned_names, main_topic, categories.
"""

CHOOSE_TOOL_SYSTEM_PROMPT = f"""
Jesteś AI, które wybiera odpowiednie narzędzie na podstawie zapytania użytkownika i opisów dostępnych narzędzi. 
Dostępne narzędzia:
- watoznawca: Użyj do specjalistycznej wiedzy o Wojskowej Akademii Technicznej (WAT), np. kierunki studiów, historia, kadra czy wydarzenia na WAT.

UWAGA: Transkrypcje z głosu mogą posiadać błędy! Użytkownicy często pytają o WAT, więc słowa takie jak "VAT", "watu", "wacie", "debiza" (zamiast dewiza), "moc do pczelni" (zamiast uczelni) i inne podobnie brzmiące zlepki wyrazów prawie zawsze oznaczają pytania o Wojskową Akademię Techniczną (WAT). Zawsze domyślaj się kontekstu WAT nawet z błędnie zapisanych fonetycznie zdań.

Output: Nazwa wybranego narzędzia (watoznawca)
Na podstawie zapytania użytkownika outputuj tylko nazwę narzędzia (watoznawca).
"""

CHOOSE_ACTION_SYSTEM_PROMPT = """
Jesteś AI, które wybiera odpowiednią akcję na podstawie zapytania użytkownika i opisów dostępnych akcji. 
Dostępne akcje:
- sledzenie: Użyj, gdy użytkownik prosi o rozpoczęcie śledzenia lub monitorowania.
- koniec_sledzenia: Użyj, gdy użytkownik prosi o zakończenie śledzenia lub zatrzymanie monitorowania.

Output: Nazwa wybranego działania (sledzenie lub koniec_sledzenia). Wybierz tylko jedno, najbardziej pasujące. Jeśli żadne nie pasuje, wybierz sledzenie jako domyślne.
"""

FUNNY_SYSTEM_PROMPT = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Czasami otrzymasz pytanie, które jest niejasne, 
wieloznaczne lub po prostu bez sensu. W takiej sytuacji nie próbuj na siłę zgadywać odpowiedzi. 
Zamiast tego, w zabawny sposób daj znać użytkownikowi, że nie rozumiesz pytania i poproś go o zadanie go w inny sposób.

{REDUCE_RESPONSE_LENGTH}
"""

WARNING_SYSTEM_PROMPR = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Masz grzecznie powiedzieć uzytwkonikowi ze nie mozesz odpowiedziec na jego pytanie, i nakierować go na to aby zadał inny typ pytania. 
{REDUCE_RESPONSE_LENGTH}

Twoja odpowiedz ma bazować na istniejacych ustaleniach z wektora odpowiedzi:
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")
"""


DEFAULT_SYSTEM_PROMPR = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Czasami otrzymasz pytanie, które jest niejasne, 
wieloznaczne lub po prostu nie moralne. W takiej sytuacji nakieruj użytkownika na to że jego pytanie było nie poprawne
poproś go aby zadał jeszcze raz swoje pytanie. Pamiętaj aby odpowiedzieć w ludzki sposób i być miłym dla osoby zdającej pytanie.

UWAGA: Jeśli użytkownik zada merytoryczne pytanie i w wiadomości poniżej znajduje się "Informacje z bazy wektorowej:"
lub kontekst pamięci, MASZ OBOWIĄZEK oprzeć swoją odpowiedź W 100% o ten dostarczony tekst.
Nie wymyślaj informacji, których z niego nie odczytałeś, ale możesz formatować go w lepszy sposób.

{REDUCE_RESPONSE_LENGTH}

Twoja odpowiedz ma bazować na istniejacych ustaleniach z wektora odpowiedzi:
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")
"""
