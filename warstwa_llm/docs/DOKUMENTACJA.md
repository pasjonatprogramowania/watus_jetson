# Dokumentacja Modulu Warstwa LLM

## Wprowadzenie

Modul warstwa_llm stanowi komponent systemu WATUS odpowiedzialny za przetwarzanie jezyka naturalnego i generowanie odpowiedzi. Wykorzystuje modele LLM (Large Language Models) wraz z systemem RAG (Retrieval-Augmented Generation) opartym na bazie wektorowej ChromaDB. Modul jest zbudowany na frameworku Pydantic-AI, co zapewnia typowanie i walidacje danych.

---

## Architektura Systemu

### Glowne Komponenty

**Config (config.py)** - Centralna konfiguracja zawierajaca:
- Ustawienia modeli LLM (Gemini, OpenAI, Anthropic)
- Sciezki do katalgoow danych
- Prompty systemowe
- Stale konfiguracyjne

**API (api.py)** - Serwer FastAPI udostepniajacy endpointy:
- /process_question - Przetwarzanie pytan
- /webhook - Odbior webhookow
- /health - Sprawdzanie stanu

**EMMA (emma.py)** - Glowny agent konwersacyjny zawierajacy logike decyzyjna.

**VectorDB (logic/vectordb.py)** - Operacje na bazie wektorowej ChromaDB.

**LLM Logic (logic/llm.py)** - Funkcje wspolpracujace z modelami LLM.

### Hierarchia Wywolan

```
main.py
  -> FastAPI app
     -> /process_question endpoint
        -> EMMA.process()
           -> decide_vector() (analiza zapytania)
           -> choose_tool() / choose_action()
           -> search_vector_db() (wyszukiwanie RAG)
           -> generate_response()
```

---

## Konfiguracja

### Plik .env

Konfiguracja jest ladowana z pliku `.env` w katalogu glownym modulu.

### Zmienne LLM

| Zmienna | Domyslna | Opis |
|---------|----------|------|
| GEMINI_MODEL | - | Nazwa modelu Gemini (np. gemini-1.5-flash) |
| GEMINI_API_KEY | - | Klucz API Google Gemini |
| OPENAI_API_KEY | - | Klucz API OpenAI (opcjonalne) |
| OPENAI_MODEL | - | Model OpenAI (opcjonalne) |
| ANTHROPIC_API_KEY | - | Klucz API Anthropic (opcjonalne) |
| ANTHROPIC_MODEL | - | Model Anthropic (opcjonalne) |

### Sciezki

Sciezki sa automatycznie ustawiane wzgledem katalogu projektu:

| Stala | Wartosc | Opis |
|-------|---------|------|
| PROJECT_ROOT | .../warstwa_llm | Katalog glowny modulu |
| DATA_DIR | PROJECT_ROOT/data | Katalog danych |
| LOGS_DIR | PROJECT_ROOT/logs | Katalog logow |
| CHROMADB_PATH | PROJECT_ROOT/chroma_db | Baza wektorowa |

### API Endpoints

| Stala | Wartosc | Opis |
|-------|---------|------|
| BASE_API_PORT | 8000 | Port serwera API |
| MAIN_PROCESS_QUESTION | /api1/process_question | Endpoint zapytan |
| MAIN_WEBHOOK | /api1/webhook | Endpoint webhookow |
| MAIN_HEALTH | /api1/health | Endpoint health check |

---

## Struktura Folderow

Oczekiwana struktura katalogu warstwa_llm:

```
warstwa_llm/
|-- .env                        # Konfiguracja (WYMAGANE DO UTWORZENIA)
|-- .env.example                # Szablon konfiguracji
|-- .gitignore
|-- README.md
|-- requirements.txt
|
|-- docs/                       # Dokumentacja
|   |-- DOKUMENTACJA.md         # Ten plik
|
|-- chroma_db/                  # Baza wektorowa (TWORZONA AUTOMATYCZNIE)
|   |-- chroma.sqlite3          # Dane SQLite
|   |-- ...                     # Pliki indeksu
|
|-- data/                       # Dane
|   |-- raw/                    # Surowe dane wejsciowe
|   |-- processed/              # Przetworzone dane
|   |-- test/                   # Dane testowe
|   |   |-- questions.jsonl     # Pytania testowe
|   |   |-- speach/             # Dane mowy
|   |   |-- watus/              # Dane WATUS
|   |-- final/                  # Dane koncowe
|   |-- models/                 # Modele
|   |-- exports/                # Eksporty
|
|-- logs/                       # Logi
|   |-- app/                    # Logi aplikacji
|   |-- error/                  # Logi bledow
|   |-- debug/                  # Logi debugowania
|
|-- config/                     # Dodatkowa konfiguracja
|
|-- src/                        # Kod zrodlowy
|   |-- __init__.py
|   |-- api.py                  # Serwer FastAPI
|   |-- config.py               # Konfiguracja
|   |-- emma.py                 # Agent EMMA
|   |-- main.py                 # Punkt wejscia
|   |-- types.py                # Definicje typow
|   |-- vectordb.py             # Operacje VectorDB (prosty wrapper)
|   |
|   |-- logic/                  # Logika biznesowa
|       |-- llm.py              # Funkcje LLM
|       |-- vectordb.py         # Zaawansowane operacje ChromaDB
|
|-- tests/                      # Testy
    |-- test_*.py               # Pliki testowe
```

### Wymagane Foldery do Utworzenia

1. **Plik .env** - Skopiuj .env.example i uzupelnij GEMINI_API_KEY
2. **data/** - Utworz strukture podkatalogow jesli nie istnieje
3. **logs/** - Tworzone automatycznie przy pierwszym uruchomieniu

---

## System RAG (Retrieval-Augmented Generation)

### Architektura ChromaDB

System wykorzystuje ChromaDB jako baze wektorowa do przechowywania i wyszukiwania dokumentow.

### Kolekcja

- **Nazwa kolekcji**: "knowledge_base"
- **Funkcja embeddingu**: DefaultEmbeddingFunction (all-MiniLM-L6-v2)

### Schemat Metadanych Dokumentu

Kazdy dokument w kolekcji posiada nastepujace metadane:

| Pole | Typ | Opis |
|------|-----|------|
| Topic | string | Glowny temat dokumentu |
| Keywords | string | Slowa kluczowe (do 20, rozdzielone przecinkami) |
| Categories | string | Kategorie (do 5, rozdzielone przecinkami) |
| Mentioned_names | string | Wspomniane imiona/nazwiska |

### Przepyw RAG

1. Uzytkownik zadaje pytanie
2. System decyduje czy potrzebne jest narzedzie (is_tool_required)
3. Jesli tak, wybierane jest narzedzie (np. watoznawca)
4. Wykonywane jest wyszukiwanie semantyczne w ChromaDB
5. Zwrocone dokumenty sa dolaczane do kontekstu LLM
6. LLM generuje odpowiedz oparta na kontekscie

### Dodawanie Dokumentow

Dokumenty sa dodawane przez wywolanie:

```python
from src.logic.vectordb import initialize_vector_db, process_file, add_to_vector_db

client, collection = initialize_vector_db()
results = process_file("data/raw/documents.jsonl")
add_to_vector_db(collection, results)
```

---

## System Decyzyjny (Decision Vector)

System analizuje kazde zapytanie i generuje wektor decyzyjny:

### Pola Wektora

| Pole | Typ | Opis |
|------|-----|------|
| is_allowed | bool | Czy zapytanie jest dozwolone (brak wulgarnosci) |
| is_actions_required | bool | Czy wymaga akcji (np. sledzenie) |
| is_serious | bool | Czy jest powazne (nie zart) |
| is_tool_required | bool | Czy wymaga zewnetrznych narzedzi |

### Przyklady Decyzji

**Pytanie: "Ile zarabia dziekan?"**
- is_allowed: true (dozwolone)
- is_actions_required: false (brak akcji)
- is_serious: true (powazne)
- is_tool_required: true (wymaga RAG)

**Polecenie: "Zacznij mnie sledzic"**
- is_allowed: true (dozwolone)
- is_actions_required: true (wymaga akcji)
- is_serious: true (powazne)
- is_tool_required: false (brak potrzeby RAG)

---

## Prompty Systemowe

### DECISION_VECTOR_SYSTEM_PROMPT

Analizuje zapytanie i generuje wektor decyzyjny z czterema polami booleowskimi.

### DOCUMENT_METADATA_SYSTEM_PROMPT

Wyodrębnia metadane z dokumentu: keywords, mentioned_names, main_topic, categories.

### CHOOSE_TOOL_SYSTEM_PROMPT

Wybiera odpowiednie narzedzie na podstawie zapytania. Dostepne narzedzia:
- watoznawca - wiedza o WAT

### CHOOSE_ACTION_SYSTEM_PROMPT

Wybiera akcje do wykonania:
- sledzenie - rozpoczecie sledzenia
- koniec_sledzenia - zakonczenie sledzenia

### FUNNY_SYSTEM_PROMPT

Generuje humorystyczne odpowiedzi na niejednoznaczne lub zabawne pytania.

### WARNING_SYSTEM_PROMPT

Generuje uprzejme odmowy dla niedozwolonych zapytan.

### REDUCE_RESPONSE_LENGTH

Dodawany do innych promptow - ogranicza odpowiedz do maksymalnie 3 zdan.

---

## Typy Danych

### DocumentMetadata

```python
class DocumentMetadata(BaseModel):
    keywords: List[str]
    mentioned_names: List[str]
    main_topic: str
    categories: List[str]
```

### ProcessingResult

```python
class ProcessingResult(BaseModel):
    filename: str
    document_content: dict
    metadata: DocumentMetadata
    processing_id: str
```

### DecisionVector

```python
class DecisionVector(BaseModel):
    is_allowed: bool
    is_actions_required: bool
    is_serious: bool
    is_tool_required: bool
```

---

## Integracja z Modelami LLM

### Pydantic-AI

System wykorzystuje framework Pydantic-AI do integracji z modelami:

```python
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

GOOGLE_PROVIDER = GoogleProvider(api_key=GEMINI_API_KEY)
GOOGLE_MODEL = GoogleModel(GEMINI_MODEL, provider=GOOGLE_PROVIDER)
```

### Obslugiwane Modele

- **Google Gemini** - Domyslny, zalecany
- **OpenAI GPT** - Alternatywa (wymaga zmian w config.py)
- **Anthropic Claude** - Alternatywa (wymaga zmian w config.py)
- **Ollama** - Lokalne modele (eksperymentalne)

---

## API Endpoints

### POST /api1/process_question

Przetwarza pytanie uzytkownika.

**Request:**
```json
{
  "question": "Jakie sa kierunki na WAT?"
}
```

**Response:**
```json
{
  "answer": "Na WAT oferowanych jest wiele kierunkow...",
  "documents": [...],
  "metadatas": [...]
}
```

### POST /api1/webhook

Odbiera webhooks z zewnetrznych zrodel.

### GET /api1/health

Sprawdza stan serwera.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Rozwiazywanie Problemow

### Blad klucza API

1. Sprawdz czy GEMINI_API_KEY jest ustawiony w .env
2. Zweryfikuj poprawnosc klucza w Google Cloud Console

### ChromaDB nie inicjalizuje sie

1. Sprawdz uprawnienia do katalogu chroma_db/
2. Usun chroma_db/ i pozwol utworzyc od nowa

### Brak wynikow wyszukiwania

1. Sprawdz czy kolekcja zawiera dokumenty
2. Zweryfikuj format dokumentow wejsciowych (JSONL)
3. Sprawdz czy embeddingi zostaly poprawnie wygenerowane

### Wolne odpowiedzi

1. Rozważ mniejszy model (np. gemini-1.5-flash zamiast pro)
2. Zmniejsz n_results w search_vector_db
3. Sprawdz lacznosc sieciowa

---

## Ograniczenia

1. **Limit tokenow** - Modele maja limity kontekstu
2. **Koszty API** - Kazde zapytanie generuje koszty
3. **Jezyk** - Prompty sa w jezyku polskim
4. **Wiedza** - RAG wymaga wcześniejszego zaladowania dokumentow
5. **Latencja** - Odpowiedzi moga trwac kilka sekund
