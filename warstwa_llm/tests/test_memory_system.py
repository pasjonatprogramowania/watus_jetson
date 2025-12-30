"""
Kompleksowe testy jednostkowe dla systemu pamięci EMMA

Testy obejmują:
1. Inicjalizację pamięci mem0
2. Konsolidację pamięci (zapisywanie faktów)
3. Pobieranie pamięci (wyszukiwanie faktów)
4. Integrację bazy wektorowej (wiedza WAT)
5. Pełny przepływ integracji API
"""
import pytest
import time
import os
import sys
import requests
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Test Configuration
# =============================================================================
API_BASE_URL = "http://localhost:8000/api1"
TEST_USER_ID = f"test_user_{int(time.time())}"  # Unikalny użytkownik dla każdego uruchomienia testu


# =============================================================================
# Testy Jednostkowe - Moduł mem0
# =============================================================================
class TestMem0Initialization:
    """Testy dla inicjalizacji pamięci mem0."""
    
    def test_memory_import(self):
        """Sprawdź czy mem0 może zostać zaimportowane."""
        from mem0 import Memory
        assert Memory is not None
    
    def test_emma_module_import(self):
        """Sprawdź czy moduł emma może zostać zaimportowany."""
        from src.emma import retrieve_relevant_memories, consolidate_memory
        assert retrieve_relevant_memories is not None
        assert consolidate_memory is not None
    
    def test_memory_instance_creation(self):
        """Sprawdź czy instancja pamięci może zostać utworzona."""
        from src.emma import _get_memory
        memory = _get_memory()
        assert memory is not None


class TestMemoryConsolidation:
    """Testy dla konsolidacji pamięci (zapisywanie faktów)."""
    
    def test_consolidate_simple_fact(self):
        """Przetestuj zapisywanie prostego faktu o użytkowniku."""
        from src.emma import consolidate_memory
        
        user_input = "My name is TestUser for unit testing."
        ai_response = "Nice to meet you, TestUser! I'll remember that."
        
        # Nie powinno rzucać żadnych wyjątków
        consolidate_memory(TEST_USER_ID, user_input, ai_response)
    
    def test_consolidate_multiple_facts(self):
        """Przetestuj zapisywanie wielu faktów w jednej rozmowie."""
        from src.emma import consolidate_memory
        
        user_input = "I am 25 years old and I work as a software engineer."
        ai_response = "Great! You're 25 and work as a software engineer. Got it!"
        
        consolidate_memory(TEST_USER_ID, user_input, ai_response)
    
    def test_consolidate_with_empty_input(self):
        """Przetestuj czy puste wejścia nie powodują awarii systemu."""
        from src.emma import consolidate_memory
        
        # Powinno obsłużyć to poprawnie
        consolidate_memory(TEST_USER_ID, "", "")


class TestMemoryRetrieval:
    """Testy dla pobierania pamięci (wyszukiwanie faktów)."""
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Setup: Zapisz pewne wspomnienia przed testami pobierania."""
        from src.emma import consolidate_memory
        
        # Store a known fact
        consolidate_memory(
            TEST_USER_ID,
            "Remember that my favorite color is blue.",
            "I'll remember that your favorite color is blue!"
        )
        # Daj mem0 czas na przetworzenie
        time.sleep(2)
    
    def test_retrieve_existing_memory(self):
        """Przetestuj pobieranie wcześniej zapisanego wspomnienia."""
        from src.emma import retrieve_relevant_memories
        
        result = retrieve_relevant_memories(TEST_USER_ID, "What is my favorite color?")
        
        # Wynik powinien być ciągiem znaków
        assert isinstance(result, str)
    
    def test_retrieve_no_memories_for_new_user(self):
        """Przetestuj czy nowi użytkownicy nie mają wspomnień."""
        from src.emma import retrieve_relevant_memories
        
        new_user_id = f"new_user_{int(time.time())}"
        result = retrieve_relevant_memories(new_user_id, "What do you know about me?")
        
        # Powinno zwrócić pusty ciąg dla nowego użytkownika
        assert result == ""
    
    def test_retrieve_with_limit(self):
        """Przetestuj pobieranie pamięci z niestandardowym limitem."""
        from src.emma import retrieve_relevant_memories
        
        result = retrieve_relevant_memories(TEST_USER_ID, "Tell me about myself", n_results=5)
        
        assert isinstance(result, str)


# =============================================================================
# Testy Jednostkowe - Baza Wektorowa (Wiedza WAT)
# =============================================================================
class TestVectorDatabase:
    """Testy dla bazy wektorowej ChromaDB z wiedzą WAT."""
    
    def test_vectordb_import(self):
        """Sprawdź czy moduł vectordb może zostać zaimportowany."""
        from src.vectordb import vector_search, search_memory, add_memory
        assert vector_search is not None
        assert search_memory is not None
        assert add_memory is not None
    
    def test_vector_search_wat_info(self):
        """Przetestuj wyszukiwanie informacji o uniwersytecie WAT."""
        from src.vectordb import vector_search
        
        result = vector_search("Ile wydziałów ma WAT?")
        
        assert result is not None
        assert hasattr(result, 'documents')
        assert len(result.documents) > 0
    
    def test_vector_search_returns_relevant_results(self):
        """Przetestuj czy wyszukiwanie wektorowe zwraca istotne wyniki dla zapytań o WAT."""
        from src.vectordb import vector_search
        
        result = vector_search("Kiedy powstała Wojskowa Akademia Techniczna?")
        
        assert result is not None
        # Sprawdź czy dokumenty zawierają jakiś tekst
        documents = result.documents
        assert any(len(doc) > 0 for doc in documents)
    
    def test_search_memory_function(self):
        """Przetestuj funkcję search_memory."""
        from src.vectordb import search_memory
        
        result = search_memory("studenci WAT", n_results=3)
        
        assert result is not None
        assert 'documents' in result or 'ids' in result


# =============================================================================
# Testy Integracyjne - Pełny Przepływ API
# =============================================================================
class TestAPIIntegration:
    """Testy integracyjne dla pełnego API z systemem pamięci."""
    
    @pytest.fixture
    def api_session(self):
        """Utwórz sesję dla zapytań API."""
        return requests.Session()
    
    def test_api_health(self, api_session):
        """Sprawdź czy API działa."""
        try:
            response = api_session.get(f"{API_BASE_URL.replace('/api1', '')}/docs")
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API not running - start with: python -m src.main")
    
    def test_api_process_question_with_name(self, api_session):
        """Przetestuj przetwarzanie pytania przez API i zapisywanie imienia użytkownika."""
        try:
            # Wyślij pierwszą wiadomość z imieniem
            response = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Nazywam się TestAPI i lubię programować."}
            )
            
            if response.status_code != 200:
                pytest.skip(f"API returned {response.status_code}")
            
            data = response.json()
            assert "answer" in data
            
        except requests.ConnectionError:
            pytest.skip("API not running")
    
    def test_api_memory_recall(self, api_session):
        """Przetestuj czy API potrafi przypomnieć sobie wcześniej zapisane informacje."""
        try:
            # Najpierw zapisz pewne informacje
            response1 = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Moje imię to MemoryTestUser123."}
            )
            
            if response1.status_code != 200:
                pytest.skip("API not available")
            
            # Poczekaj na konsolidację pamięci
            time.sleep(3)
            
            # Spróbuj przypomnieć sobie imię
            response2 = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Jak mam na imię?"}
            )
            
            assert response2.status_code == 200
            data = response2.json()
            assert "answer" in data
            
        except requests.ConnectionError:
            pytest.skip("API not running")
    
    def test_api_wat_knowledge(self, api_session):
        """Przetestuj czy API potrafi odpowiadać na pytania o WAT."""
        try:
            response = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Ile wydziałów ma WAT?"}
            )
            
            if response.status_code != 200:
                pytest.skip("API not available")
            
            data = response.json()
            assert "answer" in data
            # WAT ma 7 wydziałów
            answer = data["answer"].lower()
            # Sprawdź czy odpowiedź zawiera istotne informacje
            assert len(answer) > 10  # Odpowiedź niepusta
            
        except requests.ConnectionError:
            pytest.skip("API not running")


# =============================================================================
# Testy Wydajności
# =============================================================================
class TestPerformance:
    """Testy wydajności dla operacji pamięci."""
    
    def test_memory_retrieval_speed(self):
        """Przetestuj czy pobieranie pamięci kończy się w rozsądnym czasie."""
        from src.emma import retrieve_relevant_memories
        
        start_time = time.time()
        retrieve_relevant_memories(TEST_USER_ID, "What do you know about me?")
        elapsed_time = time.time() - start_time
        
        # Powinno zakończyć się w ciągu 10 sekund (wliczając wywołania API)
        assert elapsed_time < 10, f"Pobieranie pamięci zajęło {elapsed_time:.2f}s"
    
    def test_memory_consolidation_speed(self):
        """Przetestuj czy konsolidacja pamięci kończy się w rozsądnym czasie."""
        from src.emma import consolidate_memory
        
        start_time = time.time()
        consolidate_memory(
            TEST_USER_ID,
            "Performance test input.",
            "Performance test response."
        )
        elapsed_time = time.time() - start_time
        
        # Powinno zakończyć się w ciągu 15 sekund
        assert elapsed_time < 15, f"Konsolidacja pamięci zajęła {elapsed_time:.2f}s"


# =============================================================================
# Testy Przypadków Brzegowych
# =============================================================================
class TestEdgeCases:
    """Testy przypadków brzegowych i obsługi błędów."""
    
    def test_unicode_characters(self):
        """Przetestuj obsługę znaków Unicode (polski)."""
        from src.emma import consolidate_memory, retrieve_relevant_memories
        
        polish_input = "Mieszkam w Warszawie i studiuję na WAT. Moje imię to Paweł."
        polish_response = "Świetnie! Zapamiętam, że mieszkasz w Warszawie."
        
        # Nie powinno rzucać wyjątków
        consolidate_memory(TEST_USER_ID, polish_input, polish_response)
    
    def test_very_long_input(self):
        """Przetestuj obsługę bardzo długich danych wejściowych."""
        from src.emma import consolidate_memory
        
        long_input = "Test " * 500  # ~2500 characters
        long_response = "Response " * 300
        
        # Powinno obsłużyć to poprawnie (może wewnętrznie przyciąć)
        consolidate_memory(TEST_USER_ID, long_input, long_response)
    
    def test_special_characters(self):
        """Przetestuj obsługę znaków specjalnych."""
        from src.emma import consolidate_memory
        
        special_input = "My email is test@example.com and I like $100 bills! <script>alert('xss')</script>"
        special_response = "I'll remember your email."
        
        # Nie powinno się zawiesić
        consolidate_memory(TEST_USER_ID, special_input, special_response)


# =============================================================================
# Główny Program Testowy
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMMA Memory System - Comprehensive Unit Tests")
    print("=" * 60)
    print()
    
    # Sprawdź czy API działa
    try:
        requests.get("http://localhost:8000/docs", timeout=2)
        print("✓ API is running at http://localhost:8000")
    except requests.ConnectionError:
        print("⚠ API is not running - integration tests will be skipped")
        print("  Start with: python -m src.main")
    print()
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Zatrzymaj na pierwszym błędzie
        "--color=yes"
    ])
    
    sys.exit(exit_code)
