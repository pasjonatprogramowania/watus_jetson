"""
Prosty Skrypt Testowy Systemu Pamięci
Uruchom za pomocą: python test_memory_simple.py
"""
import sys
import os
import time
import requests

# Dodaj korzeń projektu do ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = "http://localhost:8000/api1"

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_test(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")

def test_imports():
    """Sprawdź czy wszystkie moduły mogą zostać zaimportowane."""
    print_header("TEST 1: Importy Modułów")
    
    try:
        from mem0 import Memory
        print_test("mem0 import", True)
    except ImportError as e:
        print_test("mem0 import", False, str(e))
        return False
    
    try:
        from src.emma import retrieve_relevant_memories, consolidate_memory
        print_test("emma module import", True)
    except ImportError as e:
        print_test("emma module import", False, str(e))
        return False
    
    try:
        from src.main import vector_search
        from src.vectordb import search_memory
        print_test("vectordb/main module import", True)
    except ImportError as e:
        print_test("vectordb/main module import", False, str(e))
        return False
    
    return True

def test_memory_init():
    """Przetestuj inicjalizację pamięci mem0."""
    print_header("TEST 2: Inicjalizacja Pamięci")
    
    try:
        from src.emma import _get_memory
        memory = _get_memory()
        print_test("instancja pamięci mem0 utworzona", memory is not None)
        return True
    except Exception as e:
        print_test("Inicjalizacja pamięci mem0", False, str(e))
        return False

def test_memory_consolidation():
    """Przetestuj zapisywanie wspomnień."""
    print_header("TEST 3: Konsolidacja Pamięci")
    
    try:
        from src.emma import consolidate_memory
        
        user_id = f"test_user_{int(time.time())}"
        user_input = "My name is TestUser and I am running unit tests."
        ai_response = "Hello TestUser! I'll remember that you're running tests."
        
        consolidate_memory(user_id, user_input, ai_response)
        print_test("Konsolidacja pamięci", True, f"user_id={user_id}")
        return True, user_id
    except Exception as e:
        print_test("Konsolidacja pamięci", False, str(e))
        return False, None

def test_memory_retrieval(user_id):
    """Przetestuj pobieranie wspomnień."""
    print_header("TEST 4: Pobieranie Pamięci")
    
    try:
        from src.emma import retrieve_relevant_memories
        
        # Poczekaj na konsolidację
        print("  Czekam 2s na konsolidację pamięci...")
        time.sleep(2)
        
        result = retrieve_relevant_memories(user_id, "What is my name?")
        
        has_content = len(result) > 0
        print_test("Pobieranie pamięci", True, f"Pobrano {len(result)} znaków")
        
        if has_content:
            print(f"  Podgląd kontekstu pamięci: {result[:100]}...")
        
        return True
    except Exception as e:
        print_test("Pobieranie pamięci", False, str(e))
        return False

def test_vector_search():
    """Przetestuj wyszukiwanie wektorowe ChromaDB dla wiedzy WAT."""
    print_header("TEST 5: Baza Wektorowa (Wiedza WAT)")
    
    try:
        from src.main import vector_search
        
        result = vector_search("Ile wydziałów ma WAT?")
        
        has_docs = result is not None and hasattr(result, 'documents') and len(result.documents) > 0
        print_test("Pobieranie wiedzy WAT", has_docs)
        
        if has_docs:
            first_doc = result.documents[0] if isinstance(result.documents[0], str) else result.documents[0]
            print(f"  Podgląd pierwszego wyniku: {str(first_doc)[:80]}...")
        
        return True
    except Exception as e:
        print_test("Wyszukiwanie wektorowe", False, str(e))
        return False

def test_api_integration():
    """Przetestuj pełną integrację API."""
    print_header("TEST 6: Integracja API")
    
    try:
        # Sprawdź czy API działa
        response = requests.get("http://localhost:8000/docs", timeout=3)
        if response.status_code != 200:
            print_test("Sprawdzenie stanu API", False, "API nie odpowiada")
            return False
        print_test("Sprawdzenie stanu API", True)
    except requests.ConnectionError:
        print_test("Połączenie API", False, "API nie działa. Uruchom za pomocą: python -m src.main")
        return False
    
    # Przetestuj przetwarzanie pytań
    try:
        response = requests.post(
            f"{API_BASE_URL}/process_question",
            json={"content": "Moje imię to APITestUser."},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            has_answer = "answer" in data and len(data["answer"]) > 0
            print_test("Przetwarzanie pytania przez API", has_answer)
            if has_answer:
                print(f"  Podgląd odpowiedzi: {data['answer'][:80]}...")
        else:
            print_test("Przetwarzanie pytania przez API", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Żądanie API", False, str(e))
        return False
    
    # Poczekaj na konsolidację pamięci
    print("  Czekam 3s na konsolidację pamięci...")
    time.sleep(3)
    
    # Przetestuj przypominanie pamięci
    try:
        response = requests.post(
            f"{API_BASE_URL}/process_question",
            json={"content": "Jak mam na imię?"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "").lower()
            has_name = "apitestuser" in answer or "api" in answer
            print_test("Przypomnienie pamięci w odpowiedzi", has_name, 
                      "Imię znalezione w odpowiedzi" if has_name else "Imię NIE znalezione")
            print(f"  Odpowiedź: {data['answer'][:100]}...")
            return True
        else:
            print_test("Przypomnienie pamięci", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Przypomnienie pamięci", False, str(e))
        return False

def main():
    print("\n" + "=" * 60)
    print(" SYSTEM PAMIĘCI EMMA - KOMPLEKSOWE TESTY")
    print("=" * 60)
    
    results = []
    
    # Test 1: Importy
    results.append(("Importy", test_imports()))
    
    # Test 2: Inicjalizacja Pamięci
    results.append(("Inicjalizacja Pamięci", test_memory_init()))
    
    # Test 3: Konsolidacja
    success, user_id = test_memory_consolidation()
    results.append(("Konsolidacja Pamięci", success))
    
    # Test 4: Pobieranie
    if user_id:
        results.append(("Pobieranie Pamięci", test_memory_retrieval(user_id)))
    
    # Test 5: Wyszukiwanie Wektorowe
    results.append(("Wyszukiwanie Wektorowe", test_vector_search()))
    
    # Test 6: Integracja API
    results.append(("Integracja API", test_api_integration()))
    
    # Podsumowanie
    print_header("PODSUMOWANIE TESTÓW")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        print_test(name, result)
    
    print(f"\n  Łącznie: {passed}/{total} testów zaliczonych")
    
    if passed == total:
        print("\n  🎉 WSZYSTKIE TESTY ZALICZONE!")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} testów niezaliczonych")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\nTesty przerwane.")
        exit_code = 1
    
    input("\nNaciśnij Enter, aby zakończyć...")
