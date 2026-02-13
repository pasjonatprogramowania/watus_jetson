# warstwa_llm/src/vectordb.py
# This file is kept for backwards compatibility if needed, 
# but logic is moved to logic/vectordb.py

from .logic.vectordb import (
    initialize_vector_db as initialize_vector_db_logic,
    batch_process as batch_process_logic,
    add_to_vector_db as add_to_vector_db_logic,
    search_vector_db as search_vector_db_logic,
    return_collection as return_collection_logic
)

def initialize_vector_db():
    """
    Wrapper dla initialize_vector_db_logic.

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> initialize_vector_db() -> src.logic.vectordb.initialize_vector_db()
    """
    return initialize_vector_db_logic()

def batch_process():
    """
    Wrapper dla batch_process_logic.

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> batch_process() -> src.logic.vectordb.batch_process()
    """
    return batch_process_logic()

def add_to_vector_db(collection, results):
    """
    Wrapper dla add_to_vector_db_logic.

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> add_to_vector_db() -> src.logic.vectordb.add_to_vector_db()
    """
    return add_to_vector_db_logic(collection, results)

def search_vector_db(collection, query, n_results=3):
    """
    Wrapper dla search_vector_db_logic.

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> search_vector_db() -> src.logic.vectordb.search_vector_db()
    """
    return search_vector_db_logic(collection, query, n_results)

def return_collection():
    """
    Wrapper dla return_collection_logic.

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> return_collection() -> src.logic.vectordb.return_collection()
    """
    return return_collection_logic()

def main():
    """
    Główna funkcja skryptu (kompatybilność wsteczna).

    Hierarchia wywołań:
        warstwa_llm/src/vectordb.py -> main() -> initialize_vector_db()
        warstwa_llm/src/vectordb.py -> main() -> batch_process()
        warstwa_llm/src/vectordb.py -> main() -> add_to_vector_db()
    """
    _, collection = initialize_vector_db()
    results = batch_process()
    if results and collection:
        add_to_vector_db(collection, results)

if __name__ == "__main__":
    main()