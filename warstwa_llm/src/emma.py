"""
EMMA Memory Module - Using mem0 library for conversational memory.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Set up environment variables for mem0
# GOOGLE_API_KEY is required for Gemini LLM and embedder
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize mem0 Memory with Gemini configuration
_memory_instance = None

def _get_memory():
    """
    Lazy initialization of mem0 Memory instance.
    
    Ensures that the Memory object is created only once and reused.
    Configures Gemini as the LLM and Embedder, and Qdrant as the vector store.
    
    Returns:
        Memory: The initialized mem0 Memory instance.
        
    Raises:
        Exception: If initialization fails.
        
    Call Hierarchy:
        emma.py -> retrieve_relevant_memories() -> _get_memory()
        emma.py -> consolidate_memory() -> _get_memory()
    """
    global _memory_instance
    if _memory_instance is None:
        from mem0 import Memory
        
        config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash-lite",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/text-embedding-004",
                    "embedding_dims": 768,  # Gemini text-embedding-004 uses 768 dimensions
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "watus_memory",
                    "path": "./qdrant_mem0_data",  # Local storage
                    "embedding_model_dims": 768,  # Must match embedder dims
                }
            }
        }
        
        try:
            _memory_instance = Memory.from_config(config)
            logger.info("mem0 Memory initialized successfully with Gemini.")
            print("DEBUG: mem0 Memory initialized successfully with Gemini.")
        except Exception as e:
            logger.error(f"Failed to initialize mem0 Memory: {e}")
            print(f"DEBUG: Failed to initialize mem0 Memory: {e}")
            raise
    
    return _memory_instance


def retrieve_relevant_memories(user_id: str, query: str, n_results: int = 3) -> str:
    """
    Retrieves memories relevant to the query using mem0.
    
    Args:
        user_id (str): The user identifier (e.g., session ID).
        query (str): The search query (usually the user's latest message).
        n_results (int): Maximum number of results to return. Defaults to 3.
    
    Returns:
        str: A formatted string with memory context, or empty string if no memories found.
    
    Call Hierarchy:
        warstwa_llm/src/main.py -> process_question() -> retrieve_relevant_memories()
    """
    print(f"DEBUG: mem0: Retrieving memories for user {user_id} with query: {query}")
    logger.info(f"mem0: Retrieving memories for user {user_id} with query: {query}")
    
    try:
        memory = _get_memory()
        results = memory.search(query, user_id=user_id, limit=n_results)
        
        print(f"DEBUG: mem0: Search results: {results}")
        logger.info(f"mem0: Search results: {results}")
        
        if not results or not results.get("results"):
            print("DEBUG: mem0: No memories found.")
            logger.info("mem0: No memories found.")
            return ""
        
        memories = results["results"]
        context_parts = []
        
        for mem in memories:
            memory_text = mem.get("memory", "")
            if memory_text:
                context_parts.append(f"- {memory_text}")
        
        if not context_parts:
            return ""
        
        memory_context = "MEMORY CONTEXT (from previous conversations with this user):\n" + "\n".join(context_parts) + "\n"
        print(f"DEBUG: mem0: Returning memory context: {memory_context}")
        logger.info(f"mem0: Returning memory context: {memory_context}")
        return memory_context
        
    except Exception as e:
        logger.error(f"mem0 retrieval failed: {e}")
        print(f"DEBUG: mem0 retrieval failed: {e}")
        return ""


def consolidate_memory(user_id: str, user_input: str, ai_response: str):
    """
    Consolidates the conversation turn into memory using mem0.
    
    Adds the user input and AI response to the memory system. 
    mem0 automatically handles extraction and storage of relevant facts.
    
    Args:
        user_id (str): The user identifier.
        user_input (str): The user's message.
        ai_response (str): The AI's response.
        
    Returns:
        None
        
    Call Hierarchy:
        warstwa_llm/src/main.py -> process_question() -> consolidate_memory()
    """
    print(f"DEBUG: mem0: Consolidating memory for user {user_id}")
    logger.info(f"mem0: Consolidating memory for user {user_id}")
    
    try:
        memory = _get_memory()
        
        # Create messages in the format expected by mem0
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ]
        
        # Add to memory - mem0 will automatically extract and store relevant facts
        result = memory.add(messages, user_id=user_id)
        
        print(f"DEBUG: mem0: Memory consolidation result: {result}")
        logger.info(f"mem0: Memory consolidation result: {result}")
        
    except Exception as e:
        logger.error(f"mem0 consolidation failed: {e}")
        print(f"DEBUG: mem0 consolidation failed: {e}")
