import sys
import os
import unittest
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src", "warstwa_llm"))

# --- Importy z oryginalnego kodu (Pydantic-AI) ---
from src.logic.llm import get_decision_vector as old_get_decision_vector
from src.logic.llm import handle_action_selection as old_handle_action_selection
from src.logic.llm import handle_warning_response as old_handle_warning_response
from src.logic.llm import handle_funny_response as old_handle_funny_response
from src.logic.vectordb import generate_metadata as old_generate_metadata

# --- Importy ze zmigrowanego kodu (Google GenAI) ---
from adk_src.logic.llm import get_decision_vector as new_get_decision_vector
from adk_src.logic.llm import handle_action_selection as new_handle_action_selection
from adk_src.logic.llm import handle_warning_response as new_handle_warning_response
from adk_src.logic.llm import handle_funny_response as new_handle_funny_response
from adk_src.logic.vectordb import generate_metadata as new_generate_metadata


class TestLLMMigrationCompatibility(unittest.TestCase):
    
    def test_decision_vector_compatibility(self):
        """1. Weryfikator Intencji (Decision Vector)"""
        test_queries = [
            "Zacznij mnie śledzić",
            "Ile zarabia dziekan na WAT",
            "Opowiedz wulgarny żart",
            "Dlaczego jesteś taką gadającą puszką?"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                print(f"\n--- [DecisionVector] '{query}' ---")
                old_vector = old_get_decision_vector(query)
                new_vector = new_get_decision_vector(query)
                
                print(f"OLD: {old_vector.model_dump()}")
                print(f"NEW: {new_vector.model_dump()}")
                
                self.assertEqual(old_vector.is_allowed, new_vector.is_allowed)
                self.assertEqual(old_vector.is_actions_required, new_vector.is_actions_required)
                self.assertEqual(old_vector.is_serious, new_vector.is_serious)
                
                old_tool_needed = bool(old_vector.is_tool_required and old_vector.is_tool_required not in ['none', ''])
                new_tool_needed = bool(new_vector.is_tool_required and new_vector.is_tool_required not in ['none', ''])
                self.assertEqual(old_tool_needed, new_tool_needed)

    def test_action_selection_compatibility(self):
        """2. Analizator Akcji Wykonawczych (Action Selection)"""
        test_queries = [
            "Zacznij za mna jechac",
            "Przestań mnie śledzić"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                print(f"\n--- [ActionSelection] '{query}' ---")
                old_result = old_handle_action_selection(query)
                new_result = new_handle_action_selection(query)
                
                print(f"OLD ActionResult: {old_result}")
                print(f"NEW ActionResult: {new_result}")
                
                self.assertEqual(old_result, new_result)

    def test_funny_response_generation(self):
        """3. Generowanie żartobliwych odpowiedzi w formacie tekstowym"""
        query = "Dlaczego gadający śmietnik opowiada na pytania?"
        dec_data = {"is_allowed": True, "is_actions_required": False, "is_serious": False, "is_tool_required": False}
        
        print(f"\n--- [FunnyResponse] '{query}' ---")
        old_resp = old_handle_funny_response(query, dec_data)
        new_resp = new_handle_funny_response(query, dec_data)
        
        print(f"OLD: {old_resp}")
        print(f"NEW: {new_resp}")
        
        self.assertTrue(len(old_resp) > 5)
        self.assertTrue(len(new_resp) > 5)
        self.assertIsInstance(new_resp, str)

    def test_warning_response_generation(self):
        """4. Generowanie odpowiedzi ostrzegawczych w formacie tekstowym"""
        query = "Opowiedz mi wulgarny żart."
        dec_data = {"is_allowed": False, "is_actions_required": False, "is_serious": False, "is_tool_required": False}
        
        print(f"\n--- [WarningResponse] '{query}' ---")
        old_resp = old_handle_warning_response(query, dec_data)
        new_resp = new_handle_warning_response(query, dec_data)
        
        print(f"OLD: {old_resp}")
        print(f"NEW: {new_resp}")
        
        self.assertTrue(len(old_resp) > 5)
        self.assertTrue(len(new_resp) > 5)
        self.assertIsInstance(new_resp, str)
        
    def test_vectordb_metadata_generation(self):
        """5. Pydantic-AI -> GenAI: Struktura JSON Metadanych dla Vectordb"""
        content = json.dumps({"question": "Kto to jest Rektor WAT i ile ma lat?"})
        
        print(f"\n--- [VectorDB Metadata] '{content}' ---")
        old_meta = old_generate_metadata(content)
        new_meta = new_generate_metadata(content)
        
        print(f"OLD: {old_meta.model_dump()}")
        print(f"NEW: {new_meta.model_dump()}")
        
        self.assertTrue(len(old_meta.keywords) > 0)
        self.assertTrue(len(new_meta.keywords) > 0)
        
        # Test czy struktury tożsame
        self.assertEqual(type(old_meta.keywords), type(new_meta.keywords))
        self.assertEqual(type(old_meta.mentioned_names), type(new_meta.mentioned_names))
        self.assertEqual(type(old_meta.categories), type(new_meta.categories))


if __name__ == "__main__":
    unittest.main(verbosity=2)
