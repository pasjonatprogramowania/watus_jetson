"""Testy statycznej metody build_add_info klasy PersonAnalyzer."""

import unittest


class TestPersonAnalyzerBuildAddInfo(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji.src.person_analyzer import PersonAnalyzer
        self.build_add_info = PersonAnalyzer.build_add_info

    def test_empty_cache_entry(self):
        entry = {"last_frame": 10, "clothes": [], "emotion": None, "gender": None, "age": None}
        result = self.build_add_info(entry)
        self.assertEqual(result, [])

    def test_full_cache_entry(self):
        entry = {
            "last_frame": 10,
            "clothes": [{"label": "shirt"}],
            "emotion": "Happy",
            "gender": "male",
            "age": "Adult 21-44",
        }
        result = self.build_add_info(entry)
        self.assertTrue(any("gender" in item for item in result))
        self.assertTrue(any("age" in item for item in result))
        self.assertTrue(any("emotion" in item for item in result))
        self.assertTrue(any("clothes" in item for item in result))

    def test_partial_cache_entry(self):
        entry = {"last_frame": 10, "gender": "female"}
        result = self.build_add_info(entry)
        self.assertEqual(len(result), 1)
        self.assertIn("gender", result[0])
