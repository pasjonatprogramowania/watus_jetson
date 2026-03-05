"""Testy procesorów detekcji — cache TTL i aktualizacja wpisów."""

import unittest


class TestDetectionProcessors(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import should_update_cache, update_person_cache
        self.should_update_cache = should_update_cache
        self.update_person_cache = update_person_cache

    def test_should_update_cache_new_track(self):
        cache = {}
        self.assertTrue(self.should_update_cache(cache, track_id=1, frame_index=0))

    def test_should_update_cache_fresh(self):
        cache = {1: {"last_frame": 50}}
        self.assertFalse(self.should_update_cache(cache, track_id=1, frame_index=100, cache_ttl=100))

    def test_should_update_cache_expired(self):
        cache = {1: {"last_frame": 0}}
        self.assertTrue(self.should_update_cache(cache, track_id=1, frame_index=200, cache_ttl=100))

    def test_update_person_cache_creates_entry(self):
        cache = {}
        result = self.update_person_cache(cache, track_id=5, frame_index=10, clothes_data=[{"label": "shirt"}])
        self.assertEqual(result["last_frame"], 10)
        self.assertEqual(result["clothes"], [{"label": "shirt"}])
        self.assertIn(5, cache)

    def test_update_person_cache_preserves_old_data(self):
        cache = {
            5: {
                "last_frame": 0, "clothes": [],
                "emotion": "Happy", "gender": "male",
                "age": "Adult 21-44", "lidar_data": None,
            }
        }
        result = self.update_person_cache(cache, track_id=5, frame_index=200, clothes_data=[])
        self.assertEqual(result["emotion"], "Happy")
        self.assertEqual(result["gender"], "male")

    def test_update_person_cache_with_classifier_data(self):
        cache = {}
        clf_data = {"emotion": "Sad", "gender": "female", "age": "Child 0-12"}
        result = self.update_person_cache(
            cache, track_id=1, frame_index=0,
            clothes_data=[], classifier_data=clf_data,
        )
        self.assertEqual(result["emotion"], "Sad")
        self.assertEqual(result["gender"], "female")
