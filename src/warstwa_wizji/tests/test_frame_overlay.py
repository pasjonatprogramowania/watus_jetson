"""Testy funkcji rysujących nakładki na obrazie."""

import unittest
import numpy as np


class TestFrameOverlay(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import (
            draw_clothes_boxes,
            draw_stats_overlay,
            draw_weapon_boxes,
            draw_track_history,
            draw_lidar_info,
        )
        self.draw_clothes_boxes = draw_clothes_boxes
        self.draw_stats_overlay = draw_stats_overlay
        self.draw_weapon_boxes = draw_weapon_boxes
        self.draw_track_history = draw_track_history
        self.draw_lidar_info = draw_lidar_info

    def _make_frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_draw_clothes_boxes_modifies_frame(self):
        frame = self._make_frame()
        original = frame.copy()
        clothes = [{"box_norm": [0.1, 0.1, 0.9, 0.9], "label": "clothing", "conf": 0.85}]
        self.draw_clothes_boxes(frame, clothes, (100, 100, 300, 400))
        self.assertFalse(np.array_equal(frame, original))

    def test_draw_clothes_boxes_empty_data(self):
        frame = self._make_frame()
        original = frame.copy()
        self.draw_clothes_boxes(frame, [], (0, 0, 100, 100))
        np.testing.assert_array_equal(frame, original)

    def test_draw_stats_overlay_modifies_frame(self):
        frame = self._make_frame()
        original = frame.copy()
        self.draw_stats_overlay(frame, fps=30.0, brightness=0.5, display_mode="light")
        self.assertFalse(np.array_equal(frame, original))

    def test_draw_weapon_boxes_empty_data(self):
        frame = self._make_frame()
        original = frame.copy()
        self.draw_weapon_boxes(frame, [])
        np.testing.assert_array_equal(frame, original)

    def test_draw_weapon_boxes_with_data(self):
        # Biały frame, bo domyślny kolor ramki broni to czarny (0,0,0)
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        original = frame.copy()
        weapons = [{"box": [10, 10, 100, 100], "conf": 0.9, "label": "gun"}]
        self.draw_weapon_boxes(frame, weapons)
        self.assertFalse(np.array_equal(frame, original))

    def test_draw_track_history_less_than_two_points(self):
        frame = self._make_frame()
        original = frame.copy()
        self.draw_track_history(frame, [(100, 100)])
        np.testing.assert_array_equal(frame, original)

    def test_draw_track_history_with_points(self):
        frame = self._make_frame()
        original = frame.copy()
        points = [(100, 100), (200, 200), (300, 300)]
        self.draw_track_history(frame, points)
        self.assertFalse(np.array_equal(frame, original))

    def test_draw_lidar_info_modifies_frame(self):
        frame = self._make_frame()
        original = frame.copy()
        self.draw_lidar_info(frame, lidar_track_id=7, distance_meters=3.5, text_position=(50, 50))
        self.assertFalse(np.array_equal(frame, original))
