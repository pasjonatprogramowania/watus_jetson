"""Testy integracji LiDAR — odczyt danych, kąty azymutu, matching."""

import json
import os
import tempfile
import unittest


class TestLidarIntegration(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import (
            read_lidar_tracks,
            compute_lidar_angle,
            match_camera_to_lidar,
        )
        self.read_lidar_tracks = read_lidar_tracks
        self.compute_lidar_angle = compute_lidar_angle
        self.match_camera_to_lidar = match_camera_to_lidar

    def test_read_nonexistent_file(self):
        result = self.read_lidar_tracks("/tmp/__nonexistent_lidar__.json")
        self.assertEqual(result, [])

    def test_read_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = self.read_lidar_tracks(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def test_read_valid_jsonl(self):
        data = {"tracks": [{"id": 1, "last_position": [1.0, 2.0]}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps(data) + "\n")
            path = f.name
        try:
            result = self.read_lidar_tracks(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], 1)
        finally:
            os.unlink(path)

    def test_read_multi_line_takes_last(self):
        line1 = json.dumps({"tracks": [{"id": 1, "last_position": [0, 0]}]})
        line2 = json.dumps({"tracks": [{"id": 2, "last_position": [1, 1]}]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(line1 + "\n" + line2 + "\n")
            path = f.name
        try:
            result = self.read_lidar_tracks(path)
            self.assertEqual(result[0]["id"], 2)
        finally:
            os.unlink(path)

    def test_compute_lidar_angle_straight_ahead(self):
        angle = self.compute_lidar_angle(0.0, 5.0)
        self.assertAlmostEqual(angle, 0.0, delta=0.01)

    def test_compute_lidar_angle_right(self):
        angle = self.compute_lidar_angle(3.0, 3.0)
        self.assertAlmostEqual(angle, 45.0, delta=0.01)

    def test_compute_lidar_angle_left(self):
        angle = self.compute_lidar_angle(-3.0, 3.0)
        self.assertAlmostEqual(angle, -45.0, delta=0.01)

    def test_match_camera_to_lidar_basic(self):
        cam_angles = [(0, 10.0), (1, -10.0)]
        lidar_objs = [
            {"id": "A", "last_position": [1.74, 10.0]},
            {"id": "B", "last_position": [-1.74, 10.0]},
        ]
        matches = self.match_camera_to_lidar(cam_angles, lidar_objs)
        self.assertIn(0, matches)
        self.assertIn(1, matches)
        self.assertEqual(matches[0]["id"], "A")
        self.assertEqual(matches[1]["id"], "B")

    def test_match_camera_to_lidar_no_match(self):
        cam_angles = [(0, 80.0)]
        lidar_objs = [{"id": "X", "last_position": [0.0, 10.0]}]
        matches = self.match_camera_to_lidar(cam_angles, lidar_objs, max_angle_difference_deg=5.0)
        self.assertEqual(len(matches), 0)

    def test_match_camera_to_lidar_empty_inputs(self):
        self.assertEqual(self.match_camera_to_lidar([], []), {})
        self.assertEqual(self.match_camera_to_lidar([(0, 0.0)], []), {})
        self.assertEqual(self.match_camera_to_lidar([], [{"id": 1, "last_position": [0, 1]}]), {})
