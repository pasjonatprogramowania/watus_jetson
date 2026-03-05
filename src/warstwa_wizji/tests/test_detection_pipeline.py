"""Testy klasy DetectionPipeline."""

import math
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np


class TestDetectionPipeline(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import DetectionPipeline
        self.pipeline = DetectionPipeline()

    def test_initial_state(self):
        self.assertFalse(self.pipeline.weapon_detection_enabled)
        self.assertEqual(self.pipeline.mode, "light")
        self.assertIsInstance(self.pipeline.track_history, defaultdict)

    def test_toggle_weapon_detection(self):
        self.assertFalse(self.pipeline.weapon_detection_enabled)
        self.pipeline.toggle_weapon_detection()
        self.assertTrue(self.pipeline.weapon_detection_enabled)
        self.pipeline.toggle_weapon_detection()
        self.assertFalse(self.pipeline.weapon_detection_enabled)

    def test_calc_fps_positive(self):
        import time
        self.pipeline.fps_params["t_prev"] = time.time() - 0.033
        fps = self.pipeline.calc_fps()
        self.assertGreater(fps, 0)
        self.assertTrue(math.isfinite(fps))

    def test_process_frame_no_dets(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.pipeline.process_frame(
            frame_bgr=frame, dets=None, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=MagicMock(), model_manager=MagicMock(),
            show_window=False, verbose_window=False,
        )
        detections, frame_vis = result
        self.assertIsInstance(detections, dict)
        self.assertIn("objects", detections)
        self.assertEqual(detections["objects"], [])
        self.assertEqual(detections["countOfPeople"], 0)
        np.testing.assert_array_equal(frame_vis, frame)

    def test_process_frame_with_mock_dets(self):
        """Symulacja detekcji z mockiem obiektów YOLO."""
        import torch
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        mock_boxes = MagicMock()
        mock_boxes.is_track = True
        mock_boxes.xywh.cpu.return_value = torch.tensor([[320.0, 240.0, 100.0, 200.0]])
        mock_boxes.xyxy.cpu.return_value = torch.tensor([[270.0, 140.0, 370.0, 340.0]])
        mock_boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]
        mock_boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [0]

        mock_dets = MagicMock()
        mock_dets.boxes = mock_boxes
        mock_dets.plot.return_value = frame.copy()

        person_analyzer = MagicMock()
        person_analyzer.analyze_person.return_value = {
            "last_frame": 0, "clothes": [], "emotion": None,
            "gender": None, "age": None, "lidar_data": None,
        }
        person_analyzer.build_add_info.return_value = []
        person_analyzer.person_cache = {}

        result = self.pipeline.process_frame(
            frame_bgr=frame, dets=mock_dets, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=person_analyzer, model_manager=MagicMock(),
            show_window=True, verbose_window=True,
        )
        detections, frame_vis = result
        self.assertEqual(detections["countOfPeople"], 1)
        self.assertEqual(len(detections["objects"]), 1)
        obj = detections["objects"][0]
        self.assertEqual(obj["type"], "person")
        self.assertTrue(obj["isPerson"])
