"""
Testy integracyjne przepływu danych między klasami.

Symulacja iteracji pętli CVAgent.run():
    ModelManager.detect_objects → DetectionPipeline.process_frame
    → frame_bgr_vis (annotowany) + detections (dict)
"""

import json
import unittest
from unittest.mock import MagicMock

import numpy as np


class TestDataFlowIntegration(unittest.TestCase):

    def test_full_iteration_no_detections(self):
        from warstwa_wizji import DetectionPipeline
        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        detections, vis = pipeline.process_frame(
            frame_bgr=frame, dets=None, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=MagicMock(), model_manager=MagicMock(),
            show_window=False, verbose_window=False,
        )
        self.assertEqual(detections["objects"], [])
        self.assertEqual(detections["countOfPeople"], 0)
        self.assertIsInstance(detections["brightness"], float)
        np.testing.assert_array_equal(vis, frame)

    def test_full_iteration_with_detections(self):
        import torch
        from warstwa_wizji import DetectionPipeline

        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        annotated_frame = frame.copy()
        annotated_frame[0, 0] = [255, 0, 0]

        mock_boxes = MagicMock()
        mock_boxes.is_track = True
        mock_boxes.xywh.cpu.return_value = torch.tensor([[320.0, 240.0, 100.0, 200.0]])
        mock_boxes.xyxy.cpu.return_value = torch.tensor([[270.0, 140.0, 370.0, 340.0]])
        mock_boxes.id.int.return_value.cpu.return_value.tolist.return_value = [42]
        mock_boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [0]
        mock_dets = MagicMock()
        mock_dets.boxes = mock_boxes
        mock_dets.plot.return_value = annotated_frame

        person_analyzer = MagicMock()
        person_analyzer.analyze_person.return_value = {
            "clothes": [], "emotion": None,
            "gender": None, "age": None, "lidar_data": None,
            "last_clothes_frame": 0, "last_emotion_frame": 0,
        }
        person_analyzer.build_add_info.return_value = []
        person_analyzer.person_cache = {}

        detections, vis = pipeline.process_frame(
            frame_bgr=frame, dets=mock_dets, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=person_analyzer, model_manager=MagicMock(),
            show_window=True, verbose_window=False,
        )
        self.assertFalse(np.array_equal(vis, frame))
        self.assertEqual(detections["objects"][0]["id"], 42)
        self.assertTrue(detections["objects"][0]["isPerson"])

    def test_detections_dict_keys(self):
        from warstwa_wizji import DetectionPipeline
        pipeline = DetectionPipeline()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections, _ = pipeline.process_frame(
            frame_bgr=frame, dets=None, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=MagicMock(), model_manager=MagicMock(),
            show_window=False, verbose_window=False,
        )
        required_keys = {"objects", "countOfPeople", "countOfObjects", "suggested_mode", "brightness"}
        self.assertEqual(set(detections.keys()), required_keys)

    def test_json_save_func_receives_detections(self):
        import torch
        from warstwa_wizji import DetectionPipeline

        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        mock_boxes = MagicMock()
        mock_boxes.is_track = True
        mock_boxes.xywh.cpu.return_value = torch.tensor([[320.0, 240.0, 50.0, 80.0]])
        mock_boxes.xyxy.cpu.return_value = torch.tensor([[295.0, 200.0, 345.0, 280.0]])
        mock_boxes.id.int.return_value.cpu.return_value.tolist.return_value = [10]
        mock_boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [1]
        mock_dets = MagicMock()
        mock_dets.boxes = mock_boxes
        mock_dets.plot.return_value = frame.copy()

        person_analyzer = MagicMock()
        person_analyzer.person_cache = {}

        detections, _ = pipeline.process_frame(
            frame_bgr=frame, dets=mock_dets, frame_idx=5, imgsz=640,
            fov_deg=102, class_names={0: "person", 1: "car"},
            person_analyzer=person_analyzer, model_manager=MagicMock(),
            show_window=False, verbose_window=False,
        )
        self.assertFalse(detections["objects"][0]["isPerson"])
        self.assertEqual(detections["objects"][0]["type"], "car")
        json_str = json.dumps(detections)
        self.assertIsInstance(json.loads(json_str), dict)
