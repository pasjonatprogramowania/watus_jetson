"""
Kompletny zestaw testów jednostkowych pakietu warstwa_wizji.

Testy mockują GPU i modele YOLO/HuggingFace, dzięki czemu można je
uruchamiać na dowolnej maszynie (bez CUDA / bez plików .pt).

Uruchomienie:
    python -m pytest src/warstwa_wizji/tests/test_warstwa_wizji.py -v
    python -m unittest discover -s src/warstwa_wizji/tests -v
"""

import json
import math
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. TESTY IMPORTÓW
# ---------------------------------------------------------------------------
class TestImports(unittest.TestCase):
    """Weryfikacja, że wszystkie publiczne symbole dają się zaimportować."""

    def test_import_top_level_package(self):
        import warstwa_wizji  # noqa: F401

    def test_import_cv_agent(self):
        from warstwa_wizji import CVAgent  # noqa: F401

    def test_import_model_manager(self):
        from warstwa_wizji import ModelManager  # noqa: F401

    def test_import_video_io(self):
        from warstwa_wizji import VideoIO  # noqa: F401

    def test_import_person_analyzer(self):
        from warstwa_wizji import PersonAnalyzer  # noqa: F401

    def test_import_detection_pipeline(self):
        from warstwa_wizji import DetectionPipeline  # noqa: F401

    def test_import_calc_obj_angle(self):
        from warstwa_wizji import calc_obj_angle  # noqa: F401

    def test_import_brightness_functions(self):
        from warstwa_wizji import calc_brightness, suggest_mode  # noqa: F401

    def test_import_lidar_functions(self):
        from warstwa_wizji import (  # noqa: F401
            read_lidar_tracks,
            compute_lidar_angle,
            match_camera_to_lidar,
        )

    def test_import_overlay_functions(self):
        from warstwa_wizji import (  # noqa: F401
            draw_clothes_boxes,
            draw_lidar_info,
            draw_stats_overlay,
            draw_weapon_boxes,
            draw_track_history,
        )

    def test_import_detection_processors(self):
        from warstwa_wizji import (  # noqa: F401
            process_clothes_detection,
            process_weapon_detection,
            update_person_cache,
            should_update_cache,
        )

    def test_import_parallel_process(self):
        from warstwa_wizji.src.cv_utils.parallel import (  # noqa: F401
            ParallelProcess,
            Process,
        )


# ---------------------------------------------------------------------------
# 2. TESTY calc_obj_angle
# ---------------------------------------------------------------------------
class TestCalcObjAngle(unittest.TestCase):
    """Testy obliczeń kąta obiektu względem osi kamery."""

    def setUp(self):
        from warstwa_wizji import calc_obj_angle
        self.calc = calc_obj_angle
        self.IMAGE_W = 640

    def test_center_object_angle_near_zero(self):
        angle = self.calc((300, 100), (340, 200), self.IMAGE_W, fov_deg=102)
        self.assertAlmostEqual(angle, 0.0, delta=2.0)

    def test_right_object_positive_angle(self):
        angle = self.calc((500, 100), (600, 200), self.IMAGE_W, fov_deg=102)
        self.assertGreater(angle, 0)

    def test_left_object_negative_angle(self):
        angle = self.calc((40, 100), (140, 200), self.IMAGE_W, fov_deg=102)
        self.assertLess(angle, 0)

    def test_symmetry(self):
        offset = 200
        center = self.IMAGE_W / 2
        box_w = 50
        right = self.calc(
            (center + offset, 100), (center + offset + box_w, 200),
            self.IMAGE_W, fov_deg=102,
        )
        left = self.calc(
            (center - offset - box_w, 100), (center - offset, 200),
            self.IMAGE_W, fov_deg=102,
        )
        self.assertAlmostEqual(right, -left, delta=0.5)

    def test_different_fov(self):
        narrow = self.calc((500, 0), (550, 50), self.IMAGE_W, fov_deg=60)
        wide = self.calc((500, 0), (550, 50), self.IMAGE_W, fov_deg=120)
        # Szerszy FOV → mniejszy kąt dla tego samego piksela
        self.assertLess(abs(narrow), abs(wide))

    def test_edge_object_returns_finite(self):
        angle = self.calc((0, 0), (10, 10), self.IMAGE_W, fov_deg=102)
        self.assertTrue(math.isfinite(angle))


# ---------------------------------------------------------------------------
# 3. TESTY JASNOŚCI I TRYBU
# ---------------------------------------------------------------------------
class TestBrightness(unittest.TestCase):
    """Testy obliczania jasności i sugestii trybu wyświetlania."""

    def setUp(self):
        from warstwa_wizji import calc_brightness, suggest_mode
        self.calc_brightness = calc_brightness
        self.suggest_mode = suggest_mode

    def test_black_image_brightness_near_zero(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertAlmostEqual(self.calc_brightness(black), 0.0, delta=0.01)

    def test_white_image_brightness_near_one(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self.calc_brightness(white), 1.0, delta=0.01)

    def test_gray_image_brightness_about_half(self):
        gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        val = self.calc_brightness(gray)
        self.assertGreater(val, 0.4)
        self.assertLess(val, 0.6)

    def test_brightness_range(self):
        random_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        val = self.calc_brightness(random_img)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_large_image_works(self):
        big = np.full((1920, 1080, 3), 100, dtype=np.uint8)
        val = self.calc_brightness(big)
        self.assertGreater(val, 0.0)

    # --- suggest_mode ---

    def test_suggest_mode_dark_to_light(self):
        mode = self.suggest_mode(0.8, "dark")
        self.assertEqual(mode, "light")

    def test_suggest_mode_light_to_dark(self):
        mode = self.suggest_mode(0.2, "light")
        self.assertEqual(mode, "dark")

    def test_suggest_mode_hysteresis_no_change(self):
        # W strefie między progami — nie zmienia trybu
        mode = self.suggest_mode(0.5, "light")
        self.assertEqual(mode, "light")
        mode = self.suggest_mode(0.5, "dark")
        self.assertEqual(mode, "dark")


# ---------------------------------------------------------------------------
# 4. TESTY INTEGRACJI LiDAR
# ---------------------------------------------------------------------------
class TestLidarIntegration(unittest.TestCase):
    """Testy odczytu danych LiDAR, obliczania kątów i matchowania."""

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
            {"id": "A", "last_position": [1.74, 10.0]},  # ~10°
            {"id": "B", "last_position": [-1.74, 10.0]},  # ~-10°
        ]
        matches = self.match_camera_to_lidar(cam_angles, lidar_objs)
        self.assertIn(0, matches)
        self.assertIn(1, matches)
        self.assertEqual(matches[0]["id"], "A")
        self.assertEqual(matches[1]["id"], "B")

    def test_match_camera_to_lidar_no_match(self):
        cam_angles = [(0, 80.0)]
        lidar_objs = [{"id": "X", "last_position": [0.0, 10.0]}]  # 0°
        matches = self.match_camera_to_lidar(cam_angles, lidar_objs, max_angle_difference_deg=5.0)
        self.assertEqual(len(matches), 0)

    def test_match_camera_to_lidar_empty_inputs(self):
        self.assertEqual(self.match_camera_to_lidar([], []), {})
        self.assertEqual(self.match_camera_to_lidar([(0, 0.0)], []), {})
        self.assertEqual(self.match_camera_to_lidar([], [{"id": 1, "last_position": [0, 1]}]), {})


# ---------------------------------------------------------------------------
# 5. TESTY PROCESORÓW DETEKCJI
# ---------------------------------------------------------------------------
class TestDetectionProcessors(unittest.TestCase):
    """Testy funkcji cache'u i procesorów detekcji."""

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
                "last_frame": 0,
                "clothes": [],
                "emotion": "Happy",
                "gender": "male",
                "age": "Adult 21-44",
                "lidar_data": None,
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


# ---------------------------------------------------------------------------
# 6. TESTY PERSON ANALYZER — build_add_info
# ---------------------------------------------------------------------------
class TestPersonAnalyzerBuildAddInfo(unittest.TestCase):
    """Testy statycznej metody build_add_info klasy PersonAnalyzer."""

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


# ---------------------------------------------------------------------------
# 7. TESTY RYSOWANIA NAKŁADEK (frame_overlay)
# ---------------------------------------------------------------------------
class TestFrameOverlay(unittest.TestCase):
    """Testy funkcji rysujących na obrazie."""

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


# ---------------------------------------------------------------------------
# 8. TESTY DetectionPipeline
# ---------------------------------------------------------------------------
class TestDetectionPipeline(unittest.TestCase):
    """Testy klasy DetectionPipeline."""

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
        self.pipeline.fps_params["t_prev"] = time.time() - 0.033  # ~30 fps
        fps = self.pipeline.calc_fps()
        self.assertGreater(fps, 0)
        self.assertTrue(math.isfinite(fps))

    def test_process_frame_no_dets(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person_analyzer = MagicMock()
        model_manager = MagicMock()
        model_manager.class_names = {0: "person"}

        result = self.pipeline.process_frame(
            frame_bgr=frame,
            dets=None,
            frame_idx=0,
            imgsz=640,
            fov_deg=102,
            class_names=model_manager.class_names,
            person_analyzer=person_analyzer,
            model_manager=model_manager,
            show_window=False,
            verbose_window=False,
        )

        detections, frame_vis = result
        self.assertIsInstance(detections, dict)
        self.assertIn("objects", detections)
        self.assertIn("countOfPeople", detections)
        self.assertIn("countOfObjects", detections)
        self.assertIn("brightness", detections)
        self.assertIn("suggested_mode", detections)
        self.assertEqual(detections["objects"], [])
        self.assertEqual(detections["countOfPeople"], 0)
        np.testing.assert_array_equal(frame_vis, frame)

    def test_process_frame_with_mock_dets(self):
        """Symulacja detekcji z mockiem obiektów YOLO."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Mock rezultatu detekcji (jak ultralytics)
        mock_boxes = MagicMock()
        mock_boxes.is_track = True

        import torch
        mock_boxes.xywh = MagicMock()
        mock_boxes.xywh.cpu.return_value = torch.tensor([[320.0, 240.0, 100.0, 200.0]])

        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu.return_value = torch.tensor([[270.0, 140.0, 370.0, 340.0]])

        mock_boxes.id = MagicMock()
        mock_boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]

        mock_boxes.cls = MagicMock()
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

        model_manager = MagicMock()
        model_manager.class_names = {0: "person"}

        result = self.pipeline.process_frame(
            frame_bgr=frame,
            dets=mock_dets,
            frame_idx=0,
            imgsz=640,
            fov_deg=102,
            class_names={0: "person"},
            person_analyzer=person_analyzer,
            model_manager=model_manager,
            show_window=True,
            verbose_window=True,
        )

        detections, frame_vis = result
        self.assertEqual(detections["countOfPeople"], 1)
        self.assertEqual(detections["countOfObjects"], 1)
        self.assertEqual(len(detections["objects"]), 1)

        obj = detections["objects"][0]
        self.assertEqual(obj["type"], "person")
        self.assertTrue(obj["isPerson"])
        self.assertIn("angle", obj)
        self.assertIn("lidar", obj)


# ---------------------------------------------------------------------------
# 9. TESTY VideoIO (mock)
# ---------------------------------------------------------------------------
class TestVideoIO(unittest.TestCase):
    """Testy klasy VideoIO z mockowanym VideoCapture."""

    @patch("warstwa_wizji.src.video_io.cv2.VideoCapture")
    def test_creation_with_source(self, mock_cap_cls):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        vio = VideoIO(source=0)
        mock_cap_cls.assert_called_once_with(0)
        self.assertIsNotNone(vio.cap)

    @patch("warstwa_wizji.src.video_io.cv2.VideoCapture")
    def test_creation_fails_when_not_opened(self, mock_cap_cls):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        with self.assertRaises(RuntimeError):
            VideoIO(source=0)

    def test_creation_with_existing_cap(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        self.assertIs(vio.cap, mock_cap)

    def test_read_frame_delegates(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        vio = VideoIO(cap=mock_cap)
        ret, f = vio.read_frame()
        self.assertTrue(ret)
        np.testing.assert_array_equal(f, frame)

    def test_release_delegates(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        vio.release()
        mock_cap.release.assert_called_once()

    @patch("warstwa_wizji.src.video_io.cv2.imshow")
    def test_show_frame_calls_imshow(self, mock_imshow):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vio.show_frame(frame)
        mock_imshow.assert_called_once()


# ---------------------------------------------------------------------------
# 10. TESTY CVAgent (inicjalizacja z mockami)
# ---------------------------------------------------------------------------
class TestCVAgentInit(unittest.TestCase):
    """Testy tworzenia CVAgent z mockowanymi komponentami."""

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_creates_all_components(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent

        agent = CVAgent(weights_path="fake.pt", source=0)

        MockMM.assert_called_once()
        MockVIO.assert_called_once()
        MockPA.assert_called_once()
        MockDP.assert_called_once()

        self.assertIsNotNone(agent.models)
        self.assertIsNotNone(agent.video)
        self.assertIsNotNone(agent.person_analyzer)
        self.assertIsNotNone(agent.pipeline)

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_json_save_func_stored(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent

        save_fn = MagicMock()
        agent = CVAgent(weights_path="fake.pt", json_save_func=save_fn)
        self.assertIs(agent.save_to_json, save_fn)

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_export_engine_flag_passed(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent

        agent = CVAgent(weights_path="fake.pt", export_to_engine=True)
        call_kwargs = MockMM.call_args
        self.assertTrue(call_kwargs.kwargs.get("export_to_engine", False)
                        or (len(call_kwargs.args) > 2 and call_kwargs.args[2]))


# ---------------------------------------------------------------------------
# 11. TESTY PRZEPŁYWU DANYCH (integracja)
# ---------------------------------------------------------------------------
class TestDataFlowIntegration(unittest.TestCase):
    """
    Testy integracyjne: symulacja jednej iteracji pętli CVAgent.run().

    Sprawdzają, że dane przepływają poprawnie między komponentami:
        ModelManager.detect_objects → DetectionPipeline.process_frame
        → frame_bgr_vis (annotowany) + detections (dict)
    """

    def test_full_iteration_no_detections(self):
        """Pętla z klatką bez żadnych detekcji — zwraca puste obiekty."""
        from warstwa_wizji import DetectionPipeline

        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        detections, vis = pipeline.process_frame(
            frame_bgr=frame, dets=None, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=MagicMock(), model_manager=MagicMock(),
            show_window=False, verbose_window=False,
        )

        self.assertIsInstance(detections, dict)
        self.assertEqual(detections["objects"], [])
        self.assertEqual(detections["countOfPeople"], 0)
        self.assertEqual(detections["countOfObjects"], 0)
        self.assertIsInstance(detections["brightness"], float)
        self.assertIsInstance(detections["suggested_mode"], str)
        # Bez detekcji, vis == frame
        np.testing.assert_array_equal(vis, frame)

    def test_full_iteration_with_detections(self):
        """Pętla z mockowaną detekcją osoby — sprawdza strukturę wyjściową."""
        import torch
        from warstwa_wizji import DetectionPipeline

        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        annotated_frame = frame.copy()
        annotated_frame[0, 0] = [255, 0, 0]  # celowa zmiana piksela

        # Mock dets
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
            "last_frame": 0, "clothes": [], "emotion": None,
            "gender": None, "age": None, "lidar_data": None,
        }
        person_analyzer.build_add_info.return_value = []
        person_analyzer.person_cache = {}

        detections, vis = pipeline.process_frame(
            frame_bgr=frame, dets=mock_dets, frame_idx=0, imgsz=640,
            fov_deg=102, class_names={0: "person"},
            person_analyzer=person_analyzer, model_manager=MagicMock(),
            show_window=True, verbose_window=False,
        )

        # Sprawdź, że vis to annotowany obraz (nie oryginalny!)
        self.assertFalse(np.array_equal(vis, frame),
                         "frame_bgr_vis powinien być inny niż oryginalny frame")
        np.testing.assert_array_equal(vis[0, 0], [255, 0, 0])

        # Sprawdź strukturę detekcji
        self.assertEqual(len(detections["objects"]), 1)
        obj = detections["objects"][0]
        self.assertEqual(obj["id"], 42)
        self.assertEqual(obj["type"], "person")
        self.assertTrue(obj["isPerson"])
        self.assertIsInstance(obj["angle"], float)

    def test_detections_dict_keys(self):
        """Sprawdza, że słownik detections zawiera all wymagane klucze."""
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
        """Symuluje zapis do JSONL — callback otrzymuje poprawny dict."""
        import torch
        from warstwa_wizji import DetectionPipeline

        pipeline = DetectionPipeline()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        mock_boxes = MagicMock()
        mock_boxes.is_track = True
        mock_boxes.xywh.cpu.return_value = torch.tensor([[320.0, 240.0, 50.0, 80.0]])
        mock_boxes.xyxy.cpu.return_value = torch.tensor([[295.0, 200.0, 345.0, 280.0]])
        mock_boxes.id.int.return_value.cpu.return_value.tolist.return_value = [10]
        mock_boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [1]  # nie-osoba

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

        # Obiekt klasy 1 (car) nie jest osobą
        self.assertEqual(detections["countOfPeople"], 0)
        self.assertEqual(detections["countOfObjects"], 1)
        self.assertFalse(detections["objects"][0]["isPerson"])
        self.assertEqual(detections["objects"][0]["type"], "car")

        # Sprawdź, że detections jest serializowalny do JSON
        json_str = json.dumps(detections)
        self.assertIsInstance(json.loads(json_str), dict)


# ---------------------------------------------------------------------------
# 12. TESTY ParallelProcess
# ---------------------------------------------------------------------------
class TestParallelProcess(unittest.TestCase):
    """Testy klasy ParallelProcess."""

    def test_creation(self):
        from warstwa_wizji.src.cv_utils.parallel import ParallelProcess
        proc = ParallelProcess(process_id=42)
        self.assertEqual(proc.process_id, 42)

    def test_alias(self):
        from warstwa_wizji.src.cv_utils.parallel import ParallelProcess, Process
        self.assertIs(Process, ParallelProcess)

    def test_is_multiprocessing_process(self):
        import multiprocessing
        from warstwa_wizji.src.cv_utils.parallel import ParallelProcess
        proc = ParallelProcess(process_id=1)
        self.assertIsInstance(proc, multiprocessing.Process)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
