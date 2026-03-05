"""Testy importów — weryfikacja, że wszystkie publiczne symbole dają się zaimportować."""

import unittest


class TestImports(unittest.TestCase):

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
