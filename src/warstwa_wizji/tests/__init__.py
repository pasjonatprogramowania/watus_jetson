"""
Pakiet testów warstwa_wizji — test suite loader.

Uruchomienie wszystkich testów:
    python -m unittest discover -s src/warstwa_wizji/tests -v

Uruchomienie przez test suite:
    python -m unittest warstwa_wizji.tests -v

Uruchomienie jednego modułu:
    python -m unittest warstwa_wizji.tests.test_angle -v
"""

import unittest


def load_tests(loader, tests, pattern):
    """Ładuje wszystkie moduły testowe jako jeden TestSuite."""
    suite = unittest.TestSuite()

    # --- Testy podstawowe (zawsze uruchamiane) ---
    from . import test_imports
    from . import test_angle
    from . import test_brightness
    from . import test_lidar
    from . import test_detection_processors
    from . import test_person_analyzer
    from . import test_frame_overlay
    from . import test_detection_pipeline
    from . import test_video_io
    from . import test_cv_agent
    from . import test_data_flow
    from . import test_parallel

    core_modules = [
        test_imports,
        test_angle,
        test_brightness,
        test_lidar,
        test_detection_processors,
        test_person_analyzer,
        test_frame_overlay,
        test_detection_pipeline,
        test_video_io,
        test_cv_agent,
        test_data_flow,
        test_parallel,
    ]

    for module in core_modules:
        suite.addTests(loader.loadTestsFromModule(module))

    # --- Testy warunkowe (CUDA / Linux) ---
    from . import test_models_cuda
    from . import test_lidar_hw

    suite.addTests(loader.loadTestsFromModule(test_models_cuda))
    suite.addTests(loader.loadTestsFromModule(test_lidar_hw))

    return suite
