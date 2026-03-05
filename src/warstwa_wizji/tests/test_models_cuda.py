"""
Testy dostępności i uruchamialności modeli YOLO oraz HuggingFace.

Wykonywane TYLKO gdy CUDA jest dostępna (skipowane na maszynach bez GPU).
Modele YOLO: yolo12n.pt, clothes.pt, weapon.pt
Klasyfikatory HuggingFace: emocje, płeć, wiek, typ ubrań, wzór ubrań
"""

import os
import unittest

import torch
import numpy as np

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA nie jest dostępna — pomijam testy modeli GPU"

# Ścieżki do wag YOLO (relatywne do katalogu tego pliku)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "models"))


def _model_path(name: str) -> str:
    return os.path.join(_MODELS_DIR, name)


# =========================================================================
# YOLO
# =========================================================================
@unittest.skipUnless(CUDA_AVAILABLE, SKIP_REASON)
class TestYoloModelsAvailability(unittest.TestCase):
    """Testy ładowania i uruchamiania modeli YOLO na GPU."""

    def test_ultralytics_importable(self):
        from ultralytics import YOLO  # noqa: F401

    def test_main_detector_loads(self):
        """Główny detektor (yolo12n.pt) ładuje się poprawnie."""
        from ultralytics import YOLO
        path = _model_path("yolo12n.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        self.assertIsNotNone(model)

    def test_main_detector_inference(self):
        """Główny detektor produkuje wynik na syntetycznym obrazie."""
        from ultralytics import YOLO
        path = _model_path("yolo12n.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict(source=dummy, verbose=False)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

    def test_main_detector_has_class_names(self):
        """Główny detektor posiada mapowanie nazw klas."""
        from ultralytics import YOLO
        path = _model_path("yolo12n.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        self.assertIsInstance(model.names, dict)
        self.assertGreater(len(model.names), 0)
        # klasa 0 to zazwyczaj 'person'
        self.assertIn(0, model.names)

    def test_clothes_detector_loads(self):
        """Detektor ubrań (clothes.pt) ładuje się poprawnie."""
        from ultralytics import YOLO
        path = _model_path("clothes.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        self.assertIsNotNone(model)

    def test_clothes_detector_inference(self):
        """Detektor ubrań produkuje wynik na syntetycznym obrazie."""
        from ultralytics import YOLO
        path = _model_path("clothes.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        dummy = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        results = model.predict(source=dummy, verbose=False)
        self.assertIsNotNone(results)

    def test_weapon_detector_loads(self):
        """Detektor broni (weapon.pt) ładuje się poprawnie."""
        from ultralytics import YOLO
        path = _model_path("weapon.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        self.assertIsNotNone(model)

    def test_weapon_detector_inference(self):
        """Detektor broni produkuje wynik na syntetycznym obrazie."""
        from ultralytics import YOLO
        path = _model_path("weapon.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        model = YOLO(path)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict(source=dummy, verbose=False)
        self.assertIsNotNone(results)

    def test_model_manager_init(self):
        """ModelManager inicjalizuje się poprawnie z domyślnym modelem."""
        path = _model_path("yolo12n.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        from warstwa_wizji import ModelManager
        mm = ModelManager(weights_path=path, export_to_engine=False)
        self.assertIsNotNone(mm.detector)
        self.assertIsNotNone(mm.clothes_detector)
        self.assertIsNotNone(mm.class_names)

    def test_model_manager_detect_objects(self):
        """ModelManager.detect_objects() zwraca wynik z śledzeniem."""
        path = _model_path("yolo12n.pt")
        if not os.path.isfile(path):
            self.skipTest(f"Brak pliku wag: {path}")
        from warstwa_wizji import ModelManager
        mm = ModelManager(weights_path=path, export_to_engine=False)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = mm.detect_objects(dummy, imgsz=640, run_detection=True)
        self.assertIsNotNone(result)
        # Wynik powinien mieć atrybut boxes
        self.assertTrue(hasattr(result, "boxes"))


# =========================================================================
# KLASYFIKATORY HUGGINGFACE
# =========================================================================
@unittest.skipUnless(CUDA_AVAILABLE, SKIP_REASON)
class TestHuggingFaceClassifiers(unittest.TestCase):
    """Testy ładowania i uruchamiania klasyfikatorów HuggingFace na GPU."""

    def test_transformers_importable(self):
        from transformers import AutoImageProcessor, SiglipForImageClassification  # noqa: F401
        from transformers import ViTImageProcessor, ViTForImageClassification  # noqa: F401

    def test_emotion_classifier_loads_and_runs(self):
        """Klasyfikator emocji ładuje model i zwraca predykcję."""
        from warstwa_wizji.src.img_classifiers.image_classifier import EmotionClassifier
        from PIL import Image
        clf = EmotionClassifier()
        clf.load_models()
        self.assertIsNotNone(clf.model)
        self.assertIsNotNone(clf.processor)
        # Inferencja na syntetycznym obrazie
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = clf.process(dummy_img)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        # Sprawdź, że prawdopodobieństwa sumują się do ~1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_gender_classifier_loads_and_runs(self):
        """Klasyfikator płci ładuje model i zwraca predykcję."""
        from warstwa_wizji.src.img_classifiers.image_classifier import GenderClassifier
        from PIL import Image
        clf = GenderClassifier()
        clf.load_models()
        self.assertIsNotNone(clf.model)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = clf.process(dummy_img)
        self.assertIsInstance(result, dict)
        self.assertIn("female", result)
        self.assertIn("male", result)

    def test_age_classifier_loads_and_runs(self):
        """Klasyfikator wieku ładuje model i zwraca predykcję."""
        from warstwa_wizji.src.img_classifiers.image_classifier import AgeClassifier
        from PIL import Image
        clf = AgeClassifier()
        clf.load_models()
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = clf.process(dummy_img)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_clothes_classifier_loads_and_runs(self):
        """Klasyfikator typu ubrań ładuje model i zwraca predykcję."""
        from warstwa_wizji.src.img_classifiers.image_classifier import ClothesClassifier
        from PIL import Image
        clf = ClothesClassifier()
        clf.load_models()
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = clf.process(dummy_img)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_clothes_pattern_classifier_loads_and_runs(self):
        """Klasyfikator wzoru ubrań ładuje model i zwraca predykcję."""
        from warstwa_wizji.src.img_classifiers.image_classifier import ClothesPatternClassifier
        from PIL import Image
        clf = ClothesPatternClassifier()
        clf.load_models()
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = clf.process(dummy_img)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_get_classifiers_factory(self):
        """Funkcja getClassifiers() zwraca krotkę trzech załadowanych klasyfikatorów."""
        from warstwa_wizji.src.img_classifiers.image_classifier import getClassifiers
        emotion, gender, age = getClassifiers()
        self.assertIsNotNone(emotion.model)
        self.assertIsNotNone(gender.model)
        self.assertIsNotNone(age.model)

    def test_get_clothes_classifiers_factory(self):
        """Funkcja getClothesClassifiers() zwraca krotkę trzech obiektów."""
        from warstwa_wizji.src.img_classifiers.image_classifier import getClothesClassifiers
        clothes_clf, pattern_clf, color_fn = getClothesClassifiers()
        self.assertIsNotNone(clothes_clf.model)
        self.assertIsNotNone(pattern_clf.model)
        self.assertTrue(callable(color_fn))

    def test_color_classifier_on_gpu(self):
        """Funkcja findDominantColor uruchamia się poprawnie na GPU."""
        from warstwa_wizji.src.img_classifiers.color_classifier import findDominantColor
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        color = findDominantColor(img)
        self.assertEqual(len(color), 3)
        for c in color:
            self.assertGreaterEqual(c, 0)
            self.assertLessEqual(c, 255)
