"""
Moduł zarządzania modelami detekcji obiektów.

Odpowiada za ładowanie modeli YOLO (detektor główny, ubrania, broń),
eksport do TensorRT (.engine) oraz rozgrzewanie GPU.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.__init__() -> ModelManager()
"""

import os
import torch
from torch.amp import autocast
from ultralytics import YOLO


class ModelManager:
    """
    Zarządza modelami YOLO do detekcji obiektów, ubrań i broni.

    Atrybuty:
        device (torch.device): Urządzenie obliczeniowe (cuda/cpu).
        detector (YOLO): Główny model detekcji obiektów.
        clothes_detector (YOLO): Model detekcji ubrań.
        guns_detector (YOLO): Model detekcji broni.
        class_names (dict): Mapowanie ID klasy -> nazwa dla głównego detektora.
        imgsz (int): Rozmiar wejściowy obrazu dla modelu.
    """

    def __init__(
        self,
        weights_path: str = "yolo12s.pt",
        imgsz: int = 640,
        export_to_engine: bool = False,
    ):
        self.imgsz = imgsz

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Główny detektor
        weights_path = self._resolve_engine_model(weights_path, export_to_engine)
        self.detector = YOLO(weights_path)
        self.class_names = self.detector.names

        # Detektor ubrań
        clothes_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../models/clothes.pt"
        )
        clothes_path = self._resolve_engine_model(clothes_path, export_to_engine)
        self.clothes_detector = YOLO(clothes_path)

        # Detektor broni
        guns_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../models/weapon.pt"
        )
        guns_path = self._resolve_engine_model(guns_path, export_to_engine)
        self.guns_detector = YOLO(guns_path)

    # ------------------------------------------------------------------

    def _resolve_engine_model(self, pt_path: str, export_to_engine: bool) -> str:
        """Konwertuje model .pt do .engine (TensorRT) jeśli wymagane."""
        if not export_to_engine:
            return pt_path

        engine_path = os.path.splitext(pt_path)[0] + ".engine"
        if os.path.isfile(engine_path):
            print(f"[ModelManager] Znaleziono istniejący plik .engine: {engine_path}")
            return engine_path

        print(f"[ModelManager] Eksportowanie {pt_path} -> {engine_path} (TensorRT) ...")
        temp_model = YOLO(pt_path)
        exported_path = temp_model.export(format="engine", imgsz=self.imgsz)
        print(f"[ModelManager] Eksport zakończony: {exported_path}")
        return str(exported_path)

    # ------------------------------------------------------------------

    def warm_up(self, frame):
        """Rozgrzewa model na jednej klatce, aby zainicjalizować pamięć GPU."""
        if frame is None:
            return
        with torch.inference_mode():
            if self.device.type == "cuda":
                with autocast(dtype=torch.float32, device_type=self.device.type):
                    _ = self.detector(frame)
            else:
                _ = self.detector(frame)

    # ------------------------------------------------------------------

    def detect_objects(self, frame_bgr, imgsz: int = 640, run_detection: bool = True):
        """
        Uruchamia detekcję i śledzenie obiektów na klatce.

        Zwraca:
            Wynik detekcji (ultralytics Results[0]) z bieżącym śledzeniem.
        """
        if not run_detection:
            return None

        iou = 0.7
        conf = 0.3
        with torch.inference_mode():
            if self.device.type == "cuda":
                with autocast(dtype=torch.float32, device_type=self.device.type):
                    detections = self.detector.track(
                        frame_bgr, persist=True, device=self.device,
                        verbose=False, imgsz=imgsz, iou=iou, conf=conf,
                    )
            else:
                detections = self.detector.track(
                    frame_bgr, persist=True, device=self.device,
                    verbose=False, imgsz=imgsz, iou=iou, conf=conf,
                )
        return detections[0]
