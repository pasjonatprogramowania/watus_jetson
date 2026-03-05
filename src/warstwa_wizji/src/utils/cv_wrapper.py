"""
Moduł wrappera na model detekcji obiektów.

Zapewnia zunifikowany interfejs do modeli detekcji obiektów (YOLO/RT-DETR)
z biblioteki Ultralytics. Umożliwia wykonywanie detekcji i śledzenia obiektów.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.__init__() -> CVWrapper()
"""

import numpy as np
from typing import List, Dict, Optional

import torch
from ultralytics import YOLO


class CVWrapper:
    """
    Wrapper na model detekcji obiektów (YOLO/RT-DETR) z biblioteki Ultralytics.
    
    Zapewnia zunifikowany interfejs do inicjalizacji modelu i wykonywania 
    detekcji oraz śledzenia obiektów na klatkach wideo BGR.
    
    Atrybuty:
        device (str): Urządzenie obliczeniowe ('cuda' lub 'cpu').
        model (YOLO): Załadowany model Ultralytics.
        score_thresh (float): Minimalny próg pewności detekcji (0-1).
        imgsz (int): Rozmiar wejściowy obrazu dla modelu w pikselach.
        class_names (dict): Słownik mapujący ID klasy na jej nazwę tekstową.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.__init__() -> CVWrapper()
    """
    
    def __init__(
        self,
        weights: str = "rtdetr-l.pt",
        score_thresh: float = 0.5,
        device: Optional[str] = None,
        imgsz: int = 640
    ):
        """
        Inicjalizuje wrapper modelu detekcji.
        
        Argumenty:
            weights (str): Ścieżka lub nazwa modelu Ultralytics 
                          (np. 'rtdetr-l.pt', 'yolov8n.pt').
            score_thresh (float): Minimalny próg pewności detekcji (0-1).
            device (str): Urządzenie obliczeniowe ('cuda', 'cpu' lub None dla auto).
            imgsz (int): Rozmiar przeskalowania obrazu wejściowego (kwadrat).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.score_thresh = score_thresh
        self.imgsz = imgsz
        # Słownik nazw klas (np. COCO format: {0: 'person', 1: 'bicycle', ...})
        self.class_names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Wykonuje detekcję obiektów na klatce wideo.
        
        Alias dla metody __call__ - zapewnia czytelniejszy interfejs.
        
        Argumenty:
            frame_bgr (np.ndarray): Klatka obrazu w formacie BGR (OpenCV).
            
        Zwraca:
            List[Dict]: Lista wykrytych obiektów. Każdy słownik zawiera:
                        - "bbox": [x1, y1, x2, y2] - współrzędne ramki
                        - "score": float - pewność detekcji
                        - "label": int - ID klasy obiektu
                        
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent._detect_objects() 
                -> CVWrapper.detect()
        """
        return self.__call__(frame_bgr)

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Wykonuje detekcję i śledzenie obiektów na klatce.
        
        Używa modelu YOLO/RT-DETR w trybie śledzenia (track) z opcją persist=True,
        co zapewnia ciągłość identyfikatorów obiektów między klatkami.
        
        Argumenty:
            frame_bgr (np.ndarray): Klatka obrazu BGR o wymiarach (H, W, 3).
                                    Ultralytics automatycznie konwertuje format.
            
        Zwraca:
            List[Dict]: Lista słowników z wykrytymi obiektami:
                        - "bbox": [x1, y1, x2, y2] - współrzędne float
                        - "score": float - pewność detekcji (0-1)
                        - "label": int - ID klasy obiektu
                        
        Hierarchia wywołań:
            cv_wrapper.py -> CVWrapper.detect() -> __call__()
        """
        # Uruchom model w trybie śledzenia
        results = self.model.track(
            source=frame_bgr, 
            persist=True, 
            device=self.device, 
            verbose=False,
            imgsz=self.imgsz
        )
        
        result = results[0]
        detections: List[Dict] = []
        
        # Sprawdź czy są jakiekolwiek wykrycia
        if result.boxes is None or result.boxes.shape[0] == 0:
            return detections

        # Ekstrahuj dane z wyników
        xyxy = result.boxes.xyxy.detach().cpu().numpy()   # (N, 4)
        conf = result.boxes.conf.detach().cpu().numpy()   # (N,)
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)
        
        # Zbuduj listę słowników z wykryciami
        for box, score, label in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "label": int(label)
            })
            
        return detections