import numpy as np
from typing import List, Dict, Optional

import torch
from ultralytics import YOLO


class CVWrapper:
    """
    Wrapper na model detekcji obiektów (YOLO/RT-DETR) z biblioteki Ultralytics.
    
    Zapewnia interfejs do inicjalizacji modelu i wykonywania detekcji/śledzenia na klatkach BGR.
    
    Atrybuty:
        device (str): Urządzenie obliczeniowe ('cuda' lub 'cpu').
        model (YOLO): Załadowany model Ultralytics.
        score_thresh (float): Próg pewności detekcji (confidence threshold).
        imgsz (int): Rozmiar wejściowy obrazu dla modelu.
        class_names (dict): Słownik mapujący ID klasy na nazwę.
        
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
        Inicjalizuje wrapper CV.
        
        weights: ścieżka lub nazwa wagi Ultralytics (np. 'rtdetr-l.pt', 'rtdetr-x.pt', 'rtdetr-r18.pt')
        score_thresh: minimalny confidence
        imgsz: rozmiar przeskalowania wejścia (kwadrat)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.score_thresh = score_thresh
        self.imgsz = imgsz
        # nazwy klas COCO
        self.class_names = self.model.names  # dict {idx: name}

    def detect(self, frame_bgr):
        """
        Wykonuje detekcję na klatce (alias dla __call__).
        
        Argumenty:
            frame_bgr (np.ndarray): Klatka obrazu w formacie BGR.
            
        Zwraca:
            List[Dict]: Lista wykrytych obiektów.
            
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent.process_camera() -> detect() -> __call__()
        """
        return self.__call__(frame_bgr)

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Wykonuje detekcję i śledzenie (track) na zadanej klatce.
        
        Używa modelu YOLO/RT-DETR w trybie `track(persist=True)`.
        
        Argumenty:
            frame_bgr: numpy.ndarray (H,W,3) w BGR (OpenCV). Ultralytics sam ogarnia konwersję.
            
        Zwraca:
            listę słowników {bbox:[x1,y1,x2,y2], score:float, label:int}
            
        Hierarchia wywołań:
            cv_wrapper.py -> CVWrapper.detect() -> __call__()
        """
        # Stream-off + pojedyncza klatka => results to lista długości 1
        # results = self.model.predict(
        #     source=frame_bgr,
        #     imgsz=self.imgsz,
        #     conf=self.score_thresh,
        #     device=self.device,
        #     verbose=False
        # )
        results = self.model.track(source=frame_bgr, persist=True, device=self.device, verbose=False,
                                   imgsz=self.imgsz)
        r = results[0]
        dets: List[Dict] = []
        if r.boxes is None or r.boxes.shape[0] == 0:
            return dets

        xyxy = r.boxes.xyxy.detach().cpu().numpy()       # (N,4)
        conf = r.boxes.conf.detach().cpu().numpy()       # (N,)
        cls  = r.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)
        for b, s, l in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = b.tolist()
            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(s),
                "label": int(l)
            })
        return dets