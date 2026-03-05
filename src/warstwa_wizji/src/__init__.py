"""
Pakiet src - główne moduły warstwy wizyjnej.

Zawiera:
- cv_agent - główny agent wizyjny (CVAgent) — lekki orkiestrator
- model_manager - zarządzanie modelami YOLO (ModelManager)
- video_io - obsługa kamery i nagrywania (VideoIO)
- person_analyzer - analiza atrybutów osób (PersonAnalyzer)
- detection_pipeline - przetwarzanie klatek (DetectionPipeline)
- cv_utils - narzędzia do przetwarzania obrazów i wizji komputerowej
- img_classifiers - klasyfikatory obrazów (emocje, płeć, wiek, ubrania)
- model_trainer - narzędzia do trenowania modeli YOLO

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run()
        -> ModelManager (modele)
        -> VideoIO (kamera)
        -> PersonAnalyzer (osoby)
        -> DetectionPipeline (przetwarzanie klatki)
"""

from .cv_utils import *
from .cv_agent import CVAgent
from .model_manager import ModelManager
from .video_io import VideoIO
from .person_analyzer import PersonAnalyzer
from .detection_pipeline import DetectionPipeline
