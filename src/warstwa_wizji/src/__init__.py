"""
Pakiet src - główne moduły warstwy wizyjnej.

Zawiera:
- cv_agent - główny agent wizyjny (CVAgent)
- cv_utils - narzędzia do przetwarzania obrazów i wizji komputerowej
- img_classifiers - klasyfikatory obrazów (emocje, płeć, wiek, ubrania)
- model_trainer - narzędzia do trenowania modeli YOLO

Hierarchia wywołań:
    warstwa_wizji/main.py -> warstwa_wizji/src/cv_agent.py -> CVAgent
"""

from .cv_utils import *
from .cv_agent import CVAgent
