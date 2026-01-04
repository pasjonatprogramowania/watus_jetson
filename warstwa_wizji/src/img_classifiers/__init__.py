"""
Pakiet img_classifiers - klasyfikatory obrazów.

Zawiera moduły do:
- Klasyfikacji kolorów (color_classifier)
- Klasyfikacji osób - emocje, wiek, płeć (image_classifier)
- Pomocnicze narzędzia (utils)
- Klasyfikacji obiektów wojskowych (mil_object_classifier - eksperymentalny)

Hierarchia wywołań:
    warstwa_wizji/main.py -> warstwa_wizji/src/img_classifiers/*
"""

from .color_classifier import findDominantColor
from .image_classifier import getClassifiers, getClothesClassifiers