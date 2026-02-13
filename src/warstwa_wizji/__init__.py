"""
Pakiet warstwa_wizji - główny pakiet warstwy wizyjnej.

Zawiera:
- CVAgent - główny agent wizyjny do detekcji i śledzenia obiektów
- Moduły cv_utils - narzędzia do przetwarzania obrazów
- Moduły img_classifiers - klasyfikatory obrazów

Hierarchia wywołań:
    warstwa_wizji/__init__.py -> warstwa_wizji/src/cv_agent.py -> CVAgent
"""

from .src import *
from .main import *