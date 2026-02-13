"""
Moduł narzędziowy do przetwarzania obrazów.

Zawiera funkcje pomocnicze do pobierania obrazów z URL
oraz inicjalizacji pipeline'ów HuggingFace.

Hierarchia wywołań:
    warstwa_wizji/src/img_classifiers/image_classifier.py -> retrieve_img()
"""

import urllib

from PIL import Image
from transformers import pipeline
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _get_pipe(model_id: str, task: str = "image-classification"):
    """
    Inicjalizuje i cachuje pipeline do klasyfikacji obrazów.
    
    Funkcja używa dekoratora @lru_cache do przechowywania zainicjalizowanego
    pipeline'u, co eliminuje konieczność wielokrotnego ładowania modelu.
    Automatycznie wybiera najlepsze dostępne urządzenie (CUDA > MPS > CPU).
    
    Argumenty:
        model_id (str): ID modelu na HuggingFace Hub.
        task (str): Typ zadania pipeline'u (domyślnie "image-classification").
        
    Zwraca:
        Pipeline: Zainicjalizowany pipeline HuggingFace.
        
    Hierarchia wywołań:
        Używane wewnętrznie przez klasyfikatory obrazów.
    """
    # Wybór urządzenia obliczeniowego: CUDA > MPS (Apple) > CPU
    device = -1  # CPU
    
    if torch.cuda.is_available():
        device = 0  # Pierwsza karta GPU
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = -1  # MPS używa CPU device id w transformers

    return pipeline(
        task="image-classification",
        model=model_id,
        device=device,
    )


def retrieve_img(url: str) -> Image.Image:
    """
    Pobiera obraz z podanego URL i zwraca jako obiekt PIL Image.
    
    Funkcja pobiera obraz do pliku tymczasowego i otwiera go
    za pomocą biblioteki PIL.
    
    Argumenty:
        url (str): URL do obrazu (np. "https://example.com/image.jpg").
        
    Zwraca:
        PIL.Image.Image: Załadowany obraz.
        
    Uwaga:
        Obraz jest zapisywany tymczasowo jako "img.jpg" w bieżącym katalogu.
        
    Hierarchia wywołań:
        warstwa_wizji/src/img_classifiers/image_classifier.py (test)
            -> retrieve_img()
    """
    temp_path = "img.jpg"
    urllib.request.urlretrieve(url, temp_path)
    img = Image.open(temp_path)
    return img