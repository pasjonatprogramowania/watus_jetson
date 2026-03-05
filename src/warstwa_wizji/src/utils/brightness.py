"""
Moduł analizy jasności obrazu.

Zawiera funkcje do obliczania średniej jasności klatki wideo
oraz sugerowania trybu wyświetlania (jasny/ciemny) na podstawie warunków oświetleniowych.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() -> calc_brightness()
    warstwa_wizji/main.py -> CVAgent.run() -> suggest_mode()
"""

import cv2
import numpy as np


# Stałe konfiguracyjne
DARK_MODE_THRESHOLD = 0.33   # Próg przełączenia na tryb ciemny
LIGHT_MODE_THRESHOLD = 0.66  # Próg przełączenia na tryb jasny
TARGET_ANALYSIS_SIZE = 160   # Docelowa szerokość obrazu do analizy (optymalizacja)


def calc_brightness(bgr_image: np.ndarray) -> float:
    """
    Oblicza znormalizowaną średnią jasność obrazu.
    
    Funkcja zmniejsza obraz dla optymalizacji, konwertuje do skali szarości
    i oblicza średnią wartość pikseli znormalizowaną do zakresu [0.0, 1.0].
    
    Argumenty:
        bgr_image (np.ndarray): Obraz wejściowy w formacie BGR (OpenCV).
        
    Zwraca:
        float: Znormalizowana jasność w zakresie [0.0, 1.0].
               0.0 = całkowicie ciemny, 1.0 = całkowicie jasny.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> calc_brightness()
    """
    height, width = bgr_image.shape[:2]
    
    # Oblicz współczynnik skalowania dla optymalizacji
    scale_factor = max(height, width) / TARGET_ANALYSIS_SIZE
    
    # Zmniejsz obraz jeśli jest większy niż docelowy rozmiar
    if scale_factor > 1.0:
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        resized_frame = cv2.resize(
            bgr_image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
    else:
        resized_frame = bgr_image
    
    # Konwertuj do skali szarości i oblicz średnią jasność
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_frame) / 255.0)
    
    return mean_brightness


def suggest_mode(current_brightness: float, current_mode: str) -> str:
    """
    Sugeruje tryb wyświetlania (jasny/ciemny) na podstawie jasności sceny.
    
    Funkcja implementuje histerezę - zmiana trybu następuje tylko gdy jasność
    przekroczy odpowiedni próg, co zapobiega ciągłemu przełączaniu.
    
    Argumenty:
        current_brightness (float): Znormalizowana jasność sceny [0.0, 1.0].
        current_mode (str): Aktualny tryb wyświetlania ('light' lub 'dark').
        
    Zwraca:
        str: Sugerowany tryb wyświetlania ('light' lub 'dark').
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> suggest_mode()
    """
    if current_brightness <= DARK_MODE_THRESHOLD and current_mode == 'light':
        return 'dark'
    elif current_brightness > LIGHT_MODE_THRESHOLD and current_mode == 'dark':
        return 'light'
    else:
        return current_mode
