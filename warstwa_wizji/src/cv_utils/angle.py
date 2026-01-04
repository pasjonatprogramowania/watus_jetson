"""
Moduł obliczania kąta obiektu względem osi kamery.

Zawiera funkcję do szacowania poziomego kąta między osią optyczną kamery
a środkiem wykrytego obiektu na podstawie jego pozycji w obrazie.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() -> calc_obj_angle()
"""

import math
from typing import Tuple


def calc_obj_angle(
    top_left_point: Tuple[float, float],
    bottom_right_point: Tuple[float, float],
    image_width: int,
    fov_deg: float = 102.0,
) -> float:
    """
    Oblicza poziomy kąt (w stopniach) między osią optyczną kamery
    a środkiem wykrytego obiektu.
    
    Funkcja szacuje kąt na podstawie pozycji obiektu w obrazie,
    wykorzystując parametry kamery (szerokość obrazu i pole widzenia).
    Kąt dodatni oznacza obiekt po prawej stronie, ujemny - po lewej.
    
    Argumenty:
        top_left_point (Tuple[float, float]): Współrzędne (x, y) lewego górnego rogu obiektu.
        bottom_right_point (Tuple[float, float]): Współrzędne (x, y) prawego dolnego rogu obiektu.
        image_width (int): Szerokość obrazu w pikselach.
        fov_deg (float): Poziome pole widzenia kamery (FOV) w stopniach.
        
    Zwraca:
        float: Kąt w stopniach względem osi optycznej kamery.
               Dodatni = obiekt po prawej, ujemny = obiekt po lewej.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> calc_obj_angle()
    """
    (x1, _), (x2, _) = top_left_point, bottom_right_point
    
    # Oblicz środek obiektu w osi X
    object_center_x = 0.5 * (x1 + x2)
    
    # Środek obrazu (oś optyczna kamery)
    image_center_x = image_width / 2.0
    
    # Oblicz ogniskową w pikselach na podstawie pola widzenia
    fov_rad = math.radians(fov_deg)
    focal_length_pixels = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    
    # Oblicz kąt w radianach i zamień na stopnie
    offset_from_center = object_center_x - image_center_x
    angle_rad = math.atan(offset_from_center / focal_length_pixels)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg
