"""
Moduł wizualizacji - rysowanie nakładek informacyjnych na klatkach wideo.

Zawiera funkcje do rysowania bounding boxów obiektów, informacji z LiDAR,
statystyk wydajności (FPS) i innych elementów graficznych nakładanych na obraz.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run()
        -> draw_clothes_boxes()
        -> draw_lidar_info()
        -> draw_stats_overlay()
        -> draw_weapon_boxes()
"""

import cv2
import numpy as np
from typing import Optional


# Domyślne kolory (format BGR dla OpenCV)
CLOTHES_BOX_COLOR = (0, 0, 255)        # Czerwony
LIDAR_TEXT_COLOR = (0, 255, 255)        # Żółty (cyjan)
STATS_BG_COLOR = (0, 0, 0)              # Czarny
STATS_TEXT_COLOR = (255, 255, 255)      # Biały
WEAPON_BOX_COLOR = (0, 0, 0)            # Czarny
TRACK_HISTORY_COLOR = (230, 230, 230)   # Jasnoszary


def draw_clothes_boxes(
    frame_bgr: np.ndarray,
    clothes_data: list,
    person_box_xyxy: tuple[int, int, int, int],
    box_color: tuple[int, int, int] = CLOTHES_BOX_COLOR
) -> None:
    """
    Rysuje bounding boxy wykrytych ubrań na klatce wideo.
    
    Funkcja denormalizuje współrzędne ubrań (zapisane jako 0-1 względem
    wycinka osoby) do współrzędnych globalnych klatki.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu BGR, modyfikowana in-place.
        clothes_data (list): Lista słowników z danymi ubrań.
                             Każdy musi zawierać "box_norm", "label", "conf".
        person_box_xyxy (tuple): Współrzędne osoby (x1, y1, x2, y2) w pikselach.
        box_color (tuple): Kolor ramki w formacie BGR.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> draw_clothes_boxes()
    """
    person_x1, person_y1, person_x2, person_y2 = person_box_xyxy
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1
    
    for item in clothes_data:
        # Pobierz znormalizowane współrzędne
        nx1, ny1, nx2, ny2 = item["box_norm"]
        label_text = f"{item['label']} {item['conf']:.2f}"
        
        # Denormalizuj do współrzędnych globalnych klatki
        global_x1 = int(person_x1 + nx1 * person_width)
        global_y1 = int(person_y1 + ny1 * person_height)
        global_x2 = int(person_x1 + nx2 * person_width)
        global_y2 = int(person_y1 + ny2 * person_height)
        
        # Rysuj ramkę i etykietę
        cv2.rectangle(frame_bgr, (global_x1, global_y1), (global_x2, global_y2), box_color, 2)
        cv2.putText(
            frame_bgr, label_text, (global_x1, global_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1
        )


def draw_lidar_info(
    frame_bgr: np.ndarray,
    lidar_track_id: int,
    distance_meters: float,
    text_position: tuple[int, int],
    text_color: tuple[int, int, int] = LIDAR_TEXT_COLOR
) -> None:
    """
    Rysuje informacje o obiekcie z czujnika LiDAR na klatce.
    
    Wyświetla ID śledzionego obiektu i jego odległość od czujnika.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu BGR, modyfikowana in-place.
        lidar_track_id (int): ID trackera z systemu LiDAR.
        distance_meters (float): Odległość obiektu w metrach.
        text_position (tuple): Pozycja lewego górnego rogu tekstu (x, y).
        text_color (tuple): Kolor tekstu w formacie BGR.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._handle_lidar_info() -> draw_lidar_info()
    """
    lidar_text = f"LIDAR ID: {lidar_track_id} Dystans: {distance_meters:.2f}m"
    cv2.putText(
        frame_bgr, lidar_text, text_position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
    )


def draw_stats_overlay(
    frame_bgr: np.ndarray,
    fps: float,
    brightness: float,
    display_mode: str,
    bg_color: tuple[int, int, int] = STATS_BG_COLOR,
    text_color: tuple[int, int, int] = STATS_TEXT_COLOR
) -> None:
    """
    Rysuje nakładkę ze statystykami wydajności i parametrami sceny.
    
    Wyświetla FPS, jasność obrazu i sugerowany tryb wyświetlania
    w lewym dolnym rogu klatki.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu BGR, modyfikowana in-place.
        fps (float): Aktualna wartość klatek na sekundę.
        brightness (float): Znormalizowana jasność sceny (0-1).
        display_mode (str): Aktualny tryb wyświetlania ('light' lub 'dark').
        bg_color (tuple): Kolor tła nakładki w formacie BGR.
        text_color (tuple): Kolor tekstu w formacie BGR.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> draw_stats_overlay()
    """
    frame_height = frame_bgr.shape[0]
    
    # Rysuj półprzezroczyste tło
    cv2.rectangle(frame_bgr, (0, frame_height), (200, frame_height - 95), bg_color, -1)
    
    # Wyświetl FPS
    cv2.putText(
        frame_bgr, f"FPS: {fps:0,.2f}", (0, frame_height - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
    )
    
    # Wyświetl jasność
    cv2.putText(
        frame_bgr, f"Jasność: {brightness:0,.2f}", (0, frame_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
    )
    
    # Wyświetl tryb
    cv2.putText(
        frame_bgr, f"Tryb: {display_mode}", (0, frame_height),
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
    )


def draw_weapon_boxes(
    frame_bgr: np.ndarray,
    detected_weapons: list,
    box_color: tuple[int, int, int] = WEAPON_BOX_COLOR
) -> None:
    """
    Rysuje bounding boxy wykrytej broni na klatce.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu BGR, modyfikowana in-place.
        detected_weapons (list): Lista słowników z danymi broni.
                                 Każdy musi zawierać "box", "conf", "label".
        box_color (tuple): Kolor ramki w formacie BGR.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._handle_weapon_detection()
            -> draw_weapon_boxes()
    """
    for weapon in detected_weapons:
        x1, y1, x2, y2 = weapon["box"]
        confidence = weapon["conf"]
        label = weapon["label"]
        
        cv2.rectangle(
            frame_bgr, 
            (int(x1), int(y1)), 
            (int(x2), int(y2)), 
            box_color, 
            2
        )
        cv2.putText(
            frame_bgr, 
            f"{label} {confidence:.2f}", 
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            box_color, 
            2
        )


def draw_track_history(
    frame_bgr: np.ndarray,
    position_history: list[tuple[float, float]],
    line_color: tuple[int, int, int] = TRACK_HISTORY_COLOR,
    line_thickness: int = 10
) -> None:
    """
    Rysuje historię ruchu (ścieżkę) śledzonego obiektu.
    
    Wizualizuje trajektorię obiektu jako polilinię łączącą
    kolejne pozycje z ostatnich klatek.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu BGR, modyfikowana in-place.
        position_history (list): Lista krotek (x, y) reprezentujących pozycje.
        line_color (tuple): Kolor linii ścieżki w formacie BGR.
        line_thickness (int): Grubość rysowanej linii w pikselach.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._actualize_tracks()
            -> draw_track_history()
    """
    if len(position_history) < 2:
        return
    
    points = np.hstack(position_history).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame_bgr, [points], isClosed=False, 
                  color=line_color, thickness=line_thickness)
