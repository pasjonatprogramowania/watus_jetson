"""
Moduł integracji danych z czujnika LiDAR.

Odpowiada za odczyt danych śledzenia obiektów z pliku lidar.json
oraz dopasowanie obiektów wykrytych przez kamerę do trackerów LiDAR
na podstawie kąta azymutu. Umożliwia fuzję danych 2D (kamera) z 3D (LiDAR).

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() 
        -> read_lidar_tracks()
        -> match_camera_to_lidar()
"""

import os
import json
import math
from typing import Optional


def read_lidar_tracks(lidar_file_path: str) -> list:
    """
    Odczytuje ostatnią linię z pliku danych LiDAR i zwraca listę śledzonych obiektów.
    
    Plik lidar.json zawiera dane w formacie JSONL (JSON Lines) - każda linia
    to osobny rekord JSON. Funkcja odczytuje najnowsze dane (ostatnia linia).
    
    Argumenty:
        lidar_file_path (str): Ścieżka do pliku lidar.json (format JSONL).
        
    Zwraca:
        list: Lista słowników z danymi śledzonych obiektów. Każdy słownik zawiera:
              - "id": Identyfikator obiektu
              - "last_position": [x, y] - pozycja w metrach
              Pusta lista w przypadku błędu lub braku danych.
              
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_lidar_matching() 
            -> read_lidar_tracks()
    """
    try:
        if not os.path.exists(lidar_file_path):
            return []
        
        with open(lidar_file_path, 'r') as file:
            all_lines = file.readlines()
            if not all_lines:
                return []
            
            last_line = all_lines[-1].strip()
            if not last_line:
                return []
            
            json_data = json.loads(last_line)
            tracked_objects = json_data.get("tracks", [])
            return tracked_objects
            
    except Exception as error:
        print(f"Błąd odczytu danych LiDAR: {error}")
        return []


def compute_lidar_angle(position_x: float, position_y: float) -> float:
    """
    Oblicza kąt azymutu obiektu w stopniach na podstawie jego pozycji XY.
    
    Kąt jest mierzony względem osi Y (kierunek do przodu) i wyrażony w stopniach.
    Wartość dodatnia oznacza obiekt po prawej stronie, ujemna - po lewej.
    
    Argumenty:
        position_x (float): Pozycja X obiektu w metrach (oś boczna).
                           Dodatnia = prawa strona.
        position_y (float): Pozycja Y obiektu w metrach (oś do przodu).
        
    Zwraca:
        float: Kąt azymutu w stopniach w zakresie (-180, 180].
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_lidar_matching()
            -> match_camera_to_lidar() -> compute_lidar_angle()
    """
    return math.degrees(math.atan2(position_x, position_y))


def match_camera_to_lidar(
    person_angles_from_camera: list[tuple[int, float]], 
    lidar_tracked_objects: list,
    max_angle_difference_deg: float = 20.0
) -> dict:
    """
    Dopasowuje obiekty wykryte przez kamerę do trackerów LiDAR na podstawie kąta.
    
    Wykorzystuje algorytm zachłanny - dla każdej osoby z kamery szuka najbliższego
    (pod względem kąta azymutu) trackera LiDAR, który nie został jeszcze przypisany.
    Pozwala to przypisać precyzyjną odległość z LiDAR do obiektu z kamery.
    
    Argumenty:
        person_angles_from_camera (list): Lista krotek (indeks_osoby, kat_kamery_stopnie).
                                          Kąty są obliczone z pozycji w obrazie.
        lidar_tracked_objects (list): Lista słowników z danymi LiDAR.
                                      Każdy musi zawierać "id" i "last_position".
        max_angle_difference_deg (float): Maksymalna akceptowalna różnica 
                                          kątów do uznania dopasowania.
        
    Zwraca:
        dict: Mapa dopasowań {indeks_osoby_z_kamery -> slownik_trackera_lidar}.
              Zawiera tylko pomyślnie dopasowane obiekty.
              
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_lidar_matching()
            -> match_camera_to_lidar()
    """
    matches = {}
    used_lidar_indices = set()
    
    for person_idx, person_angle in person_angles_from_camera:
        best_lidar_idx = -1
        min_angle_diff = 1000.0
        
        for lidar_idx, lidar_object in enumerate(lidar_tracked_objects):
            if lidar_idx in used_lidar_indices:
                continue
                
            lidar_position = lidar_object.get("last_position", [0, 0])
            lidar_angle = compute_lidar_angle(lidar_position[0], lidar_position[1])
            
            angle_diff = abs(person_angle - lidar_angle)
            
            if angle_diff < min_angle_diff and angle_diff < max_angle_difference_deg:
                min_angle_diff = angle_diff
                best_lidar_idx = lidar_idx
        
        if best_lidar_idx != -1:
            used_lidar_indices.add(best_lidar_idx)
            matches[person_idx] = lidar_tracked_objects[best_lidar_idx]
    
    return matches
