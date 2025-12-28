from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np

from src.config import R_MIN_M, R_MAX_M, LIDAR_ANGLE_OFFSET_DEG

ANGLE_OFFSET_RAD = np.deg2rad(LIDAR_ANGLE_OFFSET_DEG)

class BeamCategory(Enum):
    """
    Enum kategorii wiązek LiDAR po przetwarzaniu wstępnym.
    
    Wartości:
        NONE: Wiązka poza zakresem lub nieprawidłowa (do pominięcia)
        HUMAN: Wiązka należąca do segmentu wykrytego jako człowiek
        OBSTACLE: Wiązka należąca do statycznej przeszkody
    
    Hierarchia wywołań:
        Używane w: preprocess.py, segmentation.py, occupancy_grid.py, system.py
    """
    NONE = "None"
    HUMAN = "Human"
    OBSTACLE = "Obstacle"


@dataclass
class BeamResult:
    """
    Wynik przetwarzania pojedynczej wiązki LiDAR.
    
    Atrybuty:
        theta (float): Kąt wiązki w radianach, względem osi X robota.
        r (float): Odległość do punktu w metrach.
        category (BeamCategory): Kategoria wiązki (NONE/HUMAN/OBSTACLE).
    
    Hierarchia wywołań:
        Tworzony przez: preprocess_scan()
        Używany w: segmentation.py, occupancy_grid.py, system.py, Live_Vis_v3.py
    """
    theta: float            # kąt wiązki [rad]
    r: float                # odległość [m]
    category: BeamCategory  # wynik klasyfikacji


def classify_beam_by_range(r: float) -> BeamCategory:
    """
    Klasyfikuje wiązkę na podstawie odległości.
    
    Wiązki poza zakresem [R_MIN_M, R_MAX_M] są odrzucane jako NONE.
    Pozostałe traktowane jako OBSTACLE (kategoria może być zmieniona
    podczas późniejszej segmentacji i klasyfikacji).
    
    Argumenty:
        r (float): Odległość wiązki w metrach.
    
    Zwraca:
        BeamCategory: NONE jeśli poza zakresem, OBSTACLE w przeciwnym razie.
    
    Hierarchia wywołań:
        preprocess.py -> preprocess_scan() -> classify_beam_by_range()
    """
    # Klasyfikacja po zasięgu (poza zakresem -> NONE)
    if r < R_MIN_M or r > R_MAX_M:
        return BeamCategory.NONE
    return BeamCategory.OBSTACLE


def preprocess_scan(r: np.ndarray, theta: np.ndarray) -> List[BeamResult]:
    """
    Przetwarza surowy skan LiDAR na listę obiektów BeamResult.
    
    Funkcja wykonuje:
      1. Walidację rozmiaru tablic
      2. Korekcję kąta (LIDAR_ANGLE_OFFSET_DEG)
      3. Wstępną klasyfikację wiązek po zasięgu
    
    Argumenty:
        r (np.ndarray): Tablica odległości w metrach.
        theta (np.ndarray): Tablica kątów w radianach.
    
    Zwraca:
        List[BeamResult]: Lista przetworzoncy punktów z kategoriami.
    
    Wyjątki:
        ValueError: Gdy tablice r i theta mają różne rozmiary.
    
    Hierarchia wywołań:
        system.py -> AiwataLidarSystem.process_scan() -> preprocess_scan()
    """
    # r, theta (N) -> lista BeamResult
    if r.shape != theta.shape:
        raise ValueError("Wektory r i theta muszą mieć ten sam rozmiar")

    results: List[BeamResult] = []
    for ri, ti in zip(r, theta):
        category = classify_beam_by_range(float(ri))

        # UWAGA: używamy ti (pojedyncza wartość kąta), nie theta[i]
        theta_corrected = float(ti + ANGLE_OFFSET_RAD)

        beam = BeamResult(
            theta=theta_corrected,
            r=float(ri),
            category=category,
        )
        results.append(beam)

    return results

def results_to_array(beams: List[BeamResult]) -> np.ndarray:
    """
    Konwertuje listę BeamResult do tablicy numpy.
    
    Przydatne do serializacji lub dalszej obróbki numerycznej.
    
    Argumenty:
        beams (List[BeamResult]): Lista obiektów BeamResult.
    
    Zwraca:
        np.ndarray: Tablica (N, 3) gdzie kolumny to [theta, r, category_int].
            category_int: 0=NONE, 1=HUMAN, 2=OBSTACLE
    
    Hierarchia wywołań:
        Może być wywoływana z zewnątrz do eksportu danych.
    """
    # Lista BeamResult -> tablica N x 3: [theta, r, category_int]
    # category_int: 0=None, 1=Human, 2=Obstacle

    def cat_to_int(cat: BeamCategory) -> int:
        """
        Konwertuje BeamCategory na wartość całkowitą.
        
        Hierarchia wywołań:
            preprocess.py -> results_to_array() -> cat_to_int()
        """
        if cat == BeamCategory.NONE:
            return 0
        elif cat == BeamCategory.HUMAN:
            return 1
        elif cat == BeamCategory.OBSTACLE:
            return 2
        return 0

    N = len(beams)
    arr = np.zeros((N, 3), dtype=float)

    for i, b in enumerate(beams):
        arr[i, 0] = b.theta
        arr[i, 1] = b.r
        arr[i, 2] = cat_to_int(b.category)

    return arr
