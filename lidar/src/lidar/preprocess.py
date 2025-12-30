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
        lidar/src/lidar/preprocess.py -> BeamResult
        lidar/src/lidar/segmentation.py -> Segment
        lidar/src/lidar/occupancy_grid.py -> update_grid_from_scan_result()
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
        lidar/src/lidar/preprocess.py -> process_raw_scan_data() -> BeamResult()
    """
    theta: float            # kąt wiązki [rad]
    r: float                # odległość [m]
    category: BeamCategory  # wynik klasyfikacji


def filter_beam_by_range(range_m: float) -> BeamCategory:
    """
    Klasyfikuje wiązkę na podstawie odległości.
    
    Wiązki poza zakresem [R_MIN_M, R_MAX_M] są odrzucane jako NONE.
    Pozostałe traktowane jako OBSTACLE (kategoria może być zmieniona
    podczas późniejszej segmentacji i klasyfikacji).
    
    Argumenty:
        range_m (float): Odległość wiązki w metrach.
    
    Zwraca:
        BeamCategory: NONE jeśli poza zakresem, OBSTACLE w przeciwnym razie.
    
    Hierarchia wywołań:
        lidar/src/lidar/preprocess.py -> process_raw_scan_data() -> filter_beam_by_range()
    """
    # Klasyfikacja po zasięgu (poza zakresem -> NONE)
    if range_m < R_MIN_M or range_m > R_MAX_M:
        return BeamCategory.NONE
    return BeamCategory.OBSTACLE


def process_raw_scan_data(range_arr: np.ndarray, angle_arr: np.ndarray) -> List[BeamResult]:
    """
    Przetwarza surowy skan LiDAR na listę obiektów BeamResult.
    
    Funkcja wykonuje:
      1. Walidację rozmiaru tablic
      2. Korekcję kąta (LIDAR_ANGLE_OFFSET_DEG)
      3. Wstępną klasyfikację wiązek po zasięgu
    
    Argumenty:
        range_arr (np.ndarray): Tablica odległości w metrach.
        angle_arr (np.ndarray): Tablica kątów w radianach.
    
    Zwraca:
        List[BeamResult]: Lista przetworzoncy punktów z kategoriami.
    
    Wyjątki:
        ValueError: Gdy tablice r i theta mają różne rozmiary.
    
    Hierarchia wywołań:
        lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> process_raw_scan_data()
    """
    # range_arr, angle_arr (N) -> List[BeamResult]
    if range_arr.shape != angle_arr.shape:
        raise ValueError("Wektory r i theta muszą mieć ten sam rozmiar")

    results: List[BeamResult] = []
    for r_val, theta_val in zip(range_arr, angle_arr):
        category = filter_beam_by_range(float(r_val))

        # NOTE: używamy theta_val (pojedyncza wartość kąta), nie angle_arr[i]
        theta_corrected = float(theta_val + ANGLE_OFFSET_RAD)

        beam = BeamResult(
            theta=theta_corrected,
            r=float(r_val),
            category=category,
        )
        results.append(beam)

    return results

def convert_results_to_numpy_array(beams: List[BeamResult]) -> np.ndarray:
    """
    Konwertuje listę BeamResult do tablicy numpy.
    
    Przydatne do serializacji lub dalszej obróbki numerycznej.
    
    Argumenty:
        beams (List[BeamResult]): Lista obiektów BeamResult.
    
    Zwraca:
        np.ndarray: Tablica (N, 3) gdzie kolumny to [theta, r, category_int].
            category_int: 0=NONE, 1=HUMAN, 2=OBSTACLE
    
    Hierarchia wywołań:
        lidar/src/lidar/preprocess.py -> debug/export -> convert_results_to_numpy_array()
    """
    # List BeamResult -> array N x 3: [theta, r, category_int]
    # category_int: 0=None, 1=Human, 2=Obstacle

    def category_to_int(cat: BeamCategory) -> int:
        """
        Konwertuje BeamCategory na wartość całkowitą.
        
        Hierarchia wywołań:
            lidar/src/lidar/preprocess.py -> convert_results_to_numpy_array() -> category_to_int()
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
        arr[i, 2] = category_to_int(b.category)

    return arr
