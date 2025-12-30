from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np

from .preprocess import process_raw_scan_data, BeamResult, BeamCategory
from .segmentation import group_beams_into_segments, categorize_all_segments, Segment
from .occupancy_grid import OccupancyGrid, Pose2D
from .tracking import HumanTracker, HumanTrack

from src.config import (
    MAP_WIDTH_M,
    MAP_HEIGHT_M,
    CELL_SIZE_M,
    TRACK_MAX_MATCH_DISTANCE_M,
    TRACK_MAX_MISSED,
    TRACK_MIN_MOVING_SPEED_M_S,
    TRACK_HUMAN_FILTER_MIN_AGE,
    TRACK_HUMAN_FILTER_MIN_TRAVEL_DIST_M,
)

@dataclass
class ScanResult:
    """
    Wynik kompletnego przetworzenia jednego skanu LiDAR.
    
    Zawiera wszystkie dane wygenerowane przez pipeline przetwarzania:
    preprocessing -> segmentacja -> aktualizacja mapy -> tracking.
    
    Atrybuty:
        beams (List[BeamResult]): Wiązki z przypisanymi kategoriami.
        segments (List[Segment]): Wykryte segmenty obiektów.
        human_tracks (List[HumanTrack]): Aktywne ślady ludzi.
        grid (OccupancyGrid): Zaktualizowana siatka zajętości.
    
    Hierarchia wywołań:
        lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> ScanResult()
    """
    beams: List[BeamResult]         # wiązki z kategoriami
    segments: List[Segment]         # segmenty obiektów
    human_tracks: List[HumanTrack]  # tracki ludzi
    grid: OccupancyGrid             # zaktualizowana mapa


class AiwataLidarSystem:
    """
    Główna klasa systemu przetwarzania LiDAR.
    
    Integruje wszystkie komponenty pipeline'u:
      - Preprocessing (filtrowanie, korekcja kąta)
      - Segmentacja (grupowanie wiązek w obiekty)
      - Occupancy Grid (mapa zajętości 2D)
      - Tracking (sledzenie ludzi w czasie)
    
    Hierarchia wywołań:
        lidar/src/run_live.py -> main() -> AiwataLidarSystem()
        lidar/src/Live_Vis_v3.py -> record_scans() -> AiwataLidarSystem()
    """
    # Główna klasa spinająca preprocess, segmentację, mapę i tracking

    def __init__(
        self,
        map_width_m: float = MAP_WIDTH_M,
        map_height_m: float = MAP_HEIGHT_M,
        cell_size_m: float = CELL_SIZE_M,
        max_match_distance: float = TRACK_MAX_MATCH_DISTANCE_M,
        max_missed: int = TRACK_MAX_MISSED,
    ):
        """
        Inicjalizuje system LiDAR z mapą i trackerem.
        
        Argumenty:
            map_width_m (float): Szerokość mapy w metrach.
            map_height_m (float): Wysokość mapy w metrach.
            cell_size_m (float): Rozmiar komórki siatki w metrach.
            max_match_distance (float): Maks. odległość dopasowania detekcji do tracka.
            max_missed (int): Ile skanów track może zostać bez detekcji.
        
        Hierarchia wywołań:
            lidar/src/run_live.py -> main() -> AiwataLidarSystem()
            lidar/src/Live_Vis_v3.py -> record_scans() -> AiwataLidarSystem()
        """
        # inicjalizacja mapy
        self.grid = OccupancyGrid(
            width_m=map_width_m,
            height_m=map_height_m,
            cell_size_m=cell_size_m,
        )

        # inicjalizacja trackera ludzi
        self.tracker = HumanTracker()
        
        # Uwaga: Parametry max_match_distance i max_missed w oryginalnym kodzie 
        # były przekazywane do __init__, ale HumanTracker korzysta teraz ze stałych z config.py.
        # Zachowano argumenty dla kompatybilności API.

    def process_complete_lidar_scan(
        self,
        range_arr: np.ndarray,
        angle_arr: np.ndarray,
        pose: Pose2D,
        timestamp: float,
    ) -> ScanResult:
        """
        Przetwarza pojedynczy skan LiDAR przez pełny pipeline.
        
        Etapy przetwarzania:
          1. Preprocessing - filtrowanie i klasyfikacja wiązek
          2. Segmentacja - grupowanie wiązek w obiekty
          3. Klasyfikacja segmentów - oznaczenie jako HUMAN/OBSTACLE
          4. Aktualizacja mapy - ray tracing i aktualizacja siatki
          5. Tracking - dopasowanie detekcji do istniejących tracków
          6. Filtrowanie tracków - tylko potwierdzone śledzenia ludzi
        
        Argumenty:
            range_arr (np.ndarray): Tablica odległości w metrach.
            angle_arr (np.ndarray): Tablica kątów w radianach.
            pose (Pose2D): Pozycja robota (x, y, yaw).
            timestamp (float): Czas skanu w sekundach.
        
        Zwraca:
            ScanResult: Kompletny wynik przetwarzania.
        
        Hierarchia wywołań:
            lidar/src/run_live.py -> main() -> AiwataLidarSystem.process_complete_lidar_scan()
            lidar/src/Live_Vis_v3.py -> record_scans() -> AiwataLidarSystem.process_complete_lidar_scan()
        """
        # 1) preprocessing
        beams: List[BeamResult] = process_raw_scan_data(range_arr, angle_arr)

        # 2) segmentacja + klasyfikacja
        segments: List[Segment] = group_beams_into_segments(beams)
        categorize_all_segments(segments)

        # 3) aktualizacja mapy
        self.grid.update_grid_from_scan_result(pose, beams)

        # 4) detekcja HUMAN - przygotowanie segmentów dla trackera (współrzędne globalne)
        # Tracker oczekuje segmentów, ale śledzimy obiekty w układzie GLOBALNYM.
        # Segmenty z segmentacji są w układzie LOKALNYM (robota).
        # Tworzymy kopie segmentów z przeliczonymi środkami na układ globalny.
        
        segments_for_tracker: List[Segment] = []
        for seg in segments:
            # Przelicz środek segmentu na układ świata
            xw, yw = self.grid.transform_local_to_world(pose, seg.center_x, seg.center_y)
            
            # Stwórz kopię segmentu z nowym środkiem (dataclasses.replace tworzy płytką kopię)
            # Zmieniamy tylko center_x i center_y, reszta (np. beams) zostaje bez zmian (i tak ich tracker nie używa głęboko)
            global_seg = replace(seg, center_x=xw, center_y=yw)
            segments_for_tracker.append(global_seg)

        # 5) aktualizacja trackera
        self.tracker.update_tracker(segments_for_tracker, dt_s=0.1) # Zakładamy dt ok 0.1s lub można policzyć z timestamp

        # 6) filtrowanie "prawdziwych" ludzi:
        #    - minimalna długość historii,
        #    - minimalna prędkość,
        #    - minimalne całkowite przemieszczenie.
        all_tracks = self.tracker.get_all_tracks()
        human_tracks: List[HumanTrack] = []
        
        for tr in all_tracks:
            # wystarczająco długa historia?
            if len(tr.history) < TRACK_HUMAN_FILTER_MIN_AGE:
                continue

            # całkowite przemieszczenie od pierwszej do ostatniej pozycji
            t0, x0, y0 = tr.history[0]
            # tr.history[-1] to ostatnia znana pozycja (zazwyczaj bieżąca)
            t1, x1, y1 = tr.history[-1]
            travel = float(np.hypot(x1 - x0, y1 - y0))

            # prędkość – użyj vx, vy obliczonych przez filtr
            vx = float(tr.vx)
            vy = float(tr.vy)
            speed = float(np.hypot(vx, vy))

            if speed < TRACK_MIN_MOVING_SPEED_M_S:
                continue
            if travel < TRACK_HUMAN_FILTER_MIN_TRAVEL_DIST_M:
                continue

            human_tracks.append(tr)

        return ScanResult(
            beams=beams,
            segments=segments,
            human_tracks=human_tracks,
            grid=self.grid,
        )
