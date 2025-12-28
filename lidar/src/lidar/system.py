from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .preprocess import preprocess_scan, BeamResult, BeamCategory
from .segmentation import segment_scan, assign_segment_categories, Segment
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
        Zwracany przez: AiwataLidarSystem.process_scan()
        Używany w: run_live.py, Live_Vis_v3.py
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
        run_live.py -> main() -> AiwataLidarSystem.process_scan()
        Live_Vis_v3.py -> record_scans() -> AiwataLidarSystem.process_scan()
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
            run_live.py -> main() -> AiwataLidarSystem()
            Live_Vis_v3.py -> record_scans() -> AiwataLidarSystem()
        """
        # inicjalizacja mapy
        self.grid = OccupancyGrid(
            width_m=map_width_m,
            height_m=map_height_m,
            cell_size_m=cell_size_m,
        )

        # inicjalizacja trackera ludzi
        self.tracker = HumanTracker(
            max_match_distance=max_match_distance,
            max_missed=max_missed,
        )

    def process_scan(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        pose: Pose2D,
        t: float,
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
            r (np.ndarray): Tablica odległości w metrach.
            theta (np.ndarray): Tablica kątów w radianach.
            pose (Pose2D): Pozycja robota (x, y, yaw).
            t (float): Czas skanu w sekundach.
        
        Zwraca:
            ScanResult: Kompletny wynik przetwarzania.
        
        Hierarchia wywołań:
            system.py -> AiwataLidarSystem.process_scan()
                -> preprocess_scan(), segment_scan(), assign_segment_categories()
                -> grid.update_from_scan(), tracker.update()
        """
        # 1) preprocessing
        beams: List[BeamResult] = preprocess_scan(r, theta)

        # 2) segmentacja + klasyfikacja
        segments: List[Segment] = segment_scan(beams)
        assign_segment_categories(segments)

        # 3) aktualizacja mapy
        self.grid.update_from_scan(pose, beams)

        # 4) detections HUMAN – jedna detekcja na segment (środek obiektu)
        detections: List[Tuple[float, float]] = []
        for seg in segments:
            if seg.base_category == BeamCategory.HUMAN:
                # center_x, center_y są w układzie lokalnym robota
                # używamy _local_to_world z OccupancyGrid (mimo underscore)
                xw, yw = self.grid._local_to_world(pose, seg.center_x, seg.center_y)
                detections.append((xw, yw))

        # 5) update trackera – wszystkie tracki (kandydaci + potencjalni ludzie)
        all_tracks: List[HumanTrack] = self.tracker.update(detections, t)

        # 6) filtr "prawdziwych" ludzi:
        #    - minimalna długość historii,
        #    - minimalna prędkość,
        #    - minimalne całkowite przemieszczenie.
        human_tracks: List[HumanTrack] = []
        for tr in all_tracks:
            # wystarczająco długa historia?
            if len(tr.history) < TRACK_HUMAN_FILTER_MIN_AGE:
                continue

            # całkowite przemieszczenie od pierwszej do ostatniej pozycji
            t0, x0, y0 = tr.history[0]
            t1, x1, y1 = tr.history[-1]
            travel = float(np.hypot(x1 - x0, y1 - y0))

            # prędkość – użyj property "speed" jeśli jest, inaczej policz z vx, vy
            vx = float(getattr(tr, "vx", 0.0))
            vy = float(getattr(tr, "vy", 0.0))
            speed = float(getattr(tr, "speed", np.hypot(vx, vy)))

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

