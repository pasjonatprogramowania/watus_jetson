from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict, Any, Optional
import math

import numpy as np

from src.config import (
    MAP_WIDTH_M,
    MAP_HEIGHT_M,
    CELL_SIZE_M,
    MIN_HITS_FOR_STATIC,
    HIT_DECAY_FREE,
    GRID_DANGER_RADIUS_CELLS,
    GRID_MAX_OBSTACLE_HITS,
    SAFETY_STOP_RADIUS_M,       # NEW
    SAFETY_WARN_RADIUS_M,       # NEW
    SAFETY_CENTER_ANGLE_DEG,    # NEW
)
from .preprocess import BeamResult, BeamCategory


# =========================
# PODSTAWOWE TYPY GEOMETRII
# =========================

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float  # [rad]


class CellType(IntEnum):
    UNKNOWN = 0
    FREE = 1
    STATIC_OBSTACLE = 2
    HUMAN = 3


class CellDanger(IntEnum):
    NO_DANGER = 0
    DANGER = 1


# =========================
# === SAFETY OUTPUT ===
# =========================

class SafetySector(IntEnum):
    """Trzy sektory kątowe względem osi X robota."""
    LEFT = 0
    CENTER = 1
    RIGHT = 2


@dataclass
class SafetyOutput:
    """
    Zredukowany wynik dla urządzenia docelowego / PLC.

    - 2 strefy odległościowe: STOP i WARNING,
    - 3 sektory: LEFT / CENTER / RIGHT,
    - proste bity + minimalna odległość do zagrożenia.
    """

    # globalne bity
    stop: bool
    warning: bool

    # globalna minimalna odległość do dowolnego zagrożenia (m)
    min_dist_m: float

    # ile komórek zagrożenia w każdej strefie odległościowej
    humans_in_stop_zone: int
    humans_in_warn_zone: int

    # bity sektorowe w strefie STOP
    stop_left: bool
    stop_center: bool
    stop_right: bool

    # bity sektorowe w strefie WARNING
    warn_left: bool
    warn_center: bool
    warn_right: bool


# =========================
# GŁÓWNA KLASA MAPY
# =========================

class OccupancyGrid:
    """
    Prosta lokalna mapa zajętości w układzie XY robota.
    - Oś X: przód (+), tył (-)
    - Oś Y: lewo (+), prawo (-)
    Robot w (0,0), yaw w Pose2D.
    """

    def __init__(
        self,
        width_m: float = MAP_WIDTH_M,
        height_m: float = MAP_HEIGHT_M,
        cell_size_m: float = CELL_SIZE_M,
    ):
        """
        Inicjalizuje pustą siatkę zajętości (Occupancy Grid).
        
        Tworzy macierze numpy dla typów komórek, zagrożeń i liczników trafień.
        Ustawia koordynaty (0,0) robota w centrum mapy.
        
        Argumenty:
            width_m (float): Szerokość mapy w metrach.
            height_m (float): Wysokość mapy w metrach.
            cell_size_m (float): Rozmiar pojedynczej komórki w metrach.
            
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.__init__() -> OccupancyGrid()
        """
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.cell_size = float(cell_size_m)

        self.nx = int(round(self.width_m / self.cell_size))
        self.ny = int(round(self.height_m / self.cell_size))

        # zakładamy, że (0,0) robota jest w środku mapy
        self.x_min = -self.width_m / 2.0
        self.y_min = -self.height_m / 2.0

        # główne warstwy
        self.cell_type = np.full((self.ny, self.nx), CellType.UNKNOWN, dtype=np.int8)
        self.cell_danger = np.full((self.ny, self.nx), CellDanger.NO_DANGER, dtype=np.int8)
        self.obstacle_hits = np.zeros((self.ny, self.nx), dtype=np.uint8)

    # =========================
    # KONWERSJE (świat <-> siatka)
    # =========================

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Świat (m) -> indeksy komórki (ix, iy). Zwraca None gdy poza mapą.
        
        Hierarchia wywołań:
            occupancy_grid.py -> _mark_ray_free() -> world_to_cell()
            occupancy_grid.py -> update_from_scan() -> world_to_cell()
        """
        ix = int((x - self.x_min) / self.cell_size)
        iy = int((y - self.y_min) / self.cell_size)

        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Środek komórki (ix,iy) -> (x,y) w metrach.
        
        Hierarchia wywołań:
            occupancy_grid.py -> compute_safety_output() -> cell_to_world()
        """
        x = self.x_min + (ix + 0.5) * self.cell_size
        y = self.y_min + (iy + 0.5) * self.cell_size
        return x, y

    # =========================
    # WEWNĘTRZNE POMOCNICZE
    # =========================

    def _set_cell_type(self, ix: int, iy: int, new_type: CellType):
        """
        Ustawia typ komórki z uwzględnieniem priorytetów.
        
        Nie nadpisuje przeszkód statycznych (STATIC_OBSTACLE) typem HUMAN, 
        aby uniknąć migotania przy nakładaniu się obiektów.
        
        Argumenty:
            ix (int): Indeks X komórki.
            iy (int): Indeks Y komórki.
            new_type (CellType): Nowy typ komórki.
            
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _set_cell_type()
            occupancy_grid.py -> _bump_obstacle_hits() -> _set_cell_type()
        """
        """Ustawia typ komórki z prostą logiką priorytetów."""
        current = CellType(int(self.cell_type[iy, ix]))

        # HUMAN nie nadpisuje STATIC_OBSTACLE (statyczna bariera ma priorytet)
        if current == CellType.STATIC_OBSTACLE and new_type == CellType.HUMAN:
            return

        # w pozostałych przypadkach nadpisujemy
        self.cell_type[iy, ix] = int(new_type)

    def _bump_obstacle_hits(self, ix: int, iy: int):
        """
        Inkrementuje licznik trafień przeszkody w danej komórce.
        
        Jeśli licznik przekroczy próg MIN_HITS_FOR_STATIC, komórka staje się STATIC_OBSTACLE.
        Zabezpiecza przed przepełnieniem licznika (GRID_MAX_OBSTACLE_HITS).
        
        Argumenty:
            ix (int): Indeks X komórki.
            iy (int): Indeks Y komórki.
            
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _bump_obstacle_hits()
        """
        """Zwiększa licznik obstacle_hits z saturacją."""
        hits = int(self.obstacle_hits[iy, ix])
        if hits < GRID_MAX_OBSTACLE_HITS:
            hits += 1
        self.obstacle_hits[iy, ix] = hits

        if hits >= MIN_HITS_FOR_STATIC:
            self._set_cell_type(ix, iy, CellType.STATIC_OBSTACLE)

    def _decay_obstacle_hit(self, ix: int, iy: int):
        """
        Zmniejsza licznik trafień przeszkody, gdy promień przechodzi przez komórkę (jest wolna).
        
        Jeśli licznik spadnie do 0, komórka jest oznaczana jako FREE (chyba że jest HUMAN).
        
        Argumenty:
            ix (int): Indeks X komórki.
            iy (int): Indeks Y komórki.
            
        Hierarchia wywołań:
            occupancy_grid.py -> _mark_ray_free() -> _decay_obstacle_hit()
        """
        """Opcjonalny zanik obstacle_hits gdy promień przechodzi jako FREE."""
        if HIT_DECAY_FREE <= 0:
            return

        hits = int(self.obstacle_hits[iy, ix])
        if hits > 0:
            hits = max(0, hits - HIT_DECAY_FREE)
            self.obstacle_hits[iy, ix] = hits
            if hits == 0 and self.cell_type[iy, ix] != CellType.HUMAN:
                self.cell_type[iy, ix] = int(CellType.FREE)

    def _beam_to_local_xy(self, beam: BeamResult) -> Tuple[float, float]:
        """
        Konwersja wiązki lidarowej (r,theta) -> lokalne XY robota.
        
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _beam_to_local_xy()
        """
        x = beam.r * np.cos(beam.theta)
        y = beam.r * np.sin(beam.theta)
        return x, y

    def _local_to_world(self, pose: Pose2D, x_local: float, y_local: float) -> Tuple[float, float]:
        """
        Lokalne (x,y) robota -> globalne (x,y).
        
        Używa pozycji i orientacji robota (Pose2D).
        
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _local_to_world()
        """
        cos_yaw = np.cos(pose.yaw)
        sin_yaw = np.sin(pose.yaw)

        x_world = pose.x + cos_yaw * x_local - sin_yaw * y_local
        y_world = pose.y + sin_yaw * x_local + cos_yaw * y_local

        return x_world, y_world

    def _mark_ray_free(self, pose: Pose2D, x_hit: float, y_hit: float):
        """
        Wykonuje Ray Tracing od robota do punktu trafienia, czyszcząc komórki po drodze.
        
        Dla każdej komórki na drodze promienia:
        - Jeśli brak historii trafień -> ustawia FREE.
        - Jeśli są trafienia -> zmniejsza licznik (decay).
        - Pomija komórki oznaczone jako HUMAN.
        
        Argumenty:
            pose (Pose2D): Pozycja początkowa (robot).
            x_hit (float): Współrzędna X końca promienia.
            y_hit (float): Współrzędna Y końca promienia.
            
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _mark_ray_free()
        """
        # start (pozycja robota)
        x0, y0 = pose.x, pose.y
        x1, y1 = x_hit, y_hit

        dx = x1 - x0
        dy = y1 - y0
        dist = float(np.hypot(dx, dy))
        if dist <= 1e-6:
            return

        # krok co 1/2 komórki żeby nie robić dziur
        step = self.cell_size * 0.5
        n_steps = int(dist / step)

        for i in range(n_steps):
            t = (i * step) / dist
            x = x0 + t * dx
            y = y0 + t * dy

            idx = self.world_to_cell(x, y)
            if idx is None:
                continue
            ix, iy = idx

            # nie ruszamy komórek HUMAN (człowiek ma priorytet)
            if self.cell_type[iy, ix] == CellType.HUMAN:
                continue

            if self.obstacle_hits[iy, ix] == 0:
                self.cell_type[iy, ix] = int(CellType.FREE)
            else:
                self._decay_obstacle_hit(ix, iy)

    def _mark_danger_around(self, ix_hit: int, iy_hit: int):
        """
        Oznacza komórki w säsiedztwie wykrytego człowieka jako DANGER.
        
        Tworzy strefę buforową wokół człowieka o promieniu GRID_DANGER_RADIUS_CELLS.
        
        Argumenty:
            ix_hit (int): Indeks X komórki z człowiekiem.
            iy_hit (int): Indeks Y komórki z człowiekiem.
            
        Hierarchia wywołań:
            occupancy_grid.py -> update_from_scan() -> _mark_danger_around()
        """
        """Oznacza komórki w promieniu GRID_DANGER_RADIUS_CELLS jako DANGER."""
        r = GRID_DANGER_RADIUS_CELLS
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ix = ix_hit + dx
                iy = iy_hit + dy
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    self.cell_danger[iy, ix] = int(CellDanger.DANGER)

    # =========================
    # GŁÓWNA AKTUALIZACJA MAPY
    # =========================

    def update_from_scan(self, pose: Pose2D, beams: List[BeamResult]):
        """
        Aktualizuje siatkę na podstawie pojedynczego skanu.
        - ray tracing czyści FREE wzdłuż wiązki
        - końcówka wiązki zwiększa obstacle_hits lub oznacza HUMAN
        - warstwa DANGER:
            * jest czyszczona na każdy skan,
            * ustawiana TYLKO dla wiązek HUMAN.
            
        Argumenty:
            pose (Pose2D): Aktualna pozycja robota.
            beams (List[BeamResult]): Lista wiązek z Lidaru.
            
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.process_scan() -> update_from_scan()
        """
        # 1) warstwa DANGER jest chwilowa -> czyścimy na start
        self.cell_danger[:, :] = int(CellDanger.NO_DANGER)

        for beam in beams:
            if beam.category == BeamCategory.NONE:
                continue

            # przelicz wiązkę na punkt w świecie
            x_local, y_local = self._beam_to_local_xy(beam)
            x_world, y_world = self._local_to_world(pose, x_local, y_local)

            # ray tracing FREE po drodze
            self._mark_ray_free(pose, x_world, y_world)

            # końcówka wiązki -> aktualizacja komórki docelowej
            hit_idx = self.world_to_cell(x_world, y_world)
            if hit_idx is None:
                continue
            ix_hit, iy_hit = hit_idx

            if beam.category == BeamCategory.OBSTACLE:
                # tylko statyczna przeszkoda, bez DANGER
                self._bump_obstacle_hits(ix_hit, iy_hit)

            elif beam.category == BeamCategory.HUMAN:
                # człowiek:
                self._set_cell_type(ix_hit, iy_hit, CellType.HUMAN)
                # i lokalne rozszerzenie zagrożenia
                self._mark_danger_around(ix_hit, iy_hit)

            # inne kategorie można ewentualnie obsłużyć tutaj

    # =========================
    # STATYSTYKI / SERIALIZACJA
    # =========================

    def count_cells_by_type(self) -> Dict[CellType, int]:
        """
        Zlicza liczbę komórek każdego typu (FREE, OCCUPIED, HUMAN, UNKNOWN).
        
        Zwraca:
            Dict[CellType, int]: Słownik {typ_komórki: liczebność}.
            
        Hierarchia wywołań:
             lidar/src/lidar/system.py -> AiwataLidarSystem.process_scan() -> logging/debug
        """
        """Zwraca słownik: typ komórki -> liczba."""
        result: Dict[CellType, int] = {}
        for t in CellType:
            result[t] = int(np.sum(self.cell_type == int(t)))
        return result

    def count_danger_cells(self) -> int:
        """
        Zlicza komórki oznaczone jako DANGER (strefa niebezpieczna).
        
        Zwraca:
            int: Liczba niebezpiecznych komórek.
            
        Hierarchia wywołań:
             lidar/src/lidar/system.py -> AiwataLidarSystem.process_scan() -> logging/debug
        """
        """Liczba komórek z flagą DANGER."""
        return int(np.sum(self.cell_danger == int(CellDanger.DANGER)))

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serializuje stan mapy do słownika JSON.
        
        Konwertuje macierze numpy na listy Pythona.
        
        Zwraca:
            Dict[str, Any]: Słownik reprezentujący stan mapy (wymiary, dane komórek).
            
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.get_scan_data() -> to_json_dict()
        """
        """
        Struktura gotowa do zapisania w JSON (mapa + warstwa DANGER).
        Uwaga: tablice zamienione na listy intów.
        """
        return {
            "map_width_m": float(self.width_m),
            "map_height_m": float(self.height_m),
            "cell_size_m": float(self.cell_size),
            "x_min": float(self.x_min),
            "y_min": float(self.y_min),
            "cell_type": self.cell_type.astype(int).tolist(),
            "cell_danger": self.cell_danger.astype(int).tolist(),
        }

    # =========================
    # === LICZENIE SAFETY OUTPUT ===
    # =========================

    def compute_safety_output(self) -> SafetyOutput:
        """
        Analizuje bieżącą mapę cell_danger i liczy:
        - bity stop / warning,
        - bity sektorowe L/C/P dla STOP i WARNING,
        - minimalną odległość do zagrożenia,
        - liczbę komórek zagrożenia w strefie STOP / WARNING.

        Uwaga:
        - STOP ma priorytet nad WARNING (jeśli jest STOP, warning=True też).
        - Rozpatrujemy tylko komórki z CellDanger.DANGER.
        
        Zwraca:
            SafetyOutput: Obiekt z flagami bezpieczeństwa.
            
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.process_scan() -> compute_safety_output()
        """
        # znajdź wszystkie komórki zagrożenia
        ys, xs = np.where(self.cell_danger == int(CellDanger.DANGER))

        # jeśli nie ma zagrożeń -> czysty output
        if len(xs) == 0:
            return SafetyOutput(
                stop=False,
                warning=False,
                min_dist_m=float("inf"),
                humans_in_stop_zone=0,
                humans_in_warn_zone=0,
                stop_left=False,
                stop_center=False,
                stop_right=False,
                warn_left=False,
                warn_center=False,
                warn_right=False,
            )

        # konwersja kąta środkowego na radiany
        center_angle_rad = math.radians(SAFETY_CENTER_ANGLE_DEG)

        min_dist = float("inf")
        count_stop = 0
        count_warn = 0

        stop_left = stop_center = stop_right = False
        warn_left = warn_center = warn_right = False

        for iy, ix in zip(ys, xs):
            x, y = self.cell_to_world(ix, iy)
            r = math.hypot(x, y)
            if r < 1e-6:
                continue

            # strefa odległościowa
            zone = None
            if r <= SAFETY_STOP_RADIUS_M:
                zone = "stop"
            elif r <= SAFETY_WARN_RADIUS_M:
                zone = "warn"
            else:
                # poza zasięgiem safety -> ignorujemy
                continue

            # sektor kątowy względem osi X
            theta = math.atan2(y, x)  # rad
            # CENTER: [-center_angle, +center_angle]
            if -center_angle_rad <= theta <= center_angle_rad:
                sector = SafetySector.CENTER
            elif theta > center_angle_rad:
                sector = SafetySector.LEFT
            else:
                sector = SafetySector.RIGHT

            # aktualizacja min. dystansu
            if r < min_dist:
                min_dist = r

            # akumulacja per strefa/sektor
            if zone == "stop":
                count_stop += 1
                if sector == SafetySector.LEFT:
                    stop_left = True
                elif sector == SafetySector.CENTER:
                    stop_center = True
                else:
                    stop_right = True
            elif zone == "warn":
                count_warn += 1
                if sector == SafetySector.LEFT:
                    warn_left = True
                elif sector == SafetySector.CENTER:
                    warn_center = True
                else:
                    warn_right = True

        # globalne bity
        stop_flag = count_stop > 0
        # warning jest True jeśli coś jest w warn_zone LUB w stop_zone
        warning_flag = (count_warn > 0) or stop_flag

        return SafetyOutput(
            stop=stop_flag,
            warning=warning_flag,
            min_dist_m=min_dist if min_dist < float("inf") else float("inf"),
            humans_in_stop_zone=count_stop,
            humans_in_warn_zone=count_warn,
            stop_left=stop_left,
            stop_center=stop_center,
            stop_right=stop_right,
            warn_left=warn_left,
            warn_center=warn_center,
            warn_right=warn_right,
        )

    # =========================
    # WYGODNE API DLA INNYCH MODUŁÓW
    # =========================

    def beam_to_world_point(self, beam: BeamResult, pose: Pose2D) -> Tuple[float, float]:
        """
        Konwertuje wynik pomiaru wiązki na punkt globalny (x, y).
        
        Helper API dla innych modułów.
        
        Argumenty:
            beam (BeamResult): Dane wiązki (kąt, odległość).
            pose (Pose2D): Pozycja robota.
            
        Zwraca:
            Tuple[float, float]: Pozycja w układzie globalnym.
            
        Hierarchia wywołań:
            Zewnętrzne moduły wizualizacji/debuggowania.
        """
        """BeamResult + Pose2D -> punkt (x,y) w świecie."""
        x_local, y_local = self._beam_to_local_xy(beam)
        x_world, y_world = self._local_to_world(pose, x_local, y_local)
        return x_world, y_world

    def local_to_world_point(self, pose: Pose2D, x_local: float, y_local: float) -> Tuple[float, float]:
        """
        Wrapper na wewnętrzną konwersję lokalne -> globalne.
        
        Argumenty:
            pose (Pose2D): Pozycja robota.
            x_local (float): X w układzie robota.
            y_local (float): Y w układzie robota.
            
        Zwraca:
            Tuple[float, float]: X, Y w układzie świata.
            
        Hierarchia wywołań:
            Zewnętrzne moduły wizualizacji.
        """
        """Lokalne (x,y) robota -> globalne (x,y)."""
        return self._local_to_world(pose, x_local, y_local)
