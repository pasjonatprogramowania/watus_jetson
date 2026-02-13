from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Optional
import time

# =================
# Preprocess Types
# =================

class BeamCategory(Enum):
    """
    Enum kategorii wiązek LiDAR po przetwarzaniu wstępnym.
    
    Wartości:
        NONE: Wiązka poza zakresem lub nieprawidłowa (do pominięcia)
        HUMAN: Wiązka należąca do segmentu wykrytego jako człowiek
        OBSTACLE: Wiązka należąca do statycznej przeszkody
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
    """
    theta: float            # kąt wiązki [rad]
    r: float                # odległość [m]
    category: BeamCategory  # wynik klasyfikacji


# =================
# Segmentation Types
# =================

@dataclass
class Segment:
    """
    Reprezentacja segmentu wykrytego w skanie LiDAR.
    
    Segment to grupa sąsiadujących wiązek reprezentujących jeden obiekt.
    
    Atrybuty:
        id (int): Unikalny identyfikator segmentu w danym skanie.
        beams (List[BeamResult]): Lista wiązek tworzących segment.
        center_x (float): Środek segmentu w osi X (metry, układ robota).
        center_y (float): Środek segmentu w osi Y (metry, układ robota).
        mean_r (float): Średnia odległość segmentu od LiDAR (metry).
        mean_theta (float): Średni kąt segmentu (radiany).
        length (float): Przybliżona długość obiektu (metry).
        base_category (BeamCategory): Kategoria segmentu (OBSTACLE/HUMAN).
    """
    id: int
    beams: List[BeamResult]

    center_x: float
    center_y: float
    mean_r: float
    mean_theta: float
    length: float          # przybliżona długość obiektu [m]
    base_category: BeamCategory  # kategoria segmentu (np. OBSTACLE / HUMAN)


# =================
# Tracking Types
# =================

@dataclass
class HumanTrack:
    """
    Reprezentacja śledzonego człowieka.
    
    Atrybuty:
        id (str): Unikalne UUID śladu.
        x (float): Pozycja X po filtracji.
        y (float): Pozycja Y po filtracji.
        vx (float): Prędkość X.
        vy (float): Prędkość Y.
        state (str): Stan śladu: "init", "confirmed", "archived", "deleted".
        missed_frames (int): Liczba kolejnych skanów bez dopasowania wykrycia.
        matched_frames (int): Liczba kolejnych skanów z dopasowaniem.
        age (float): Czas od utworzenia śladu (sekundy).
        last_update_time (float): Znacznik czasu ostatniej aktualizacji.
        history (List[Tuple[float, float, float]]): Historia pozycji (czas, x, y).
    """
    id: str
    x: float
    y: float
    vx: float
    vy: float
    state: str  # "init", "confirmed", "archived", "deleted"
    
    missed_frames: int
    matched_frames: int
    
    age: float
    last_update_time: float
    
    # Historia pozycji: lista krotek (timestamp, x, y)
    history: List[Tuple[float, float, float]] = field(default_factory=list)


# =================
# Occupancy Grid Types
# =================

@dataclass
class Pose2D:
    """Reprezentacja pozycji i orientacji robota 2D."""
    x: float
    y: float
    yaw: float  # [rad]


class CellType(IntEnum):
    """Typy zawartości komórki mapy zajętości."""
    UNKNOWN = 0
    FREE = 1
    STATIC_OBSTACLE = 2
    HUMAN = 3


class CellDanger(IntEnum):
    """Poziom zagrożenia w komórce."""
    NO_DANGER = 0
    DANGER = 1


class SafetySector(IntEnum):
    """Trzy sektory kątowe względem osi X robota dla systemu bezpieczeństwa."""
    LEFT = 0
    CENTER = 1
    RIGHT = 2


@dataclass
class SafetyOutput:
    """
    Zredukowany wynik dla urządzenia docelowego / PLC / Systemu Bezpieczeństwa.
    
    Zawiera flagi STOP/WARNING oraz szczegółowe informacje sektorowe.
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
