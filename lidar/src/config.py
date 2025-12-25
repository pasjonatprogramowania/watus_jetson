from __future__ import annotations

# =========================
# LIDAR – KOREKCJA KĄTA
# =========================

# Stały offset między osią strzałki na obudowie a "zerem" kątowym,
# w stopniach. DODATNI = obrót przeciwnie do ruchu wskazówek zegara (CCW),
# patrząc z góry na lidar.
#
# Przykład:
# - jeśli ściana idealnie na strzałce pojawia się przy +5°,
#   ustaw LIDAR_ANGLE_OFFSET_DEG = -5.0
LIDAR_ANGLE_OFFSET_DEG: float = 0.0

# =========================
# MAPA / OCCUPANCY GRID
# =========================

MAP_WIDTH_M: float = 24.0
MAP_HEIGHT_M: float = 24.0
CELL_SIZE_M: float = 0.25

MIN_HITS_FOR_STATIC: int = 3
HIT_DECAY_FREE: int = 0

# Promień strefy zagrożenia w liczbie komórek (1 = sąsiedztwo 8).
GRID_DANGER_RADIUS_CELLS: int = 1

# Maksymalna wartość licznika obstacle_hits (musi być <= 255, bo uint8).
GRID_MAX_OBSTACLE_HITS: int = 255


# =========================
# LIDAR / HARDWARE
# =========================

LIDAR_PORT: str = "/dev/ttyUSB0"

LIDAR_BAUDRATE: int = 230400
LIDAR_TIMEOUT: float = 1.0     # [s]

# ile pakietów sklejamy na jeden pełniejszy skan
FULL_SCAN_PACKETS: int = 60


# =========================
# PREPROCESS (zasięg wiązek)
# =========================

R_MIN_M: float = 0.05          # [m] – martwa strefa < 5 cm
R_MAX_M: float = 12.0          # [m] – ucinamy > 12 m


# =========================
# SEGMENTACJA
# =========================

SEG_MAX_DISTANCE_JUMP_M: float = 1.0
SEG_MAX_ANGLE_JUMP_DEG: float = 10.0
SEG_MIN_BEAMS: int = 2

# Heurystyka "segment wygląda jak człowiek"
SEG_HUMAN_MIN_R_M: float = 0.7        # minimalna odległość środka segmentu
SEG_HUMAN_MAX_R_M: float = 6.0        # maksymalna odległość środka segmentu

SEG_HUMAN_MIN_LENGTH_M: float = 0.3   # minimalna długość segmentu
SEG_HUMAN_MAX_LENGTH_M: float = 0.8   # maksymalna długość segmentu

SEG_HUMAN_MIN_BEAMS: int = 9          # minimalna liczba wiązek w segmencie

# =========================
# TRACKING LUDZI – dopasowanie / życie tracka
# =========================

# maks. odległość między detekcją a trackiem, żeby je dopasować
TRACK_MAX_MATCH_DISTANCE_M: float = 1.0

# ile kolejnych skanów możemy "nie widzieć" tracka, zanim go uznamy za zniknięty
TRACK_MAX_MISSED: int = 5

# prędkość od której uznajemy obiekt za ruchomy (logika "is_moving")
TRACK_MIN_MOVING_SPEED_M_S: float = 0.15


# =========================
# TRACKING LUDZI – filtry / stany
# =========================

# wygładzanie pozycji i prędkości (0..1, im wyżej, tym większa waga nowego pomiaru)
TRACK_POS_ALPHA: float = 0.6
TRACK_VEL_ALPHA: float = 0.5

# ile trafień zanim track stanie się "confirmed"
TRACK_MIN_CONFIRM_HITS: int = 3

# ile skanów może przeżyć niepotwierdzony track bez detekcji
TRACK_MAX_MISSED_TENTATIVE: int = 2

# minimalna surowa prędkość, powyżej której traktujemy ruch jako istotny
TRACK_RAW_VEL_MIN_SPEED_M_S: float = 0.3

# Minimalna długość historii tracka (ile aktualizacji), żeby traktować go jako człowieka.
TRACK_HUMAN_FILTER_MIN_AGE: int = 4

# Minimalne całkowite przemieszczenie [m] od powstania tracka.
TRACK_HUMAN_FILTER_MIN_TRAVEL_DIST_M: float = 0.20

# =========================
# TRACKING LUDZI – predykcja trajektorii
# =========================

# horyzont czasowy i krok predykcji
TRACK_PREDICTION_HORIZON_S: float = 0.8
TRACK_PREDICTION_STEP_S: float = 0.2

# parametry promienia niepewności predykcji
TRACK_PREDICTION_BASE_UNCERTAINTY_M: float = 0.1
TRACK_PREDICTION_GROWTH_UNCERTAINTY: float = 0.25
TRACK_PREDICTION_MAX_SPEED_FOR_UNCERTAINTY: float = 3.0


# =========================
# TRACKING LUDZI – RE-ID (mini pamięć)
# =========================

# jak długo pamiętamy stary track w archiwum (sekundy)
TRACK_REID_MAX_AGE_S: float = 60.0

# maks. odległość nowej detekcji od przewidywanej pozycji archiwalnego tracka
TRACK_REID_MAX_DISTANCE_M: float = 4.0

TRACK_MOTION_ZONE_RADIUS_M: float = 6

# =========================
# SAFETY – strefy i sektory
# =========================

# promienie stref (m)
SAFETY_STOP_RADIUS_M: float = 2.5
SAFETY_WARN_RADIUS_M: float = 4.0   # musi być >= SAFETY_STOP_RADIUS_M

# kąt półsektora CENTER (deg)
# CENTER = [-SAFETY_CENTER_ANGLE_DEG, +SAFETY_CENTER_ANGLE_DEG]
# LEFT   = > +SAFETY_CENTER_ANGLE_DEG
# RIGHT  = < -SAFETY_CENTER_ANGLE_DEG
SAFETY_CENTER_ANGLE_DEG: float = 30.0

