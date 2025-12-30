from __future__ import annotations

# =========================
# =========================
# LIDAR – KOREKCJA KĄTA
# =========================

# Stałe przesunięcie między osią strzałki na obudowie a "zerem" kątowym,
# w stopniach. DODATNIE = obrót przeciwnie do ruchu wskazówek zegara (CCW),
# patrząc od góry lidara.
#
# Przykład:
# - jeśli ściana idealnie zrównana ze strzałką pojawia się na +5°,
#   ustaw LIDAR_ANGLE_OFFSET_DEG = -5.0
LIDAR_ANGLE_OFFSET_DEG: float = 0.0

# =========================
# =========================
# MAPA / SIATKA ZAJĘTOŚCI
# =========================

# Szerokość mapy zajętości w metrach.
MAP_WIDTH_M: float = 24.0
# Wysokość mapy zajętości w metrach.
MAP_HEIGHT_M: float = 24.0
# Rozmiar pojedynczej komórki mapy w metrach (rozdzielczość).
CELL_SIZE_M: float = 0.25

# Ile razy komórka musi zostać trafiona, aby uznać ją za "statyczną przeszkodę".
MIN_HITS_FOR_STATIC: int = 3
# O ile zmniejszyć licznik trafień podczas "wolnego" przelotu (ray tracing).
HIT_DECAY_FREE: int = 0

# Promień strefy zagrożenia w liczbie komórek (1 = sąsiedztwo 8-elementowe).
GRID_DANGER_RADIUS_CELLS: int = 1

# Maksymalna wartość licznika obstacle_hits (musi być <= 255, ponieważ jest uint8).
GRID_MAX_OBSTACLE_HITS: int = 255


# =========================
# =========================
# LIDAR / SPRZĘT
# =========================

# Port szeregowy LiDAR-a.
# Dla Windows: "COM3", "COM4". Dla Linux: "/dev/ttyUSB0".
LIDAR_PORT: str = "/dev/ttyUSB0"

LIDAR_BAUDRATE: int = 230400
LIDAR_TIMEOUT: float = 1.0     # [s]

# ile pakietów połączyć w jeden pełniejszy skan
FULL_SCAN_PACKETS: int = 60


# =========================
# =========================
# PREPROCESS (zasięg wiązki)
# =========================

R_MIN_M: float = 0.05          # [m] – martwa strefa < 5 cm
R_MAX_M: float = 12.0          # [m] – odcięcie > 12 m


# =========================
# =========================
# SEGMENTACJA
# =========================

# Maksymalna różnica odległości (metry) między punktami, aby uznać je za ciągły segment.
SEG_MAX_DISTANCE_JUMP_M: float = 1.0
# Maksymalna różnica kątowa (stopnie) dla ciągłości segmentu.
SEG_MAX_ANGLE_JUMP_DEG: float = 10.0
SEG_MIN_BEAMS: int = 2

# Heurystyka "segment wygląda jak człowiek"
SEG_HUMAN_MIN_R_M: float = 0.7        # minimalna odległość środka segmentu
SEG_HUMAN_MAX_R_M: float = 6.0        # maksymalna odległość środka segmentu

SEG_HUMAN_MIN_LENGTH_M: float = 0.3   # minimalna długość segmentu
SEG_HUMAN_MAX_LENGTH_M: float = 0.8   # maksymalna długość segmentu

SEG_HUMAN_MIN_BEAMS: int = 9          # minimalna liczba wiązek w segmencie

# =========================
# =========================
# ŚLEDZENIE LUDZI – dopasowanie / czas życia śladu
# =========================

# maksymalna odległość między detekcją a śladem, aby je dopasować
# Maksymalna odległość (metry) środka obiektu od predykcji śladu, aby nadal był to ten sam obiekt.
TRACK_MAX_MATCH_DISTANCE_M: float = 1.0

# ile kolejnych skanów ślad może być "niewidoczny", zanim zostanie uznany za utracony
TRACK_MAX_MISSED: int = 5

# prędkość, od której uznajemy obiekt za poruszający się (logika "is_moving")
TRACK_MIN_MOVING_SPEED_M_S: float = 0.15


# =========================
# =========================
# ŚLEDZENIE LUDZI – filtry / stany
# =========================

# wygładzanie pozycji i prędkości (0..1, wyższa wartość oznacza większą wagę nowego pomiaru)
TRACK_POS_ALPHA: float = 0.6
TRACK_VEL_ALPHA: float = 0.5

# ile trafień zanim ślad stanie się "potwierdzony"
TRACK_MIN_CONFIRM_HITS: int = 3

# ile skanów niepotwierdzony ślad może przetrwać bez detekcji
TRACK_MAX_MISSED_TENTATIVE: int = 2

# minimalna surowa prędkość, powyżej której traktujemy ruch jako znaczący
TRACK_RAW_VEL_MIN_SPEED_M_S: float = 0.3

# Minimalna długość historii śladu (ile aktualizacji), aby traktować go jako człowieka.
TRACK_HUMAN_FILTER_MIN_AGE: int = 4

# Minimalne całkowite przemieszczenie [m] od momentu utworzenia śladu.
TRACK_HUMAN_FILTER_MIN_TRAVEL_DIST_M: float = 0.20

# =========================
# =========================
# ŚLEDZENIE LUDZI – predykcja trajektorii
# =========================

# horyzont czasowy i krok predykcji
TRACK_PREDICTION_HORIZON_S: float = 0.8
TRACK_PREDICTION_STEP_S: float = 0.2

# parametry promienia niepewności predykcji
TRACK_PREDICTION_BASE_UNCERTAINTY_M: float = 0.1
TRACK_PREDICTION_GROWTH_UNCERTAINTY: float = 0.25
TRACK_PREDICTION_MAX_SPEED_FOR_UNCERTAINTY: float = 3.0


# =========================
# ŚLEDZENIE LUDZI – RE-ID (mini pamięć)
# =========================

# jak długo pamiętamy stary ślad w archiwum (sekundy)
TRACK_REID_MAX_AGE_S: float = 60.0

# maksymalna odległość nowej detekcji od przewidywanej pozycji zarchiwizowanego śladu
TRACK_REID_MAX_DISTANCE_M: float = 4.0

TRACK_MOTION_ZONE_RADIUS_M: float = 6

# =========================
# =========================
# BEZPIECZEŃSTWO – strefy i sektory
# =========================

# promienie stref (m)
SAFETY_STOP_RADIUS_M: float = 2.5
SAFETY_WARN_RADIUS_M: float = 4.0   # musi być >= SAFETY_STOP_RADIUS_M

# Kąt pół-sektora ŚRODKOWEGO (stopnie)
# CENTER = [-SAFETY_CENTER_ANGLE_DEG, +SAFETY_CENTER_ANGLE_DEG]
# LEFT   = > +SAFETY_CENTER_ANGLE_DEG
# RIGHT  = < -SAFETY_CENTER_ANGLE_DEG
SAFETY_CENTER_ANGLE_DEG: float = 30.0

