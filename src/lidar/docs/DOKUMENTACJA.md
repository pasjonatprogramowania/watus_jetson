# Dokumentacja Modulu Lidar

## Wprowadzenie

Modul lidar odpowiada za przetwarzanie danych z czujnika LiDAR. Wykrywa i sledzi ludzi, tworzy mape zajetosci oraz identyfikuje zagrozenia. System wykorzystuje filtr Alfa-Beta do sledzenia i algorytm Wegierski do asocjacji detekcji.

---

## Architektura

### Komponenty

- **AiwataLidarSystem (system.py)** - Glowna klasa systemu
- **LidarDriver (hardware/lidar_driver.py)** - Sterownik sprzetowy
- **Preprocess (lidar/preprocess.py)** - Filtrowanie zasiegu
- **Segmentation (lidar/segmentation.py)** - Grupowanie punktow
- **HumanTracker (lidar/tracking.py)** - Sledzenie ludzi
- **OccupancyGrid (lidar/occupancy_grid.py)** - Mapa zajetosci

---

## Konfiguracja (src/config.py)

### Sprzet LiDAR

| Parametr | Domyslna | Opis |
|----------|----------|------|
| LIDAR_PORT | /dev/ttyUSB0 | Port szeregowy (Windows: COM3) |
| LIDAR_BAUDRATE | 230400 | Predkosc transmisji |
| LIDAR_TIMEOUT | 1.0 | Timeout (s) |
| LIDAR_ANGLE_OFFSET_DEG | 0.0 | Korekcja kata |

### Mapa Zajetosci

| Parametr | Domyslna | Opis |
|----------|----------|------|
| MAP_WIDTH_M | 24.0 | Szerokosc mapy (m) |
| MAP_HEIGHT_M | 24.0 | Wysokosc mapy (m) |
| CELL_SIZE_M | 0.25 | Rozmiar komorki (m) |
| MIN_HITS_FOR_STATIC | 3 | Trafienia dla przeszkody statycznej |

### Preprocessing

| Parametr | Domyslna | Opis |
|----------|----------|------|
| R_MIN_M | 0.05 | Martwa strefa (m) |
| R_MAX_M | 12.0 | Maksymalny zasieg (m) |

### Segmentacja

| Parametr | Domyslna | Opis |
|----------|----------|------|
| SEG_MAX_DISTANCE_JUMP_M | 1.0 | Max roznica odleglosci |
| SEG_MAX_ANGLE_JUMP_DEG | 10.0 | Max roznica kata |
| SEG_HUMAN_MIN_R_M | 0.7 | Min odleglosc czlowieka |
| SEG_HUMAN_MAX_R_M | 6.0 | Max odleglosc czlowieka |
| SEG_HUMAN_MIN_LENGTH_M | 0.3 | Min szerokosc segmentu |
| SEG_HUMAN_MAX_LENGTH_M | 0.8 | Max szerokosc segmentu |
| SEG_HUMAN_MIN_BEAMS | 9 | Min wiazek w segmencie |

### Sledzenie

| Parametr | Domyslna | Opis |
|----------|----------|------|
| TRACK_MAX_MATCH_DISTANCE_M | 1.0 | Max odleglosc dopasowania |
| TRACK_MAX_MISSED | 5 | Max brakujacych skanow |
| TRACK_POS_ALPHA | 0.6 | Waga filtra pozycji |
| TRACK_VEL_ALPHA | 0.5 | Waga filtra predkosci |
| TRACK_MIN_CONFIRM_HITS | 3 | Trafien do potwierdzenia |

### Strefy Bezpieczenstwa

| Parametr | Domyslna | Opis |
|----------|----------|------|
| SAFETY_STOP_RADIUS_M | 2.5 | Promien strefy STOP |
| SAFETY_WARN_RADIUS_M | 4.0 | Promien ostrzegania |
| SAFETY_CENTER_ANGLE_DEG | 30.0 | Pol-kat sektora centralnego |

---

## Struktura Folderow

```
lidar/
|-- run.py                      # Uruchomienie przetwarzania
|-- run_vis.py                  # Uruchomienie wizualizacji
|-- requirements.txt
|-- docs/DOKUMENTACJA.md
|-- data/                       # Dane wyjsciowe
|   |-- lidar.json              # Aktualny stan sladow
|-- src/
    |-- config.py               # Konfiguracja
    |-- run_live.py             # Petla na zywo
    |-- Live_Vis_v3.py          # Wizualizacja
    |-- Grid_vis.py             # Wizualizacja siatki
    |-- check_lidar.py          # Test polaczenia
    |-- hardware/
    |   |-- lidar_driver.py     # Sterownik
    |-- lidar/
        |-- preprocess.py       # Preprocessing
        |-- segmentation.py     # Segmentacja
        |-- tracking.py         # Sledzenie
        |-- occupancy_grid.py   # Mapa zajetosci
        |-- system.py           # System glowny
```

---

## Algorytm Sledzenia

### Stany Sladu
- **init** - Nowy, zbieranie dowodow
- **confirmed** - Potwierdzony, aktywny
- **archived** - Tymczasowo utracony
- **deleted** - Usuniety

### Filtr Alfa-Beta
Predykcja pozycji i korekcja na podstawie pomiarow. Wspolczynniki ALPHA i BETA kontroluja wygladzanie.

### Algorytm Wegierski
Optymalne dopasowanie detekcji do sladow minimalizujace laczny koszt (odleglosc).

---

## Format Wyjsciowy (data/lidar.json)

```json
{
  "tracks": [{
    "id": "a1b2c3d4",
    "type": "human",
    "last_position": [1.5, 3.2],
    "velocity": [0.2, 0.5],
    "state": "confirmed",
    "age": 5.3
  }],
  "timestamp": 1704380401.5
}
```

---

## Rozwiazywanie Problemow

| Problem | Rozwiazanie |
|---------|-------------|
| LiDAR nie odpowiada | Sprawdz port, uprawnienia (Linux: chmod 666) |
| Brak detekcji ludzi | Sprawdz zasieg 0.7-6m, parametry SEG_HUMAN_* |
| Niestabilne sledzenie | Zmniejsz TRACK_POS_ALPHA |
| Falszywe detekcje | Zwieksz SEG_HUMAN_MIN_BEAMS |

---

## Ograniczenia

1. Skanowanie tylko w jednej plaszczyznie (2D)
2. Odbicia lustrzane moga powodowac bledy
3. Przesloniecia - obiekty za innymi niewidoczne
4. Heurystyka geometryczna dla klasyfikacji czlowieka
