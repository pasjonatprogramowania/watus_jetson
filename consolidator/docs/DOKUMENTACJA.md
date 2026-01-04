# Dokumentacja Modulu Consolidator

## Wprowadzenie

Modul consolidator odpowiada za fuzje danych z kamery i lidara w jeden spojny format. Laczy informacje wizualne (typ obiektu, bounding box) z danymi przestrzennymi (pozycja, odleglosc) na podstawie zgodnosci katow widzenia.

---

## Architektura

Modul sklada sie z pojedynczego pliku `consolidator.py` zawierajacego:
- Funkcje `compute_lidar_angle()` - obliczanie kata z pozycji lidar
- Funkcje `main()` - glowna petla konsolidacji

### Hierarchia Wywolan

```
consolidator.py
  -> main()
     -> compute_lidar_angle() (dla kazdego tracka lidar)
     -> dopasowanie kamera-lidar
     -> zapis do consolidator.json
```

---

## Algorytm Fuzji

### Krok 1: Odczyt Danych

System odczytuje dwa pliki JSON:
- `../lidar/data/lidar.json` - dane z lidara (slady osob)
- `../oczy_watusia/camera.json` - dane z kamery (wykryte obiekty)

### Krok 2: Obliczanie Kata Lidar

Dla kazdego sladu lidar obliczany jest kat poziomy wzgledem osi kamery:

```python
angle_deg = math.degrees(math.atan2(x_offset, y_forward))
```

Gdzie:
- `x_offset` - przesuniecie boczne (+ lewo, - prawo)
- `y_forward` - odleglosc do przodu

### Krok 3: Dopasowanie

Obiekt z kamery jest dopasowywany do sladu lidar gdy:
```python
abs(angle_camera - angle_lidar) <= CAMERA_FOV_HALF
```

Gdzie `CAMERA_FOV_HALF = 51.0` stopni (polowa kata widzenia 102 stopni).

### Krok 4: Laczenie Danych

Po dopasowaniu, wpis zawiera:
- Dane z lidara: id, pozycja, czas
- Dane z kamery: plec/typ, bounding box, kat

---

## Konfiguracja

### Stale w Kodzie

| Stala | Wartosc | Opis |
|-------|---------|------|
| CAMERA_FOV_HALF | 51.0 | Polowa kata widzenia kamery (stopnie) |

### Sciezki Plikow

| Plik | Sciezka | Opis |
|------|---------|------|
| Lidar Input | ../lidar/data/lidar.json | Dane wejsciowe z lidara |
| Camera Input | ../oczy_watusia/camera.json | Dane wejsciowe z kamery |
| Output | consolidator.json | Polaczone dane wyjsciowe |

---

## Struktura Folderow

```
consolidator/
|-- consolidator.py             # Glowny skrypt
|-- consolidator.json           # Dane wyjsciowe (generowane)
|-- docs/
    |-- DOKUMENTACJA.md         # Ten plik
```

### Wymagania

Modul oczekuje, ze nastepujace pliki beda dostepne:
1. `../lidar/data/lidar.json` - generowany przez modul lidar
2. `../oczy_watusia/camera.json` - generowany przez modul warstwa_wizji

---

## Format Wejsciowy

### lidar.json

```json
{
  "tracks": [{
    "id": "abc123",
    "type": "human",
    "last_position": [1.5, 3.2],
    "last_update_time": 1704380401.5
  }]
}
```

### camera.json

```json
{
  "countOfPeople": 1,
  "objects": [{
    "isPerson": true,
    "type": "male",
    "angle": 15.5,
    "left": 100, "top": 50,
    "width": 150, "height": 300
  }]
}
```

---

## Format Wyjsciowy (consolidator.json)

```json
{
  "tracks": [{
    "id": "abc123",
    "type": "human",
    "last_position": [1.5, 3.2],
    "last_update_time": 1704380401.5,
    "gender": "male",
    "camera_bbox": {
      "left": 100, "top": 50,
      "width": 150, "height": 300
    },
    "camera_angle": 15.5
  }]
}
```

### Opis Pol

| Pole | Zrodlo | Opis |
|------|--------|------|
| id | lidar | ID sladu |
| type | lidar | Typ obiektu |
| last_position | lidar | Pozycja [x, y] w metrach |
| last_update_time | lidar | Timestamp |
| gender | camera | Plec (jesli isPerson=true) |
| object_type | camera | Typ obiektu (jesli isPerson=false) |
| camera_bbox | camera | Bounding box z obrazu |
| camera_angle | camera | Kat z kamery |

---

## Petla Glowna

Modul dziala w nieskonczonej petli z interwałem 0.1s:

1. Odczyt plikow JSON (z obsluga bledow)
2. Dla kazdego tracka lidar:
   - Oblicz kat
   - Sprawdz zgodnosc z kamera
   - Polacz dane jesli pasuja
3. Zapis do consolidator.json
4. Czekaj 100ms

---

## Ograniczenia

1. **Jedna osoba** - Algorytm zaklada jedna osobe w polu widzenia kamery (camera_data["objects"][0])
2. **Synchronizacja czasowa** - Brak mechanizmu synchronizacji timestampow miedzy kamera a lidarem
3. **Kat FOV** - Dopasowanie opiera sie wylacznie na zgodnosci katow
4. **Sciezki wzgledne** - Sciezki do plikow sa hardkodowane wzglednie

---

## Rozwiazywanie Problemow

| Problem | Rozwiazanie |
|---------|-------------|
| Brak dopasowania | Sprawdz kalibracje FOV kamery |
| Puste dane | Sprawdz czy lidar i camera generuja pliki |
| Bledy JSON | Sprawdz czy pliki nie sa zapisywane w tym samym momencie |
| Zle katy | Dostosuj CAMERA_FOV_HALF |

---

## Uruchomienie

```bash
cd consolidator
python consolidator.py
```

Modul dziala w petli - zatrzymaj Ctrl+C.
