# Dokumentacja Modulu Warstwa Wizji (Computer Vision)

## Wprowadzenie

Modul warstwa_wizji stanowi komponent systemu WATUS odpowiedzialny za przetwarzanie obrazu w czasie rzeczywistym. Jego glownym zadaniem jest detekcja osob i obiektow w strumieniu wideo, sledzenie ich ruchow oraz ekstrakcja dodatkowych atrybutow takich jak plec, wiek, emocje czy kolor ubrania. Modul zostal zaprojektowany z mysla o pracy na urzadzeniu NVIDIA Jetson, jednak moze rowniez dzialac na standardowym komputerze z karta graficzna NVIDIA lub w trybie CPU (ze znacznie nizsza wydajnoscia).

---

## Architektura Systemu

### Glowne Komponenty

System wizyjny sklada sie z nastepujacych elementow:

**CVAgent** - Centralna klasa agenta wizyjnego znajdujaca sie w pliku `src/cv_agent.py`. Odpowiada za koordynacje calego procesu przetwarzania obrazu, od przechwytywania klatek po generowanie wynikowego JSON z informacjami o wykrytych obiektach.

**cv_utils** - Pakiet pomocniczych funkcji i klas zawierajacy:
- Obliczenia geometryczne (katy, pozycje)
- Przetwarzanie jasnosci obrazu
- Sledzenie obiektow (tracker)
- Integracje z danymi Lidar
- Rysowanie nakladek na obraz

**img_classifiers** - Pakiet klasyfikatorow obrazu zawierajacy:
- Klasyfikator kolorow ubrania
- Klasyfikator typu ubrania
- Klasyfikator obiektow wojskowych (MIL)

**model_trainer** - Pakiet do trenowania modeli YOLO zawierajacy:
- Przygotowanie zbiorow danych
- Skrypty treningowe
- Wizualizacja danych

### Hierarchia Wywolan

Glowna sciezka wykonania programu wyglada nastepujaco:

```
main.py
  -> CVAgent.__init__()
     -> _get_classifiers() (ladowanie klasyfikatorow)
     -> _init_video_capture() (inicjalizacja strumienia wideo)
     -> _load_models() (ladowanie modeli YOLO)
  -> CVAgent.run()
     -> _detect_objects() (detekcja obiektow)
     -> _process_person() (przetwarzanie osob)
     -> _process_lidar_matching() (dopasowanie do Lidar)
     -> _handle_weapon_detection() (wykrywanie broni)
```

---

## Wymagania Sprzetowe i Programowe

### Minimalne Wymagania Sprzetowe

**Dla trybu GPU (zalecany):**
- NVIDIA Jetson Orin AGX lub inna platforma Jetson
- Alternatywnie: komputer z karta graficzna NVIDIA (CUDA Compute Capability 6.0+)
- Minimum 8 GB RAM
- Kamera USB lub strumien sieciowy H.264

**Dla trybu CPU:**
- Procesor x86_64 lub ARM64
- Minimum 8 GB RAM
- Kamera USB lub strumien sieciowy

### Wymagania Programowe

- Python 3.10 lub nowszy
- PyTorch z obsluga CUDA (dla trybu GPU)
- OpenCV z obsluga GStreamer (dla strumieni sieciowych na Jetson)
- Ultralytics YOLO
- Biblioteki z pliku requirements.txt

---

## Konfiguracja

### Zmienne Konfiguracyjne

Konfiguracja modulu odbywa sie poprzez stale w pliku `src/cv_agent.py`:

```python
GPU_ENABLED = True        # Wlaczenie trybu GPU
ESCAPE_BUTTON = "q"       # Klawisz wyjscia z podgladu
```

### Parametry Inicjalizacji CVAgent

Przy tworzeniu instancji CVAgent mozna przekazac nastepujace parametry:

| Parametr | Typ | Domyslna Wartosc | Opis |
|----------|-----|------------------|------|
| weights_path | str | "yolo12s.pt" | Sciezka do wag modelu YOLO |
| imgsz | int | 640 | Rozmiar obrazu wejsciowego dla modelu |
| source | int/str | 0 | Zrodlo wideo (indeks kamery lub sciezka) |
| cap | VideoCapture | None | Opcjonalny obiekt przechwytywania |
| json_save_func | callable | None | Funkcja do zapisu wynikow JSON |
| use_net_stream | bool | True | Czy uzywac strumienia sieciowego GStreamer |

### Parametry Metody run()

| Parametr | Typ | Domyslna | Opis |
|----------|-----|----------|------|
| save_video | bool | False | Zapisuj przetworzony film |
| out_path | str | "output.mp4" | Sciezka wyjsciowego pliku wideo |
| show_window | bool | True | Wyswietlaj okno podgladu |
| det_stride | int | 1 | Co ile klatek uruchamiac detekcje |
| show_fps | bool | True | Wyswietlaj licznik FPS |
| verbose | bool | True | Logowanie na konsole |
| verbose_window | bool | True | Rysowanie szczegolowych informacji |
| fov_deg | int | 102 | Kat widzenia kamery w stopniach |
| consolidate_with_lidar | bool | False | Integracja z danymi Lidar |

---

## Struktura Folderow

Oczekiwana struktura katalogu warstwa_wizji:

```
warstwa_wizji/
|-- main.py                     # Punkt wejscia programu
|-- __init__.py                 # Inicjalizacja pakietu
|-- camera.json                 # Aktualny stan kamery (wyjscie)
|-- camera.jsonl                # Historia detekcji (wyjscie)
|-- requirements.txt            # Zaleznosci Python
|-- readme.md                   # Podstawowy opis
|
|-- docs/                       # Dokumentacja
|   |-- DOKUMENTACJA.md         # Ten plik
|
|-- models/                     # Modele ML (WYMAGANE DO UTWORZENIA)
|   |-- clothes.pt              # Model detekcji ubrania
|   |-- guns.pt                 # Model detekcji broni
|   |-- yolo12s.pt              # Glowny model detekcji (lub inny YOLO)
|
|-- src/
|   |-- __init__.py             # Inicjalizacja
|   |-- cv_agent.py             # Glowny agent wizyjny
|   |
|   |-- cv_utils/               # Narzedzia pomocnicze
|   |   |-- __init__.py
|   |   |-- angle.py            # Obliczenia katowe
|   |   |-- brightness.py       # Analiza jasnosci
|   |   |-- cv_wrapper.py       # Wrapper OpenCV
|   |   |-- detection_processors.py  # Procesory detekcji
|   |   |-- frame_overlay.py    # Nakladki na obraz
|   |   |-- lidar_integration.py     # Integracja z Lidar
|   |   |-- new_tracker.py      # Nowy tracker obiektow
|   |   |-- old_tracker.py      # Stary tracker (legacy)
|   |   |-- parallel.py         # Przetwarzanie rownolegle
|   |
|   |-- img_classifiers/        # Klasyfikatory obrazu
|   |   |-- __init__.py
|   |   |-- color_classifier.py      # Klasyfikacja kolorow
|   |   |-- image_classifier.py      # Klasyfikatory glowne
|   |   |-- mil_object_classifier.py # Obiekty wojskowe
|   |   |-- utils.py                 # Pomocnicze funkcje
|   |
|   |-- model_trainer/          # Trening modeli
|       |-- __init__.py
|       |-- __main__.py         # Punkt wejscia trenera
|       |-- data_visualizer.py  # Wizualizacja danych
|       |-- yolo_dataset.py     # Przygotowanie zbiorow
|       |-- yolo_train.py       # Skrypty treningowe
```

### Wymagane Foldery do Utworzenia

Przed uruchomieniem systemu nalezy utworzyc folder `models/` i umiescic w nim wymagane modele:

1. **clothes.pt** - Model YOLO do detekcji ubrania
2. **guns.pt** - Model YOLO do detekcji broni
3. **Glowny model** - np. yolo12s.pt, yolov8n.pt lub RT-DETR

Modele mozna pobrac z oficjalnych zrodel Ultralytics lub wytrenowac samodzielnie uzywajac modulu model_trainer.

---

## Format Wyjsciowy JSON

System generuje dane wyjsciowe w formacie JSON/JSONL. Struktura pojedynczego rekordu:

```json
{
  "objects": [
    {
      "id": 1,
      "type": "person",
      "left": 120.5,
      "top": 80.3,
      "width": 150.0,
      "height": 300.0,
      "isPerson": true,
      "angle": -15.5,
      "additionalInfo": [
        {"gender": "male"},
        {"age": "adult"},
        {"clothes": [{...}]}
      ],
      "lidar": {
        "lidar_id": "abc123",
        "distance": 3.5
      }
    }
  ],
  "countOfPeople": 1,
  "countOfObjects": 1,
  "suggested_mode": "light",
  "brightness": 0.65
}
```

### Opis Pol

| Pole | Typ | Opis |
|------|-----|------|
| objects | array | Lista wykrytych obiektow |
| id | int | Identyfikator sledzenia obiektu |
| type | string | Typ obiektu (person, car, etc.) |
| left, top | float | Pozycja lewego-gornego rogu |
| width, height | float | Wymiary bounding box |
| isPerson | bool | Czy obiekt jest osoba |
| angle | float | Kat wzgledem osi kamery (stopnie) |
| additionalInfo | array | Dodatkowe atrybuty |
| lidar | object | Dane z integracji Lidar |
| countOfPeople | int | Liczba wykrytych osob |
| countOfObjects | int | Laczna liczba obiektow |
| suggested_mode | string | Sugerowany tryb ("light"/"dark") |
| brightness | float | Srednia jasnosc obrazu (0-1) |

---

## Strumien Wideo GStreamer

W trybie `use_net_stream=True` system uzywa potoku GStreamer do odbioru strumienia H.264 przez UDP:

```
udpsrc port=5000 ! 
application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! 
rtph264depay ! h264parse ! nvv4l2decoder ! 
nvvidconv ! video/x-raw, format=BGRx ! 
videoconvert ! video/x-raw, format=BGR ! 
appsink drop=1
```

### Wymagania dla GStreamer

- Na Jetson: OpenCV musi byc skompilowany z obsluga GStreamer
- Wymagane wtyczki: gst-plugins-base, gst-plugins-good, gst-plugins-bad
- Na Jetson wymagany jest nvv4l2decoder do dekodowania sprzetowego

### Konfiguracja Strumienia

Zrodlo musi wysylac strumien RTP H.264 na port UDP 5000. Przykladowa komenda FFmpeg do wysylania:

```bash
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtp rtp://JETSON_IP:5000
```

---

## Integracja z Lidar

System moze laczyc dane wizualne z danymi Lidar dla uzyskania dokladniejszych informacji o odleglosci i pozycji obiektow.

### Wymagania

- Plik `lidar.json` musi byc dostepny pod sciezka `../lidar/data/lidar.json`
- Parametr `consolidate_with_lidar=True` przy wywolaniu metody run()

### Algorytm Dopasowania

1. Obliczany jest kat poziomy obiektu na podstawie jego pozycji w obrazie
2. Pobierane sa dane trackow z pliku lidar.json
3. Dla kazdego tracka Lidar obliczany jest kat na podstawie wspolrzednych (x, y)
4. Obiekty sa dopasowywane gdy roznica katow jest mniejsza niz polowa kata widzenia kamery

### Ograniczenia Integracji

- Dzialanie jest optymalne dla pojedynczej osoby w polu widzenia
- Przy wielu osobach dopasowanie moze byc niejednoznaczne
- Wymaga precyzyjnej kalibracji kata widzenia kamery (parametr fov_deg)

---

## Klasyfikatory Obrazu

### Klasyfikator Kolorow (color_classifier.py)

Analizuje kolor ubrania wykrytej osoby. Dziala na wycietym fragmencie obrazu odpowiadajacym wykrytemu ubraniu.

### Klasyfikator Obrazu (image_classifier.py)

Zawiera trzy glowne klasyfikatory:
- **emotion_classifier** - Rozpoznawanie emocji twarzy
- **gender_classifier** - Klasyfikacja plci
- **age_classifier** - Szacowanie grupy wiekowej

Klasyfikatory sa domyslnie wylaczone (linia 255: `if GPU_ENABLED and False`). Aby je wlaczyc, nalezy zmodyfikowac warunek.

### Klasyfikator Obiektow Wojskowych (mil_object_classifier.py)

Specjalistyczny klasyfikator do rozpoznawania pojazdow wojskowych i sprzetu.

---

## Detekcja Broni

System posiada wbudowany modul detekcji broni. Funkcjonalnosc jest domyslnie wylaczona i mozna ja aktywowac klawiszem 'w' podczas dzialania programu.

### Wlaczanie/Wylaczanie

W czasie dzialania programu nacisnij klawisz 'w' aby przelaczyc tryb detekcji broni. Stan zostanie wyswietlony w konsoli:
```
Weapon Detection: ON
Weapon Detection: OFF
```

### Model Broni

Model musi byc umieszczony jako `models/guns.pt`. Powinien byc to model YOLO wytrenowany na zbiorze danych z bronia.

---

## Mechanizm Cache

System implementuje mechanizm cache dla wynikow klasyfikacji, aby unikac ponownego przetwarzania tych samych osob w kazdej klatce.

### Parametry Cache

- **cache_ttl** - Czas zycia wpisu cache w klatkach (domyslnie 100)
- Cache jest indeksowany po track_id obiektu
- Przechowuje: ubrania, emocje, plec, wiek, dane Lidar

### Aktualizacja Cache

Cache jest aktualizowany gdy:
- Obiekt o danym track_id nie istnieje w cache
- Minelo wiecej niz cache_ttl klatek od ostatniej aktualizacji

---

## Obliczenia FPS

System oblicza wygladzony FPS uzywajac filtru wykladniczego (EMA - Exponential Moving Average):

```python
ema_fps = (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps
```

Gdzie:
- `ema_alpha = 0.1` - wspolczynnik wygladzania
- `inst_fps` - chwilowy FPS obliczony z roznica czasow miedzy klatkami

### Wyswietlanie FPS

FPS jest aktualizowany na ekranie co 0.5 sekundy aby unikac migotania wartosci.

---

## Ograniczenia i Znane Problemy

### Ograniczenia Wydajnosciowe

1. **Tryb CPU** - Wydajnosc jest znacznie nizsza niz w trybie GPU. Zalecane jest uzycie prostszych modeli (np. yolov8n) i zmniejszenie rozmiaru obrazu.

2. **Strumien GStreamer** - Na systemach bez Jetson moze byc trudny do skonfigurowania. Alternatywnie mozna uzyc kamery USB (source=0).

3. **Klasyfikatory** - Klasyfikatory emocji/plci/wieku sa wylaczone domyslnie ze wzgledu na obciazenie obliczeniowe.

### Znane Problemy

1. **Wielokrotne osoby** - Integracja z Lidar dziala najlepiej dla pojedynczej osoby w polu widzenia.

2. **Utrata sledzenia** - Przy szybkim ruchu lub zaslonieniu obiekt moze otrzymac nowy track_id.

3. **Tryb dark/light** - Automatyczne przelaczanie trybow na podstawie jasnosci moze byc niedokladne przy nierownomiernym oswietleniu.

---

## Rozwiazywanie Problemow

### Brak obrazu z kamery

1. Sprawdz czy kamera jest podlaczona i dostepna
2. Dla strumienia sieciowego sprawdz czy port 5000 jest otwarty
3. Zweryfikuj kompilacje OpenCV z GStreamer: `print(cv2.getBuildInformation())`

### Niski FPS

1. Zmniejsz rozdzielczosc obrazu (imgsz)
2. Zwieksz det_stride (np. do 2 lub 3)
3. Uzyj mniejszego modelu YOLO
4. Wylacz verbose_window jezeli jest wlaczony

### Brak detekcji

1. Sprawdz czy model jest poprawnie zaladowany
2. Zweryfikuj czy obiekt jest w zasiegu kamery
3. Dostosuj prog pewnosci (conf) w metodzie _detect_objects

### Bledy CUDA

1. Zainstaluj odpowiednia wersje PyTorch dla wersji CUDA
2. Sprawdz dostepnosc GPU: `torch.cuda.is_available()`
3. Zweryfikuj kompatybilnosc modelu z wersja CUDA
