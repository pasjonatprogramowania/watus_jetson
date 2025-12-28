# Wielki Plan Unifikacji Systemu Watus Jetson (Szczegółowy)

## Tło i Cel

Obecny system `watus_jetson` działa, ale jest "posklejany" z różnych kawałków, które komunikują się w sposób nieefektywny (pliki JSON na dysku) i są trudne w utrzymaniu. Celem jest stworzenie jednolitej, solidnej architektury, która będzie działać szybko, niezawodnie i będzie łatwa do rozwijania.

Poniższy plan jest rozpisany na bardzo małe, konkretne kroki. Traktuj go jak dokładną instrukcję montażu.

## Moduł 0: Fundamenty (watus_core)

Zanim zaczniemy cokolwiek zmieniać w logice, musimy stworzyć wspólny grunt. Obecnie każdy moduł radzi sobie sam. To się zmieni.

### Zadanie 0.1: Struktura Katalogów

* [ ]  Utwórz na poziomie głównym folder `watus_core`.
* [ ]  Utwórz w nim pusty plik `__init__.py`.

### Zadanie 0.2: Centralna Konfiguracja (watus_core/config.py)

Zamiast szukać `.env` w różnych folderach, zrobimy jeden loader.

* [ ]  Stwórz plik `watus_core/config.py`.

* **Kod:** Użyj biblioteki `python-dotenv`. Niech szuka pliku `.env` w katalogu głównym projektu (tam gdzie folder `.git`).
* **Zadanie:** Zdefiniuj w nim klasę `Config` lub zestaw stałych, które pokrywają WSZYSTKIE potrzeby systemu.
  * Wszystkie porty ZMQ (np. `ZMQ_PORT_LIDAR`, `ZMQ_PORT_VISION`, `ZMQ_PORT_AUDIO`).
  * Ścieżki do modeli (YOLO, Whisper).
  * Parametry urządzeń (ID kamery, port LiDARu).
* **Weryfikacja:** Napisz mały skrypt `test_config.py`, który importuje `Config` i wypisuje jedną zmienną.

### Zadanie 0.3: Wspólny Logger (watus_core/logger.py)

Koniec z `print("dupa")`.

* [ ]  Stwórz plik `watus_core/logger.py`.

* **Kod:** Skonfiguruj standardowy `logging` w Pythonie.
  * Format: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
  * Handler 1: Konsola (`StreamHandler`).
  * Handler 2: Plik rotowany (`RotatingFileHandler`), np. `logs/watus_system.log`.
* **Funkcja pomocnicza:** `get_logger(name)` która zwraca skonfigurowany logger.

### Zadanie 0.4: Komunikacja ZMQ (watus_core/comm.py)

To będzie krwiobieg systemu.

* [ ]  Stwórz plik `watus_core/comm.py`.

* **Klasa Publisher:**
  * W `__init__` przyjmuje port i opcjonalnie bind/connect (zazwyczaj bind).
  * Metoda `send(topic: str, data: dict)`:
    * Dodaje timestamp: `data['ts'] = time.time()`.
    * Serializuje do JSON.
    * Wysyła multipart: `[topic.encode(), json_bytes]`.
* **Klasa Subscriber:**
  * W `__init__` przyjmuje listę portów lub adresów.
  * Metoda `recv()` (z timeoutem lub non-blocking):
    * Odbiera multipart.
    * Deserializuje JSON.
    * Zwraca `(topic, data)`.

## Moduł 1: Lidar (Migracja na ZMQ)

Lidar obecnie zarzyna dysk zapisując JSONa co klatkę.

### Zadanie 1.1: Przełączenie `run_live.py`

Plik: `lidar/src/run_live.py`.

* [ ]  Dodaj import: `from watus_core.comm import Publisher` oraz `from watus_core.config import Config`.
* [ ]  W `main()`: Zainicjalizuj `pub = Publisher(port=Config.ZMQ_PORT_LIDAR)`.
* [ ]  Znajdź funkcję `export_scan` (lub miejsce gdzie jest wywoływana).
* [ ]  ZAMIAST pisać do pliku `lidar.json`, wywołaj `pub.send("sensor.lidar", { "tracks": ... })`.
  * Format danych JSON zostaw taki sam, żeby nie popsuć logiki (chyba że jest tam mnóstwo śmieci, wtedy wyślij tylko listę obiektów/tracków).
* [ ]  (Opcjonalnie) Zapis do plików sesji (`session_...`) możesz zostawić do debugowania, ale dodaj flagę `Config.SAVE_SESSION_LOGS = False` domyślnie.

## Moduł 2: Wizja (Migracja na ZMQ)

Ta sama historia co z Lidarem.

### Zadanie 2.1: Przełączenie `warstwa_wizji/main.py`

* [ ]  Dodaj importy z `watus_core`.
* [ ]  W klasie `CVAgent`, w metodzie `run`:
  * Zainicjalizuj `self.pub = Publisher(port=Config.ZMQ_PORT_VISION)`.
  * W pętli, po wykryciu obiektów: `self.pub.send("sensor.vision", detections)`.
  * Usuń/zakomentuj zapis do `camera.json`.

## Moduł 3: Audio (Unifikacja)

Audio już używa ZMQ, ale pewnie "po swojemu". Sprowadźmy to do standardu.

### Zadanie 3.1: Config i Logger

* [ ]  Przejrzyj `warstwa_audio/watus_audio/config.py`. Zastąp lokalne zmienne odwołaniami do `watus_core.config`.
* [ ]  Wszędzie gdzie jest `print`, użyj `watus_core.logger`.

### Zadanie 3.2: Użycie wspólnego ZMQ

* [ ]  Jeśli Audio ma własną klasę do ZMQ, podmień ją na `watus_core.comm`. Dzięki temu, jeśli kiedyś zmienimy JSON na inny format, zmienimy to tylko w jednym miejscu.

## Moduł 4: Consolidator (Mózg Operacyjny)

To jest najważniejsza zmiana architektoniczna. Consolidator przestaje być "czytaczem plików".

### Zadanie 4.1: Nowy `consolidator_zmq.py`

Stwórz nowy plik, nie psuj starego od razu.

* **Klasa WorldModel:**

  * Trzyma w pamięci (zmienne instancji): `latest_lidar_data`, `latest_vision_data`, `latest_audio_status`.
  * Metoda `update_lidar(data)`: Nadpisuje `latest_lidar_data`.
  * Metoda `update_vision(data)`: Nadpisuje `latest_vision_data`.
  * Metoda `fuse()`: Wykonuje logikę łączenia (tę samą co wcześniej: dopasowanie po kącie). Zwraca scalony stan świata.
* **Pętla Główna (`main`):**

  * `sub = Subscriber(ports=[Config.ZMQ_PORT_LIDAR, Config.ZMQ_PORT_VISION])`.
  * `pub = Publisher(port=Config.ZMQ_PORT_WORLD)`.
  * `model = WorldModel()`.
  * `while True:`
    * `topic, data = sub.recv()` (non-blocking).
    * Jeśli przyszły dane -> zaktualizuj `model`.
    * Raz na X milisekund (np. 100ms) -> `state = model.fuse()`.
    * `pub.send("world.state", state)`.

## Moduł 5: Warstwa LLM (Brain)

LLM musi wiedzieć co się dzieje, bez pytania.

### Zadanie 5.1: Kontekst w Pamięci

W `warstwa_llm/src/main.py` (FastAPI):

* [ ]  Przy starcie aplikacji uruchom w osobnym wątku (`threading.Thread`) prostego Subskrybenta ZMQ (`Config.ZMQ_PORT_WORLD`).
* [ ]  Ten wątek aktualizuje zmienną globalną (np. `CURRENT_WORLD_CONTEXT`).

### Zadanie 5.2: Wzbogacanie Promptu

W funkcji `process_question`:

* [ ]  Przed wysłaniem pytania do LLM, pobierz `CURRENT_WORLD_CONTEXT`.
* [ ]  Sformatuj to jako tekst (np. "Widzę: Jan Kowalski (3m, 15 stopni w lewo), Nieznany Obiekt (5m).").
* [ ]  Doklej to do promptu systemowego lub user message.

* **Rezultat:** LLM wie co widzi robot w czasie rzeczywistym!

## Moduł 6: Zarządzanie Procesami (Ecosystem Watchdog)

Żeby nie odpalać 5 terminali.

### Zadanie 6.1: `ecosystem.json`

Plik konfiguracyjny definiujący jakie procesy mają działać.

```json
[
  { "name": "lidar", "cmd": "python lidar/src/run_live.py", "cwd": "." },
  { "name": "vision", "cmd": "python warstwa_wizji/main.py", "cwd": "." },
  { "name": "audio", "cmd": "python warstwa_audio/run_watus.py", "cwd": "." },
  { "name": "consolidator", "cmd": "python consolidator/consolidator_zmq.py", "cwd": "." },
  { "name": "brain", "cmd": "python warstwa_llm/src/main.py", "cwd": "." }
]
```

### Zadanie 6.2: `run_system.py`

Skrypt w Pythonie:

* [ ]  Wczytuje `ecosystem.json`.
* [ ]  Dla każdego wpisu tworzy `subprocess.Popen`.
* [ ]  Monitoruje w pętli `poll()`.
* [ ]  Jeśli proces padł (zwrócił kod != 0 lub None), restartuje go po 5 sekundach i loguje błąd.
* [ ]  Obsługuje Ctrl+C -> zabija wszystkie procesy potomne.

## Podsumowanie Protokołu Komunikacji (ZMQ Topics)


| Temat            | Nadawca      | Odbiorca      | Treść (przykład)                                                 |
| ---------------- | ------------ | ------------- | ------------------------------------------------------------------- |
| `sensor.lidar`   | Lidar        | Consolidator  | `{ "tracks": [ { "id": 1, "pos": [2.0, 1.0] } ] }`                  |
| `sensor.vision`  | Wizja        | Consolidator  | `{ "objects": [ { "label": "person", "bbox": [...] } ] }`           |
| `world.state`    | Consolidator | Brain (LLM)   | `{ "entities": [ { "id": 1, "type": "person", "merged": true } ] }` |
| `user.voice_cmd` | Audio        | Brain (LLM)   | `{ "text": "Co widzisz?", "speaker": "Pawel" }`                     |
| `robot.action`   | Brain (LLM)  | Audio / Motor | `{ "action": "speak", "payload": "Widzę człowieka." }`            |
