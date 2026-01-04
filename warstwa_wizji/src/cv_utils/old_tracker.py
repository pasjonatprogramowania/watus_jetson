"""
Moduł trackera IoU - śledzenie obiektów metodą zachłanną.

Prosty tracker wieloobiektowy oparty na metryce IoU (Intersection over Union).
Używany do śledzenia obiektów między klatkami wideo bez użycia filtra Kalmana.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() -> IoUTracker.step()
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from ultralytics import YOLO


def iou_xyxy(a: List[float], b: List[float]) -> float:
    """
    Oblicza współczynnik IoU (Intersection over Union) dla dwóch bounding boxów.
    
    IoU to stosunek powierzchni części wspólnej do powierzchni sumy dwóch prostokątów.
    Wartość 1.0 oznacza pełne pokrycie, 0.0 oznacza brak przecięcia.
    
    Argumenty:
        a (List[float]): Pierwszy bounding box w formacie [x1, y1, x2, y2].
        b (List[float]): Drugi bounding box w formacie [x1, y1, x2, y2].
        
    Zwraca:
        float: Wartość IoU w zakresie [0.0, 1.0].
        
    Hierarchia wywołań:
        old_tracker.py -> IoUTracker._match() -> iou_xyxy()
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter + 1e-6)


def bbox_center(b: List[float]) -> Tuple[float, float]:
    """
    Oblicza współrzędne środka bounding boxa.
    
    Argumenty:
        b (List[float]): Bounding box w formacie [x1, y1, x2, y2].
        
    Zwraca:
        Tuple[float, float]: Współrzędne środka (cx, cy).
        
    Hierarchia wywołań:
        old_tracker.py -> Track.predict() -> bbox_center()
        old_tracker.py -> Track.update() -> bbox_center()
    """
    x1, y1, x2, y2 = b
    return (0.5*(x1 + x2), 0.5*(y1 + y2))


def clamp_bbox(b: List[float], w: int, h: int) -> List[float]:
    """
    Ogranicza współrzędne bounding boxa do wymiarów obrazu.
    
    Zapobiega wychodzeniu bounding boxa poza granice klatki wideo.
    
    Argumenty:
        b (List[float]): Bounding box w formacie [x1, y1, x2, y2].
        w (int): Szerokość obrazu w pikselach.
        h (int): Wysokość obrazu w pikselach.
        
    Zwraca:
        List[float]: Skorygowany bounding box mieszczący się w obrazie.
        
    Hierarchia wywołań:
        old_tracker.py -> IoUTracker.step() -> clamp_bbox()
    """
    x1, y1, x2, y2 = b
    x1 = min(max(0.0, x1), w - 1)
    x2 = min(max(0.0, x2), w - 1)
    y1 = min(max(0.0, y1), h - 1)
    y2 = min(max(0.0, y2), h - 1)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


@dataclass
class Track:
    """
    Reprezentacja pojedynczego śledzonego obiektu (toru).
    
    Przechowuje stan obiektu i jego historię ruchu dla predykcji pozycji.
    
    Atrybuty:
        track_id (int): Unikalny identyfikator toru.
        label (int): Klasa obiektu (np. 0 = osoba).
        bbox (List[float]): Bieżąca pozycja w formacie [x1, y1, x2, y2].
        score (float): Pewność detekcji (0-1).
        hits (int): Liczba pomyślnych dopasowań do detekcji.
        age (int): Liczba klatek istnienia toru.
        time_since_update (int): Liczba klatek od ostatniego dopasowania.
        vx (float): Prędkość w osi X (piksele/klatka).
        vy (float): Prędkość w osi Y (piksele/klatka).
        last_center (Tuple[float, float]): Ostatnia pozycja środka.
        
    Hierarchia wywołań:
        old_tracker.py -> IoUTracker.step() -> Track
    """
    track_id: int
    label: int
    bbox: List[float]
    score: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    vx: float = 0.0
    vy: float = 0.0
    last_center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    def predict(self):
        """
        Przewiduje pozycję obiektu w następnej klatce.
        
        Używa ekstrapolacji liniowej na podstawie poprzedniej prędkości.
        Rozmiar bounding boxa pozostaje bez zmian.
        
        Hierarchia wywołań:
            old_tracker.py -> IoUTracker.step() -> Track.predict()
        """
        cx, cy = bbox_center(self.bbox)
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        cx_pred = cx + self.vx
        cy_pred = cy + self.vy
        x1 = cx_pred - w/2
        y1 = cy_pred - h/2
        x2 = cx_pred + w/2
        y2 = cy_pred + h/2
        self.bbox = [x1, y1, x2, y2]
        self.age += 1
        self.time_since_update += 1

    def update(self, det_bbox: List[float], det_score: float):
        """
        Aktualizuje pozycję i prędkość obiektu na podstawie nowej detekcji.
        
        Oblicza nową prędkość jako różnicę środków poprzedniej i bieżącej pozycji.
        
        Argumenty:
            det_bbox (List[float]): Nowy bounding box z detekcji.
            det_score (float): Pewność nowej detekcji.
            
        Hierarchia wywołań:
            old_tracker.py -> IoUTracker.step() -> Track.update()
        """
        cx_prev, cy_prev = bbox_center(self.bbox)
        cx_new, cy_new = bbox_center(det_bbox)
        self.vx = cx_new - cx_prev
        self.vy = cy_new - cy_prev
        self.bbox = det_bbox
        self.score = det_score
        self.hits += 1
        self.time_since_update = 0
        self.last_center = (cx_new, cy_new)


class IoUTracker:
    """
    Tracker wieloobiektowy wykorzystujący metrykę IoU do dopasowywania.
    
    Prosty algorytm zachłanny sortujący pary (tor, detekcja) malejąco po IoU
    i dopasowujący niekonfliktujące pary. Tory bez dopasowania są przedłużane
    przez predykcję, a niedopasowane detekcje tworzą nowe tory.
    
    Atrybuty:
        iou_thresh (float): Minimalny próg IoU do uznania dopasowania.
        score_thresh (float): Minimalny próg pewności detekcji.
        max_age (int): Maksymalna liczba klatek bez dopasowania przed usunięciem.
        min_hits (int): Minimalna liczba dopasowań do potwierdzenia toru.
        next_id (int): Następny wolny identyfikator toru.
        tracks (List[Track]): Lista aktywnych torów.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.run() -> IoUTracker.step()
    """
    
    def __init__(
        self,
        iou_thresh: float = 0.35,
        score_thresh: float = 0.4,
        max_age: int = 10,
        min_hits: int = 6
    ):
        """
        Inicjalizuje tracker z parametrami konfiguracyjnymi.
        
        Argumenty:
            iou_thresh (float): Próg IoU do dopasowania (domyślnie 0.35).
            score_thresh (float): Próg pewności detekcji (domyślnie 0.4).
            max_age (int): Maksymalny wiek toru bez aktualizacji (domyślnie 10).
            min_hits (int): Minimalne dopasowania do potwierdzenia (domyślnie 6).
        """
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: List[Track] = []

    def _match(self, dets: List[Dict]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Dopasowuje istniejące tory do nowych detekcji metodą zachłanną.
        
        Tworzy macierz par (tor, detekcja, IoU), sortuje malejąco po IoU
        i wybiera niekonfliktujące pary z IoU >= próg i zgodną klasą.
        
        Argumenty:
            dets (List[Dict]): Lista detekcji z kluczami bbox, score, label.
            
        Zwraca:
            Tuple: (matches, unmatched_tracks, unmatched_dets) gdzie:
                   - matches: lista par (indeks_toru, indeks_detekcji)
                   - unmatched_tracks: indeksy torów bez dopasowania
                   - unmatched_dets: indeksy detekcji bez dopasowania
                   
        Hierarchia wywołań:
            old_tracker.py -> IoUTracker.step() -> IoUTracker._match()
        """
        if not self.tracks or not dets:
            return [], list(range(len(self.tracks))), list(range(len(dets)))

        # Buduj macierz par (tor_idx, det_idx, iou) dla zgodnych klas
        pairs = []
        for ti, t in enumerate(self.tracks):
            for dj, d in enumerate(dets):
                if d["score"] < self.score_thresh:
                    continue
                if d["label"] != t.label:
                    continue
                iou = iou_xyxy(t.bbox, d["bbox"])
                if iou >= self.iou_thresh:
                    pairs.append((ti, dj, iou))
        pairs.sort(key=lambda x: x[2], reverse=True)

        # Zachłanny wybór niekonfliktujących par
        used_t, used_d = set(), set()
        matches: List[Tuple[int,int]] = []
        for ti, dj, _ in pairs:
            if ti in used_t or dj in used_d:
                continue
            used_t.add(ti)
            used_d.add(dj)
            matches.append((ti, dj))

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_dets = [j for j in range(len(dets)) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def step(self, frame, dets: List[Dict]) -> List[Dict]:
        """
        Wykonuje jeden krok śledzenia - przetwarza nową klatkę.
        
        Sekwencja operacji:
        1. Predykcja pozycji wszystkich istniejących torów
        2. Dopasowanie torów do nowych detekcji
        3. Aktualizacja dopasowanych torów
        4. Tworzenie nowych torów z niedopasowanych detekcji
        5. Usuwanie martwych torów (zbyt długo bez aktualizacji)
        
        Argumenty:
            frame: Klatka wideo (używana do określenia rozmiaru).
            dets (List[Dict]): Lista detekcji z kluczami bbox, score, label.
            
        Zwraca:
            List[Dict]: Lista aktywnych torów do wizualizacji.
                        Każdy słownik zawiera: track_id, label, bbox, score, confirmed.
                        
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent.run() -> IoUTracker.step()
        """
        W, H = len(frame[0]), len(frame)

        # 1) Predykcja pozycji istniejących torów
        for t in self.tracks:
            t.predict()
            t.bbox = clamp_bbox(t.bbox, W, H)

        # 2) Dopasowanie torów do detekcji
        matches, unmatched_tracks, unmatched_dets = self._match(dets)

        # 3) Aktualizacja dopasowanych torów
        for ti, dj in matches:
            det = dets[dj]
            self.tracks[ti].update(det["bbox"], det["score"])
            self.tracks[ti].bbox = clamp_bbox(self.tracks[ti].bbox, W, H)

        # 4) Tworzenie nowych torów z niedopasowanych detekcji
        for dj in unmatched_dets:
            d = dets[dj]
            if d["score"] < self.score_thresh:
                continue
            t = Track(
                track_id=self.next_id,
                label=d["label"],
                bbox=clamp_bbox(d["bbox"], W, H),
                score=d["score"],
            )
            t.last_center = bbox_center(t.bbox)
            self.tracks.append(t)
            self.next_id += 1

        # 5) Usuwanie martwych torów
        alive: List[Track] = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive

        # 6) Przygotowanie wyniku
        outputs = []
        for t in self.tracks:
            outputs.append({
                "track_id": t.track_id,
                "label": t.label,
                "bbox": t.bbox,
                "score": t.score,
                "confirmed": t.hits >= self.min_hits
            })
        return outputs
