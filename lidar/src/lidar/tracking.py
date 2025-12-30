import math
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import (
    TRACK_MAX_MATCH_DISTANCE_M,
    TRACK_MAX_PREDICTION_ERROR_M,
    TRACK_INIT_STEPS,
    TRACK_CONFIRM_STEPS,
    TRACK_DELETE_MISSED_READINGS,
    TRACK_DT_S,
    TRACK_VELOCITY_DECAY,
    TRACK_ALPHA,
    TRACK_BETA,
)
from .segmentation import Segment, BeamCategory

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
        
    Hierarchia wywołań:
        lidar/src/lidar/tracking.py -> HumanTracker.initialize_new_track() -> HumanTrack()
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


class HumanTracker:
    """
    Tracker wielo-obiektowy do wykrywania ludzi używający algorytmu Najbliższego Sąsiada / Węgierskiego.
    
    Filtr: Alfa-Beta (uproszczony Kalman).
    Cykl życia:
      - init: Kandydat, zbieranie dowodów.
      - confirmed: Potwierdzony obiekt, śledzony.
      - archived: Utracony tymczasowo (pamięć).
      - deleted: Usunięty z systemu.
    """

    def __init__(self):
        """
        Inicjalizuje Tracker Ludzi.
        
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.__init__() -> HumanTracker()
        """
        self.tracks: List[HumanTrack] = []
        self.archived_tracks: List[HumanTrack] = []
        
        # opcjonalny licznik ID (obecnie używamy UUID)
        self._id_counter = 0

    def predict_future_position(self, track: HumanTrack, dt: float) -> Tuple[float, float]:
        """
        Przewiduje pozycję po czasie dt na podstawie obecnej prędkości.
        
        Argumenty:
            track (HumanTrack): Ślad do przewidzenia.
            dt (float): Krok czasowy w sekundach.
            
        Zwraca:
            Tuple[float, float]: Przewidziane (x, y).
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.compute_cost_matrix() -> predict_future_position()
            lidar/src/lidar/tracking.py -> HumanTracker.predict_track_state() -> predict_future_position()
        """
        pred_x = track.x + track.vx * dt
        pred_y = track.y + track.vy * dt
        return pred_x, pred_y

    def initialize_new_track(self, segment: Segment) -> HumanTrack:
        """
        Tworzy nowy track na podstawie segmentu.
        
        Argumenty:
            segment (Segment): Segment sklasyfikowany jako HUMAN.
            
        Zwraca:
            HumanTrack: Nowy obiekt tracka w stanie 'init'.
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.update_tracker() -> initialize_new_track()
        """
        # Utwórz nowy track
        self._id_counter += 1
        now = time.time()
        new_track = HumanTrack(
            id=str(uuid.uuid4())[:8],
            x=segment.center_x,
            y=segment.center_y,
            vx=0.0,
            vy=0.0,
            state="init",
            missed_frames=0,
            matched_frames=1,
            age=0.0,
            last_update_time=now,
            history=[(now, segment.center_x, segment.center_y)]
        )
        return new_track

    def predict_track_state(self, track: HumanTrack, dt: float):
        """
        Aktualizuje stan tracka (x, y) na podstawie modelu predykcji (krok predykcji filtra Alfa-Beta).
        
        Argumenty:
            track (HumanTrack): Track do przewidzenia.
            dt (float): Delta czasu.
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.update_tracker() -> predict_track_state()
        """
        # Predykcja filtra Alfa-Beta
        track.x += track.vx * dt
        track.y += track.vy * dt
        
        # Zanik prędkości (tarcie)
        track.vx *= TRACK_VELOCITY_DECAY
        track.vy *= TRACK_VELOCITY_DECAY
        
        track.age += dt

    def update_existing_track(self, track: HumanTrack, segment: Segment, dt: float):
        """
        Krok korekcji filtra Alfa-Beta przy użyciu pomiaru (segmentu).
        
        Argumenty:
            track (HumanTrack): Track do zaktualizowania.
            segment (Segment): Dopasowany pomiar.
            dt (float): Delta czasu.
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.update_tracker() -> update_existing_track()
        """
        # Residuum (błąd predykcji)
        rx = segment.center_x - track.x
        ry = segment.center_y - track.y
        
        # Aktualizacja pozycji
        track.x += TRACK_ALPHA * rx
        track.y += TRACK_ALPHA * ry
        
        # Aktualizacja prędkości
        if dt > 0:
            track.vx += (TRACK_BETA * rx) / dt
            track.vy += (TRACK_BETA * ry) / dt

        # Aktualizacja metadanych
        track.matched_frames += 1
        track.missed_frames = 0
        track.last_update_time = time.time()
        
        # Dodanie do historii
        track.history.append((track.last_update_time, track.x, track.y))
        
        # Logika zmiany stanu
        if track.state == "init" and track.matched_frames >= TRACK_INIT_STEPS:
            track.state = "confirmed"
        elif track.state == "archived":
            track.state = "confirmed"

    def compute_cost_matrix(self, tracks: List[HumanTrack], detections: List[Segment]) -> np.ndarray:
        """
        Oblicza macierz kosztów (odległości) między trackami a detekcjami.
        
        Argumenty:
            tracks (List[HumanTrack]): Lista aktywnych tracków.
            detections (List[Segment]): Lista nowych detekcji (segmentów).
            
        Zwraca:
            np.ndarray: Macierz odległości (len(tracks) x len(detections)).
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.associate_detections_to_tracks() -> compute_cost_matrix()
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        for i, trk in enumerate(tracks):
            # Predykcja dla obecnego czasu jest już zrobiona w predict_track_state
            for j, det in enumerate(detections):
                dist = np.hypot(trk.x - det.center_x, trk.y - det.center_y)
                
                # Bramkowanie (Gating)
                if dist > TRACK_MAX_MATCH_DISTANCE_M:
                    cost_matrix[i, j] = 1000.0  # Duża liczba
                else:
                    cost_matrix[i, j] = dist
                    
        return cost_matrix

    def associate_detections_to_tracks(
        self, 
        tracks: List[HumanTrack], 
        detections: List[Segment]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Dopasowuje detekcje do tracków używając algorytmu Węgierskiego.
        
        Argumenty:
            tracks (List[HumanTrack]): Predykcje.
            detections (List[Segment]): Pomiary.
            
        Zwraca:
            matches (List[Tuple[int, int]]): Indeksy par (track_idx, det_idx).
            unmatched_tracks (List[int]): Indeksy tracków bez dopasowania.
            unmatched_detections (List[int]): Indeksy detekcji bez dopasowania.
            
        Hierarchia wywołań:
            lidar/src/lidar/tracking.py -> HumanTracker.update_tracker() -> associate_detections_to_tracks()
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
            
        cost_matrix = self.compute_cost_matrix(tracks, detections)
        
        # Algorytm węgierski (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for r, c in zip(row_ind, col_ind):
            # Sprawdź próg odległości
            if cost_matrix[r, c] > TRACK_MAX_MATCH_DISTANCE_M:
                continue
                
            matches.append((r, c))
            
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_detections:
                unmatched_detections.remove(c)
                
        return matches, unmatched_tracks, unmatched_detections

    def update_tracker(self, segments: List[Segment], dt_s: float = TRACK_DT_S):
        """
        Główna metoda aktualizacji trackera.
        
        1. Filtruje segmenty (tylko HUMAN).
        2. Przewiduje stany istniejących tracków.
        3. Dopasowuje detekcje do tracków.
        4. Aktualizuje dopasowane tracki.
        5. Tworzy nowe tracki z niedopasowanych detekcji.
        6. Obsługuje niedopasowane tracki (starzenie, usuwanie).
        
        Argumenty:
            segments (List[Segment]): Lista wszystkich segmentów z aktualnego skanu.
            dt_s (float): Delta czasu od ostatniej aktualizacji.
            
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> update_tracker()
        """
        # 1. Wybierz tylko segmenty ludzkie
        human_candidates = [s for s in segments if s.base_category == BeamCategory.HUMAN]
        
        # 2. Predykcja
        for trk in self.tracks:
            self.predict_track_state(trk, dt_s)
            
        # 3. Asocjacja
        matches, unmatched_trk_idxs, unmatched_det_idxs = self.associate_detections_to_tracks(
            self.tracks, human_candidates
        )
        
        # 4. Aktualizacja dopasowanych
        for (trk_idx, det_idx) in matches:
            track = self.tracks[trk_idx]
            track.history.append((track.last_update_time, track.x, track.y)) # ensure history
            seg = human_candidates[det_idx]
            self.update_existing_track(track, seg, dt_s)
            
        # 5. Tworzenie nowych
        for det_idx in unmatched_det_idxs:
            seg = human_candidates[det_idx]
            new_trk = self.initialize_new_track(seg)
            self.tracks.append(new_trk)
            
        # 6. Obsługa niedopasowanych
        active_tracks = []
        for trk_idx in unmatched_trk_idxs:
            track = self.tracks[trk_idx]
            track.missed_frames += 1
            track.matched_frames = 0
            
            if track.missed_frames > TRACK_DELETE_MISSED_READINGS:
                track.state = "deleted"
                # Usunięty z listy
            else:
                active_tracks.append(track)
        
        # Dodaj dopasowane tracki z powrotem do listy
        for (trk_idx, _) in matches:
            active_tracks.append(self.tracks[trk_idx])
            
        # Dodaj nowe tracki
        # (Zostały już dodane do self.tracks w kroku 5, więc musimy odbudować listę poprawnie)
        # Najczystsze podejście: odbuduj self.tracks usuwając "deleted"
        
        final_tracks = []
        for trk in self.tracks:
            if trk.state == "deleted":
                continue
            final_tracks.append(trk)
            
        self.tracks = final_tracks

    def get_confirmed_tracks(self) -> List[HumanTrack]:
        """
        Zwraca listę potwierdzonych tracków.
        
        Hierarchia wywołań:
            lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> get_confirmed_tracks()
        """
        return [t for t in self.tracks if t.state == "confirmed"]

    def get_all_tracks(self) -> List[HumanTrack]:
        """
        Zwraca wszystkie tracki (cele debugowe).
        """
        return self.tracks
