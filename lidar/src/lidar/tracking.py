from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

from src.config import (
    TRACK_MAX_MATCH_DISTANCE_M,
    TRACK_MAX_MISSED,
    TRACK_MIN_MOVING_SPEED_M_S,
    TRACK_REID_MAX_AGE_S,
    TRACK_REID_MAX_DISTANCE_M,
    TRACK_POS_ALPHA,
    TRACK_VEL_ALPHA,
    TRACK_MIN_CONFIRM_HITS,
    TRACK_MAX_MISSED_TENTATIVE,
    TRACK_RAW_VEL_MIN_SPEED_M_S,
    TRACK_PREDICTION_HORIZON_S,
    TRACK_PREDICTION_STEP_S,
    TRACK_PREDICTION_BASE_UNCERTAINTY_M,
    TRACK_PREDICTION_GROWTH_UNCERTAINTY,
    TRACK_PREDICTION_MAX_SPEED_FOR_UNCERTAINTY,
    TRACK_MOTION_ZONE_RADIUS_M,
)


@dataclass
class HumanTrack:
    """
    Reprezentuje śledzony obiekt (potencjalnie człowieka) w czasie.
    
    Przechowuje historię pozycji, wektor prędkości oraz stan (potwierdzony/niepotwierdzony).
    Zapewnia metody do predykcji przyszłej pozycji.
    
    Atrybuty:
        id (int): Unikalny identyfikator śladu.
        type (str): Typ obiektu (domyślnie "Human").
        history (List[Tuple]): Historia pozycji (czas, x, y).
        last_position (Tuple): Ostatnia znana pozycja (x, y).
        last_update_time (float): Czas ostatniej aktualizacji.
        missed_count (int): Liczba skanów bez dopasowania detekcji.
        vx (float): Prędkość w osi X [m/s].
        vy (float): Prędkość w osi Y [m/s].
        hits_count (int): Liczba udanych aktualizacji.
        confirmed (bool): Czy ślad jest stabilny i potwierdzony.
    
    Hierarchia wywołań:
        Tworzony przez: HumanTracker._create_track()
        Używany w: HumanTracker, system.py, Live_Vis_v3.py
    """
    # Track śledzonej osoby w układzie świata
    id: int
    type: str = "Human"

    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    last_position: Tuple[float, float] = (0.0, 0.0)
    last_update_time: float = 0.0

    missed_count: int = 0          # kolejne skany bez dopasowania
    vx: float = 0.0                # prędkość w osi X [m/s]
    vy: float = 0.0                # prędkość w osi Y [m/s]
    hits_count: int = 1            # ile razy track był zaktualizowany
    confirmed: bool = False        # czy track jest potwierdzony

    @property
    def x(self) -> float:
        return float(self.last_position[0])

    @property
    def y(self) -> float:
        return float(self.last_position[1])

    @property
    def speed(self) -> float:
        # moduł prędkości [m/s]
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading_rad(self) -> float:
        # kierunek ruchu [rad], 0 = oś X, CCW
        if self.speed < 1e-6:
            return 0.0
        return float(np.arctan2(self.vy, self.vx))

    @property
    def age(self) -> int:
        # liczba aktualizacji (długość historii)
        return len(self.history)

    @property
    def is_active(self) -> bool:
        # aktywne są tylko tracki potwierdzone i nieprzegapione zbyt długo
        return self.confirmed and self.missed_count <= TRACK_MAX_MISSED

    @property
    def is_moving(self) -> bool:
        # obiekt uznajemy za ruchomy dopiero od pewnej prędkości
        return self.speed >= TRACK_MIN_MOVING_SPEED_M_S

    # ---------- predykcja ----------

    def predict_position(self, t: float) -> Tuple[float, float]:
        """
        Przewiduje pozycję obiektu w zadanym czasie t.
        
        Zakłada ruch jednostajny prostoliniowy z aktualną prędkością (vx, vy).
        
        Argumenty:
            t (float): Docelowy czas predykcji.
            
        Zwraca:
            Tuple[float, float]: Przewidywana pozycja (x, y).
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker._find_best_match() -> HumanTrack.predict_position()
            tracking.py -> HumanTracker._find_best_archived_match() -> HumanTrack.predict_position()
        """
        # predykcja pozycji przy stałej prędkości w chwili t
        t = float(t)
        dt = t - float(self.last_update_time)
        if dt <= 0.0:
            return self.last_position
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        return float(px), float(py)

    def predict_future_points(
        self,
        t_start: float,
        horizon_s: float = TRACK_PREDICTION_HORIZON_S,
        step_s: float = TRACK_PREDICTION_STEP_S,
    ) -> List[Tuple[float, float, float]]:
        """
        Generuje listę punktów przyszłej trajektorii.
        
        Argumenty:
            t_start (float): Czas początkowy.
            horizon_s (float): Horyzont czasowy predykcji w sekundach.
            step_s (float): Krok czasowy w sekundach.
            
        Zwraca:
            List[Tuple[float, float, float]]: Lista punktów (t, x, y).
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.get_future_predictions() -> HumanTrack.predict_future_points()
        """
        # krótka trajektoria w przyszłość (t, x, y) przy stałej prędkości
        t_start = float(t_start)
        horizon_s = max(0.0, float(horizon_s))
        step_s = max(1e-3, float(step_s))

        if not self.is_moving or horizon_s <= 0.0:
            return []

        points: List[Tuple[float, float, float]] = []
        n_steps = int(horizon_s / step_s) + 1

        for i in range(1, n_steps + 1):
            dt = step_s * i
            t_pred = t_start + dt
            x_pred = self.x + self.vx * dt
            y_pred = self.y + self.vy * dt
            points.append((float(t_pred), float(x_pred), float(y_pred)))

        return points

    def prediction_radius(
        self,
        dt: float,
        base: float = TRACK_PREDICTION_BASE_UNCERTAINTY_M,
        growth: float = TRACK_PREDICTION_GROWTH_UNCERTAINTY,
        max_speed_for_uncertainty: float = TRACK_PREDICTION_MAX_SPEED_FOR_UNCERTAINTY,
    ) -> float:
        # prosty promień niepewności predykcji [m]
        dt = max(0.0, float(dt))
        v = min(self.speed, max_speed_for_uncertainty)
        return float(base + growth * dt * (1.0 + v))


class HumanTracker:
    """
    System wieloobiektowego śledzenia (Multi-Object Tracking).
    
    Wykorzystuje algorytm Najbliższego Sąsiada (NN) z predykcją pozycji (model stałej prędkości)
    oraz mechanizm re-identyfikacji obiektów po chwilowym zaniku (okluzji).
    Obsługuje wygładzanie pozycji (alfa-beta filter) oraz zarządzanie cyklem życia śladu
    (tworzenie, potwierdzanie, archiwizacja, usuwanie).
    
    Hierarchia wywołań:
        system.py -> AiwataLidarSystem.__init__() -> HumanTracker()
    """
    # Multi-target tracker z NN, predykcją, wygładzaniem i etapem potwierdzania

    def __init__(
        self,
        max_match_distance: float = TRACK_MAX_MATCH_DISTANCE_M,
        max_missed: float = TRACK_MAX_MISSED,
        pos_alpha: float = TRACK_POS_ALPHA,
        vel_alpha: float = TRACK_VEL_ALPHA,
        min_confirm_hits: int = TRACK_MIN_CONFIRM_HITS,
        max_missed_tentative: int = TRACK_MAX_MISSED_TENTATIVE,
    ):
        # max_match_distance    - maks. dystans dopasowania [m]
        # max_missed            - ile skanów można "zgubić" potwierdzony track
        # pos_alpha, vel_alpha  - wagi wygładzania pozycji i prędkości
        # min_confirm_hits      - ile aktualizacji zanim track będzie potwierdzony
        # max_missed_tentative  - ile skanów może przeżyć niepotwierdzony track
        self.max_match_distance = float(max_match_distance)
        self.max_missed_confirmed = int(max_missed)
        self.max_missed_tentative = int(max_missed_tentative)
        self.pos_alpha = float(pos_alpha)
        self.vel_alpha = float(vel_alpha)
        self.min_confirm_hits = int(min_confirm_hits)

        self.tracks: List[HumanTrack] = []
        self._next_id: int = 1

        # archiwum "uśpionych" tracków do re-identyfikacji
        self.archived_tracks: List[HumanTrack] = []

        # promień strefy ruchu
        self.motion_zone_radius = float(TRACK_MOTION_ZONE_RADIUS_M)

    # --- pomocnicze ---

    def _create_track(self, t: float, x: float, y: float) -> HumanTrack:
        """
        Tworzy nową instancję śledzonego obiektu (HumanTrack).
        
        Inicjalizuje historię pozycji, czas ostatniej aktualizacji oraz liczniki jakości śledzenia.
        Nowy track jest początkowo niepotwierdzony (confirmed=False).
        
        Argumenty:
            t (float): Czas utworzenia (timestamp).
            x (float): Pozycja początkowa X.
            y (float): Pozycja początkowa Y.
            
        Zwraca:
            HumanTrack: Nowy obiekt tracka.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._create_track()
        """
        # tworzy nowy, jeszcze niepotwierdzony track
        t = float(t)
        xf = float(x)
        yf = float(y)

        track = HumanTrack(
            id=self._next_id,
            type="Human",
            history=[(t, xf, yf)],
            last_position=(xf, yf),
            last_update_time=t,
            missed_count=0,
            vx=0.0,
            vy=0.0,
            hits_count=1,
            confirmed=(self.min_confirm_hits <= 1),
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _find_best_match(self, x: float, y: float, t: float) -> Optional[HumanTrack]:
        """
        Znajduje najlepiej pasujący aktywny track do zadanego punktu pomiarowego.
        
        Wykorzystuje predykcję pozycji tracków na czas 't' i oblicza odległość euklidesową.
        Zwraca track o najmniejszym dystansie, jeśli jest mniejszy niż max_match_distance.
        
        Argumenty:
            x (float): Współrzędna X punktu pomiarowego.
            y (float): Współrzędna Y punktu pomiarowego.
            t (float): Czas pomiaru.
            
        Zwraca:
            Optional[HumanTrack]: Znaleziony track lub None.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._find_best_match()
        """
        # zwraca najbliższy track względem predykowanej pozycji
        xf = float(x)
        yf = float(y)
        t = float(t)

        best_track: Optional[HumanTrack] = None
        best_dist = self.max_match_distance

        for tr in self.tracks:
            # track może być zarówno potwierdzony, jak i nie
            px, py = tr.predict_position(t)
            dx = xf - px
            dy = yf - py
            d = float(np.hypot(dx, dy))

            if d < best_dist:
                best_dist = d
                best_track = tr

        return best_track

    def _is_in_motion_zone(self, x: float, y: float) -> bool:
        """
        Sprawdza, czy punkt znajduje się w zdefiniowanej strefie ruchu.
        
        Argumenty:
            x (float): Współrzędna X.
            y (float): Współrzędna Y.
            
        Zwraca:
            bool: True jeśli punkt jest w zasięgu promienia strefy ruchu.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker (różne metody) -> HumanTracker._is_in_motion_zone()
        """
        """Zwraca True, jeśli (x,y) leży w strefie ruchu (prosty okrąg)."""
        return float(np.hypot(x, y)) <= self.motion_zone_radius

    def _update_track_state(
        self,
        track: HumanTrack,
        t: float,
        meas_x: float,
        meas_y: float,
    ) -> None:
        """
        Aktualizuje stan (pozycję, prędkość/filtr Kalmana uproszczony) wybranego tracka.
        
        Stosuje filtr alfa-beta do wygładzania pozycji i prędkości. 
        Zarządza logiką potwierdzania tracka (hits_count).
        
        Argumenty:
            track (HumanTrack): Track do zaktualizowania.
            t (float): Czas aktualizacji.
            meas_x (float): Zmierzona pozycja X.
            meas_y (float): Zmierzona pozycja Y.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._update_track_state()
        """
        # aktualizacja pozycji i prędkości tracka
        t = float(t)
        mx = float(meas_x)
        my = float(meas_y)

        dt = t - float(track.last_update_time)
        if dt <= 1e-6:
            track.last_position = (mx, my)
            track.last_update_time = t
            track.history.append((t, mx, my))
            track.missed_count = 0
            track.hits_count += 1
            if not track.confirmed and track.hits_count >= self.min_confirm_hits:
                track.confirmed = True
            return

        raw_vx = (mx - track.x) / dt
        raw_vy = (my - track.y) / dt

        # minimalna prędkość, żeby tłumić drobny szum
        raw_speed = float(np.hypot(raw_vx, raw_vy))
        if raw_speed < TRACK_RAW_VEL_MIN_SPEED_M_S:
            raw_vx = 0.0
            raw_vy = 0.0

        track.vx = (1.0 - self.vel_alpha) * track.vx + self.vel_alpha * raw_vx
        track.vy = (1.0 - self.vel_alpha) * track.vy + self.vel_alpha * raw_vy

        new_x = (1.0 - self.pos_alpha) * track.x + self.pos_alpha * mx
        new_y = (1.0 - self.pos_alpha) * track.y + self.pos_alpha * my

        track.last_position = (float(new_x), float(new_y))
        track.last_update_time = t
        track.history.append((t, float(new_x), float(new_y)))
        track.missed_count = 0
        track.hits_count += 1

        if not track.confirmed and track.hits_count >= self.min_confirm_hits:
            track.confirmed = True

    def _archive_track(self, track: HumanTrack) -> None:
        """
        Przenosi track do archiwum zamiast go usuwać.
        
        Umożliwia późniejszą re-identyfikację, jeśli obiekt pojawi się ponownie.
        
        Argumenty:
            track (HumanTrack): Track do zarchiwizowania.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._archive_track()
        """
        # przenosi track do archiwum zamiast całkowitego usunięcia
        self.archived_tracks.append(track)

    def _prune_archive(self, t: float) -> None:
        """
        Usuwa z archiwum tracki, które przekroczyły maksymalny czas życia w archiwum.
        
        Argumenty:
            t (float): Aktualny czas.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._prune_archive()
        """
        # usuwa z archiwum tracki, które są za stare
        t = float(t)
        fresh: List[HumanTrack] = []
        for tr in self.archived_tracks:
            age_s = t - float(tr.last_update_time)
            if age_s <= TRACK_REID_MAX_AGE_S:
                fresh.append(tr)
        self.archived_tracks = fresh

    def _find_best_archived_match(
            self,
            x: float,
            y: float,
            t: float,
    ) -> Optional[HumanTrack]:
        """
        Próbuje dopasować punkt pomiarowy do tracków znajdujących się w archiwum (Re-ID).
        
        Sprawdza zarówno pozycję predykowaną jak i ostatnią znaną pozycję.
        
        Argumenty:
            x (float): Zmierzona pozycja X.
            y (float): Zmierzona pozycja Y.
            t (float): Czas pomiaru.
            
        Zwraca:
            Optional[HumanTrack]: Dopasowany archiwalny track lub None.
            
        Hierarchia wywołań:
            tracking.py -> HumanTracker.update() -> HumanTracker._find_best_archived_match()
        """
        xf = float(x)
        yf = float(y)
        t = float(t)

        best_tr: Optional[HumanTrack] = None
        best_dist = TRACK_REID_MAX_DISTANCE_M

        for tr in self.archived_tracks:
            # pozycja predykowana w chwili t
            px, py = tr.predict_position(t)
            d_pred = float(np.hypot(xf - px, yf - py))

            # ostatnia znana pozycja z historii
            _, lx, ly = tr.history[-1]
            d_last = float(np.hypot(xf - lx, yf - ly))

            # bierzemy lepszą z dwóch odległości
            d = min(d_pred, d_last)

            if d < best_dist:
                best_dist = d
                best_tr = tr

        return best_tr

    # --- API trackera ---

    def update(self, detections: List[Tuple[float, float]], t: float) -> List[HumanTrack]:
        """
        Aktualizuje stan trackera na podstawie nowych detekcji.
        
        Główna pętla algorytmu śledzenia:
        1. Czyszczenie archiwum ze starych śladów.
        2. Dopasowanie detekcji do aktywnych śladów (Najbliższy Sąsiad).
        3. Dopasowanie pozostałych detekcji do śladów w archiwum (re-id).
        4. Utworzenie nowych śladów dla niedopasowanych detekcji.
        5. Aktualizacja liczników (missed/hits) i zarządzanie cyklem życia (archiwizacja).
        
        Argumenty:
            detections (List[Tuple[float, float]]): Lista wykrytych pozycji (x, y).
            t (float): Czas aktualnego skanu.
            
        Zwraca:
            List[HumanTrack]: Lista aktualnie aktywnych śladów.
            
        Hierarchia wywołań:
            system.py -> AiwataLidarSystem.process_scan() -> HumanTracker.update()
        """
        t = float(t)
        updated_tracks: set[int] = set()

        # porządkujemy archiwum (wyrzucamy bardzo stare tracki)
        self._prune_archive(t)

        # 1. dopasowanie punktów do istniejących tracków
        for (x, y) in detections:
            xf = float(x)
            yf = float(y)

            track = self._find_best_match(xf, yf, t)
            if track is not None:
                # dopasowaliśmy do aktywnego tracka
                self._update_track_state(track, t, xf, yf)
                updated_tracks.add(track.id)
                continue

            # 2. jeśli nie znaleziono aktywnego tracka, próbujemy z archiwum
            archived = self._find_best_archived_match(xf, yf, t)
            if archived is not None:
                # reaktywujemy stary track (odzyskujemy ID)
                self.archived_tracks.remove(archived)
                archived.missed_count = 0
                if archived not in self.tracks:
                    self.tracks.append(archived)

                self._update_track_state(archived, t, xf, yf)
                updated_tracks.add(archived.id)
                continue

            # 3. nie pasuje do niczego -> tworzymy nowy track
            new_track = self._create_track(t, xf, yf)
            updated_tracks.add(new_track.id)

        # 4. zwiększamy missed_count dla tracków, które nic nie dostały
        still_active: List[HumanTrack] = []
        for tr in self.tracks:
            if tr.id not in updated_tracks:
                tr.missed_count += 1

            # track trafia do archiwum TYLKO jeśli jest długo niewidoczny
            if tr.missed_count > self.max_missed_confirmed:
                self._archive_track(tr)
            else:
                still_active.append(tr)

        self.tracks = still_active


        self.tracks = still_active

        return self.tracks

    def get_future_predictions(
        self,
        t_now: float,
        horizon_s: float = TRACK_PREDICTION_HORIZON_S,
        step_s: float = TRACK_PREDICTION_STEP_S,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Zwraca przewidywane trajektorie dla wszystkich aktywnych tracków.
        
        Dla każdego aktywnego tracka generuje listę punktów w przyszłości (czas, x, y).
        
        Argumenty:
            t_now (float): Aktualny czas.
            horizon_s (float): Horyzont czasowy predykcji.
            step_s (float): Krok czasowy predykcji.
            
        Zwraca:
            Dict[int, List[Tuple]]: Słownik {id_tracka: lista_punktów_trajektorii}.
            
        Hierarchia wywołań:
            run_vis.py -> main() -> AiwataLidarSystem.get_prediction_data() -> HumanTracker.get_future_predictions()
        """
        # zwraca przewidywane trajektorie (t,x,y) dla aktywnych tracków
        t_now = float(t_now)
        preds: Dict[int, List[Tuple[float, float, float]]] = {}
        for tr in self.tracks:
            if not tr.is_active:
                continue
            traj = tr.predict_future_points(
                t_start=t_now,
                horizon_s=horizon_s,
                step_s=step_s,
            )
            preds[tr.id] = traj
        return preds
