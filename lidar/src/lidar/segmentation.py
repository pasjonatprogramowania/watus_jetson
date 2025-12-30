from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .preprocess import BeamResult, BeamCategory
from src.config import (
    SEG_MAX_DISTANCE_JUMP_M,
    SEG_MAX_ANGLE_JUMP_DEG,
    SEG_MIN_BEAMS,
    SEG_HUMAN_MIN_R_M,
    SEG_HUMAN_MAX_R_M,
    SEG_HUMAN_MIN_LENGTH_M,
    SEG_HUMAN_MAX_LENGTH_M,
    SEG_HUMAN_MIN_BEAMS,
)



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
    
    Hierarchia wywołań:
        lidar/src/lidar/segmentation.py -> group_beams_into_segments() -> finalize_segment() -> Segment()
    """
    id: int
    beams: List[BeamResult]

    center_x: float
    center_y: float
    mean_r: float
    mean_theta: float
    length: float          # przybliżona długość obiektu [m]
    base_category: BeamCategory  # kategoria segmentu (np. OBSTACLE / HUMAN)


def convert_beam_to_local_cartesian(beam: BeamResult) -> Tuple[float, float]:
    """
    Konwertuje wiązkę ze współrzędnych biegunowych na kartezjańskie.
    
    Argumenty:
        beam (BeamResult): Obiekt wiązki z polami r i theta.
    
    Zwraca:
        Tuple[float, float]: Para (x, y) w układzie lokalnym robota.
    
    Hierarchia wywołań:
        lidar/src/lidar/segmentation.py -> group_beams_into_segments() -> convert_beam_to_local_cartesian()
    """
    # BeamResult -> (x,y) w układzie lokalnym robota
    x = beam.r * np.cos(beam.theta)
    y = beam.r * np.sin(beam.theta)
    return x, y


def group_beams_into_segments(
    beams: List[BeamResult],
    max_distance_jump: float = SEG_MAX_DISTANCE_JUMP_M,
    max_angle_jump: float = np.deg2rad(SEG_MAX_ANGLE_JUMP_DEG),
    min_beams_in_segment: int = SEG_MIN_BEAMS,
) -> List[Segment]:
    """
    Dzieli skan LiDAR na segmenty na podstawie ciągłości przestrzennej.
    
    Algorytm analizuje kolejne wiązki i tworzy nowy segment gdy:
      - skok odległości > max_distance_jump, lub
      - skok kąta > max_angle_jump, lub
      - wiązka ma kategorię NONE
    
    Argumenty:
        beams (List[BeamResult]): Lista wiązek z process_raw_scan_data().
        max_distance_jump (float): Maksymalny skok przestrzenny w metrach.
        max_angle_jump (float): Maksymalny skok kątowy w radianach.
        min_beams_in_segment (int): Minimalna liczba wiązek w segmencie.
    
    Zwraca:
        List[Segment]: Lista wykrytych segmentów.
    
    Hierarchia wywołań:
        lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> group_beams_into_segments()
    """
    # Dzieli skan (BeamResult) na segmenty na podstawie skoków odległości i kąta
    segments: List[Segment] = []
    current_beams: List[BeamResult] = []

    prev_beam: BeamResult | None = None
    prev_x: float | None = None
    prev_y: float | None = None

    def finalize_segment(seg_id: int, seg_beams: List[BeamResult]) -> Segment | None:
        # Tworzy Segment, liczy środek, długość itp.
        if len(seg_beams) < min_beams_in_segment:
            return None

        xs = []
        ys = []
        rs = []
        thetas = []

        for b in seg_beams:
            x, y = convert_beam_to_local_cartesian(b)
            xs.append(x)
            ys.append(y)
            rs.append(b.r)
            thetas.append(b.theta)

        xs = np.array(xs)
        ys = np.array(ys)
        rs = np.array(rs)
        thetas = np.array(thetas)

        center_x = float(xs.mean())
        center_y = float(ys.mean())
        mean_r = float(rs.mean())
        mean_theta = float(thetas.mean())

        # długość = przekątna prostokąta otaczającego
        length = float(
            np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
        )

        base_cat = seg_beams[0].category

        return Segment(
            id=seg_id,
            beams=seg_beams,
            center_x=center_x,
            center_y=center_y,
            mean_r=mean_r,
            mean_theta=mean_theta,
            length=length,
            base_category=base_cat,
        )

    seg_id_counter = 1

    for b in beams:
        # pomijamy wiązki NONE
        if b.category == BeamCategory.NONE:
            if current_beams:
                seg = finalize_segment(seg_id_counter, current_beams)
                if seg is not None:
                    segments.append(seg)
                    seg_id_counter += 1
                current_beams = []
                prev_beam = None
                prev_x = None
                prev_y = None
            continue

        x, y = convert_beam_to_local_cartesian(b)

        if prev_beam is None:
            # start nowego segmentu
            current_beams = [b]
            prev_beam = b
            prev_x, prev_y = x, y
            continue

        # skok w kącie i w przestrzeni między kolejnymi punktami
        angle_jump_val = abs(b.theta - prev_beam.theta)
        distance_jump_val = np.hypot(x - prev_x, y - prev_y)

        # jeśli skok za duży -> zamykamy segment, zaczynamy nowy
        if angle_jump_val > max_angle_jump or distance_jump_val > max_distance_jump:
            seg = finalize_segment(seg_id_counter, current_beams)
            if seg is not None:
                segments.append(seg)
                seg_id_counter += 1
            current_beams = [b]
        else:
            # kontynuujemy segment
            current_beams.append(b)

        prev_beam = b
        prev_x, prev_y = x, y

    # zamknięcie ostatniego segmentu
    if current_beams:
        seg = finalize_segment(seg_id_counter, current_beams)
        if seg is not None:
            segments.append(seg)

    return segments


def classify_object_type(segment: Segment) -> BeamCategory:
    """
    Klasyfikuje segment jako człowieka lub przeszkodę.
    
    Heurystyka oparta na cechach geometrycznych:
      - Odległość środka w zakresie [SEG_HUMAN_MIN_R_M, SEG_HUMAN_MAX_R_M]
      - Długość w zakresie [SEG_HUMAN_MIN_LENGTH_M, SEG_HUMAN_MAX_LENGTH_M]
      - Minimalna liczba wiązek >= SEG_HUMAN_MIN_BEAMS
    
    Argumenty:
        segment (Segment): Segment do sklasyfikowania.
    
    Zwraca:
        BeamCategory: HUMAN jeśli spełnia kryteria, OBSTACLE w przeciwnym razie.
    
    Hierarchia wywołań:
        lidar/src/lidar/segmentation.py -> categorize_all_segments() -> classify_object_type()
    """
    # heurystyka: czy segment wygląda jak człowiek

    # zbyt blisko lub zbyt daleko -> nie człowiek
    if segment.mean_r < SEG_HUMAN_MIN_R_M or segment.mean_r > SEG_HUMAN_MAX_R_M:
        return BeamCategory.OBSTACLE

    # zbyt mały albo zbyt duży segment -> nie człowiek
    if segment.length < SEG_HUMAN_MIN_LENGTH_M or segment.length > SEG_HUMAN_MAX_LENGTH_M:
        return BeamCategory.OBSTACLE

    # za mało wiązek -> raczej szum / krawędź
    if len(segment.beams) < SEG_HUMAN_MIN_BEAMS:
        return BeamCategory.OBSTACLE

    return BeamCategory.HUMAN



def categorize_all_segments(segments: List[Segment]) -> None:
    """
    Przypisuje kategorie do wszystkich segmentów i ich wiązek.
    
    Modyfikuje obiekty in-place, ustawiając base_category segmentu
    i category każdej wiązki na podstawie classify_object_type().
    
    Argumenty:
        segments (List[Segment]): Lista segmentów do sklasyfikowania.
    
    Zwraca:
        None (modyfikuje obiekty in-place)
    
    Hierarchia wywołań:
        lidar/src/lidar/system.py -> AiwataLidarSystem.process_complete_lidar_scan() -> categorize_all_segments()
    """
    # Ustawia kategorię segmentu i wszystkich jego wiązek
    for seg in segments:
        new_cat = classify_object_type(seg)
        seg.base_category = new_cat
        for b in seg.beams:
            b.category = new_cat
