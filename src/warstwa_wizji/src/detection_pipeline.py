"""
Moduł pipeline'u detekcji — przetwarzanie jednej klatki.

Odpowiada za iterację po wykrytych obiektach, analizę osób,
integrację z LiDAR, detekcję broni i wizualizację wyników.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() -> DetectionPipeline.process_frame()
"""

import math
import time
import torch
import numpy as np
from collections import defaultdict

from warstwa_wizji.src import (
    calc_brightness,
    calc_obj_angle,
    suggest_mode,
    read_lidar_tracks,
    match_camera_to_lidar,
    compute_lidar_angle,
    draw_clothes_boxes,
    draw_lidar_info,
    draw_stats_overlay,
    draw_weapon_boxes,
    draw_track_history,
)


class DetectionPipeline:
    """
    Przetwarza jedną klatkę: buduje listę detekcji, obsługuje
    lidar, broń, wizualizację i ścieżki ruchu.

    Atrybuty:
        track_history (defaultdict): Historia pozycji per track_id.
        last_detected_guns (list): Ostatni wynik detekcji broni.
        weapon_detection_enabled (bool): Flaga włączenia detekcji broni.
        mode (str): Aktualny tryb wyświetlania ('light'/'dark').
        fps_params (dict): Parametry obliczania EMA FPS.
    """

    def __init__(self):
        self.track_history = defaultdict(list)
        self.last_detected_guns: list = []
        self.weapon_detection_enabled = False
        self.mode = "light"

        self.fps_params = {
            "ema_alpha": 0.1,
            "last_stat_t": time.time(),
            "t_prev": time.time(),
            "show_fps_every": 0.5,
        }

    # ------------------------------------------------------------------

    def toggle_weapon_detection(self):
        """Przełącza detekcję broni ON/OFF."""
        self.weapon_detection_enabled = not self.weapon_detection_enabled
        print(f"Weapon Detection: {'ON' if self.weapon_detection_enabled else 'OFF'}")

    # ------------------------------------------------------------------

    def calc_fps(self) -> float:
        """Oblicza wygładzone FPS (EMA)."""
        now = time.time()
        t_prev = self.fps_params["t_prev"]
        ema_alpha = self.fps_params["ema_alpha"]
        inst_fps = 1.0 / max(1e-6, (now - t_prev))
        self.fps_params["t_prev"] = now

        ema_fps = 0.0
        ema_fps = (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps if ema_fps > 0 else inst_fps

        if (now - self.fps_params["last_stat_t"]) >= self.fps_params["show_fps_every"]:
            self.fps_params["last_stat_t"] = now
        return ema_fps

    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        dets,
        frame_idx: int,
        imgsz: int,
        fov_deg: int,
        class_names: dict,
        person_analyzer,
        model_manager,
        consolidate_with_lidar: bool = False,
        show_window: bool = True,
        verbose_window: bool = True,
        show_fps: bool = True,
        lidar_path: str = "../lidar/data/lidar.json",
    ) -> dict:
        """
        Przetwarza jedną klatkę i zwraca słownik detekcji.

        Argumenty:
            frame_bgr: Klatka BGR.
            dets: Wynik detekcji z ModelManager.detect_objects().
            frame_idx: Indeks klatki.
            imgsz: Rozmiar obrazu wejściowego modelu.
            fov_deg: Pole widzenia kamery w stopniach.
            class_names: Mapowanie ID klasy -> nazwa.
            person_analyzer: Instancja PersonAnalyzer.
            model_manager: Instancja ModelManager (do detekcji broni).
            consolidate_with_lidar: Czy łączyć dane z LiDAR.
            show_window: Czy wyświetlać okno.
            verbose_window: Czy rysować nakładki.
            show_fps: Czy obliczać FPS.
            lidar_path: Ścieżka do pliku lidar.json.

        Zwraca:
            Słownik z kluczami: objects, countOfPeople, countOfObjects,
            suggested_mode, brightness.
        """
        detections = {
            "objects": [],
            "countOfPeople": 0,
            "countOfObjects": 0,
            "suggested_mode": "",
            "brightness": 0.0,
        }

        # Jasność i tryb
        detections["brightness"] = calc_brightness(frame_bgr)
        detections["suggested_mode"] = suggest_mode(detections["brightness"], self.mode)

        # Przetwarzanie boxów
        if dets is not None and dets.boxes and dets.boxes.is_track:
            boxes = dets.boxes.xywh.cpu()
            boxes_xyxy = dets.boxes.xyxy.cpu()
            track_ids = dets.boxes.id.int().cpu().tolist()
            labels = dets.boxes.cls.int().cpu().tolist()

            # --- Lidar ---
            lidar_matches = {}
            if consolidate_with_lidar:
                lidar_matches = self._match_lidar(
                    boxes, labels, imgsz, fov_deg, lidar_path
                )

            frame_bgr_vis = dets.plot() if show_window else frame_bgr

            for i, (box, track_id, label) in enumerate(zip(boxes, track_ids, labels)):
                x, y, w, h = box
                angle = calc_obj_angle((x, y), (x + w, y + h), imgsz, fov_deg=fov_deg)
                add_info = []

                # Ścieżki ruchu
                if verbose_window:
                    self._update_track(frame_bgr_vis, track_id, (x, y))

                # --- Analiza osoby ---
                if label == 0:
                    b_x1, b_y1, b_x2, b_y2 = map(int, boxes_xyxy[i].tolist())
                    H_frame, W_frame = frame_bgr.shape[:2]
                    b_x1, b_y1 = max(0, b_x1), max(0, b_y1)
                    b_x2, b_y2 = min(W_frame, b_x2), min(H_frame, b_y2)

                    if b_x2 > b_x1 and b_y2 > b_y1:
                        person_crop = frame_bgr[b_y1:b_y2, b_x1:b_x2]

                        cache_entry = person_analyzer.analyze_person(
                            person_crop, track_id, frame_idx,
                            model_manager.clothes_detector,
                        )

                        # Rysowanie ubrań
                        if verbose_window and cache_entry.get("clothes"):
                            draw_clothes_boxes(
                                frame_bgr_vis, cache_entry["clothes"],
                                (b_x1, b_y1, b_x2, b_y2),
                            )

                        add_info = person_analyzer.build_add_info(cache_entry)

                # --- Lidar info ---
                lidar_info = self._handle_lidar_info(
                    i, track_id, x, y, lidar_matches,
                    person_analyzer.person_cache, frame_bgr_vis if verbose_window else None,
                )

                detections["objects"].append({
                    "id": track_id,
                    "type": class_names[label],
                    "left": x.item(),
                    "top": y.item(),
                    "width": w.item(),
                    "height": h.item(),
                    "isPerson": label == 0,
                    "angle": angle,
                    "additionalInfo": add_info,
                    "lidar": lidar_info,
                })
                detections["countOfObjects"] += 1
                detections["countOfPeople"] += 1 if label == 0 else 0
        else:
            frame_bgr_vis = frame_bgr

        # --- Broń ---
        if self.weapon_detection_enabled:
            self._handle_weapons(
                frame_bgr, frame_bgr_vis, frame_idx,
                model_manager.guns_detector, detections, show_window,
            )

        # --- Nakładka FPS/jasność ---
        ema_fps = self.calc_fps() if show_fps else 0
        if verbose_window:
            draw_stats_overlay(
                frame_bgr_vis, ema_fps,
                detections["brightness"], detections["suggested_mode"],
            )

        return detections

    # ======================== METODY PRYWATNE ========================

    def _update_track(self, frame_bgr, track_id: int, point: tuple):
        """Aktualizuje i rysuje ścieżkę ruchu obiektu."""
        x, y = point
        track = self.track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)
        draw_track_history(frame_bgr, track)

    def _match_lidar(self, boxes, labels, imgsz, fov_deg, lidar_path) -> dict:
        """Dopasowuje osoby z kamery do trackerów LiDAR."""
        l_tracks = read_lidar_tracks(lidar_path)
        if not l_tracks:
            return {}

        people_indices = [idx for idx, lbl in enumerate(labels) if lbl == 0]
        person_angles = []
        for idx in people_indices:
            _x, _y, _w, _h = boxes[idx]
            _ang = calc_obj_angle((_x, _y), (_x + _w, _y + _h), imgsz, fov_deg=fov_deg)
            person_angles.append((idx, _ang))

        return match_camera_to_lidar(person_angles, l_tracks)

    def _handle_lidar_info(
        self, box_idx, track_id, x, y, lidar_matches, person_cache, frame_bgr_vis
    ) -> dict:
        """Obsługuje dane LiDAR: nowe dopasowanie lub cache."""
        lidar_info = {}
        l_id = None
        l_dist = None

        if box_idx in lidar_matches:
            lm = lidar_matches[box_idx]
            l_id = lm.get("id", "?")
            l_dist = lm.get("last_position", [0, 0])[1]

            if track_id not in person_cache:
                person_cache[track_id] = {"last_frame": -9999}
            person_cache[track_id]["lidar_data"] = {"id": l_id, "dist": l_dist}

        elif track_id in person_cache and person_cache[track_id].get("lidar_data"):
            cached_l = person_cache[track_id]["lidar_data"]
            l_id = cached_l["id"]
            l_dist = cached_l["dist"]

        if l_id is not None:
            lidar_info = {"lidar_id": l_id, "distance": l_dist}
            if frame_bgr_vis is not None:
                draw_lidar_info(frame_bgr_vis, l_id, l_dist, (int(x), int(y) - 20))

        return lidar_info

    def _handle_weapons(
        self, frame_bgr, frame_bgr_vis, frame_idx, guns_detector, detections, show_window
    ):
        """Detekcja broni i dodanie wyników."""
        if frame_idx % 10 == 0:
            with torch.inference_mode():
                res_guns = guns_detector(frame_bgr, verbose=False)

            self.last_detected_guns = []
            for rg in res_guns:
                for box in rg.boxes:
                    gx1, gy1, gx2, gy2 = box.xyxy[0].cpu().numpy()
                    g_conf = float(box.conf[0].cpu().numpy())
                    g_cls_id = int(box.cls[0].cpu().numpy())
                    g_label = rg.names[g_cls_id]
                    self.last_detected_guns.append({
                        "box": [gx1, gy1, gx2, gy2],
                        "conf": g_conf,
                        "label": g_label,
                    })

        # Rysowanie + dodanie do detekcji
        if show_window:
            draw_weapon_boxes(frame_bgr_vis, self.last_detected_guns)

        for gun in self.last_detected_guns:
            gx1, gy1, gx2, gy2 = gun["box"]
            detections["objects"].append({
                "id": -1,
                "type": gun["label"],
                "left": float(gx1),
                "top": float(gy1),
                "width": float(gx2 - gx1),
                "height": float(gy2 - gy1),
                "isPerson": False,
                "angle": 0.0,
                "additionalInfo": [{"type": "weapon"}],
                "lidar": {},
            })
