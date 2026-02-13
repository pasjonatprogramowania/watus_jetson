"""
Moduł CVAgent - główny agent wizyjny.

Obsługuje detekcję obiektów, śledzenie, klasyfikację atrybutów
oraz integrację z danymi Lidara.

Hierarchia wywołań:
    main.py -> CVAgent() -> run()
"""

import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from torch.amp import autocast

from .cv_utils import (
    calc_obj_angle,
    calc_brightness,
    suggest_mode,
    read_lidar_tracks,
    match_camera_to_lidar,
    process_clothes_detection,
    process_weapon_detection,
    should_update_cache,
    draw_clothes_boxes,
    draw_lidar_info,
    draw_stats_overlay,
    draw_weapon_boxes,
)

# Konfiguracja
GPU_ENABLED = True
ESCAPE_BUTTON = "q"


def _get_classifiers():
    """Ładuje klasyfikatory jeśli GPU jest włączone."""
    if GPU_ENABLED:
        from .img_classifiers.image_classifier import getClassifiers, getClothesClassifiers
        return getClassifiers(), getClothesClassifiers()
    return (None, None, None), (None, None, None)


class CVAgent:
    """
    Główny agent wizyjny (Computer Vision).
    
    Obsługuje pobieranie obrazu z kamery/strumienia, detekcję obiektów (YOLO/RT-DETR),
    śledzenie, klasyfikację atrybutów (płeć, wiek, emocje, ubrania) oraz integrację z Lidarem.
    
    Atrybuty:
        cap (cv2.VideoCapture): Obiekt przechwytywania wideo.
        detector (YOLO): Główny model detekcji osób.
        clothes_detector (YOLO): Model detekcji ubrań.
        guns_detector (YOLO): Model detekcji broni.
    """

    def __init__(
            self,
            weights_path: str = "yolo12s.pt",
            imgsz: int = 640,
            source: int | str = 0,
            cap=None,
            json_save_func=None,
            use_net_stream: bool = True
    ):
        """
        Inicjalizuje CVAgent.
        
        Argumenty:
            weights_path (str): Ścieżka do wag modelu YOLO.
            imgsz (int): Rozmiar obrazu dla modelu.
            source (int|str): Źródło wideo (indeks kamery lub ścieżka).
            cap: Opcjonalny obiekt VideoCapture.
            json_save_func: Funkcja do zapisu JSON.
            use_net_stream (bool): Czy używać strumienia sieciowego.
        """
        self.save_to_json = json_save_func
        self.imgsz = imgsz
        self.track_history = defaultdict(lambda: [])

        cv2.setUseOptimized(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Ładowanie klasyfikatorów
        classifiers, clothes_classifiers = _get_classifiers()
        if GPU_ENABLED:
            self.emotion_classifier, self.gender_classifier, self.age_classifier = classifiers
            self.clothes_type_clf, self.clothes_pattern_clf, self.color_clf = clothes_classifiers
        else:
            self.color_clf = None

        self.person_cache = {}

        self._init_video_capture(cap, source, use_net_stream)

        self.fps_params = {
            "ema_alpha": 0.1,
            "last_stat_t": time.time(),
            "t_prev": 0,
            "show_fps_every": 0.5
        }
        self.frame_idx = 0
        self.mil_vehicles_details = {}
        self.clothes_details = {}

        self._load_models(weights_path)

        self.window_name = f"YOLOv12 – naciśnij '{ESCAPE_BUTTON}' aby wyjść"
        self.lidar_path = "../lidar/data/lidar.json"
        self.last_detected_guns = []

    def _init_video_capture(self, cap, source, use_net_stream):
        """Inicjalizuje przechwytywanie wideo."""
        if cap is None:
            if use_net_stream:
                gst_pipeline = (
                    "udpsrc port=5000 ! "
                    "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! "
                    "appsink drop=1"
                )
                print(f"Opening network stream with pipeline:\n{gst_pipeline}")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                print("Nie mogę otworzyć kamery")
                return
        else:
            self.cap = cap

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_recorder = None

    def _load_models(self, weights_path):
        """Ładuje modele YOLO."""
        self.detector = YOLO(weights_path)
        self.class_names = self.detector.names

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clothes_path = os.path.join(base_dir, 'models/clothes.pt')
        guns_path = os.path.join(base_dir, 'models/guns.pt')
        self.clothes_detector = YOLO(clothes_path)
        self.guns_detector = YOLO(guns_path)

    def init_recorder(self, out_path):
        """Inicjalizuje nagrywanie wideo do pliku."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        fps_output = float(fps_cap) if fps_cap and fps_cap > 1.0 else 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_recorder = cv2.VideoWriter(out_path, fourcc, fps_output, (width, height))
        return self.video_recorder is not None

    def init_window(self):
        """Tworzy okno podglądu OpenCV."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _actualize_tracks(self, frame_bgr, track_id, point: tuple[int, int]):
        """Rysuje historię (ścieżkę) poruszania się obiektu."""
        x, y = point
        track = self.track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_bgr, [points], False, color=(230, 230, 230), thickness=10)

    def _calc_fps(self):
        """Oblicza FPS wygładzone wykładniczo (EMA)."""
        ema_alpha, last_stat_t, t_prev, show_fps_every = self.fps_params.values()
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - t_prev))
        self.fps_params["t_prev"] = now
        ema_fps = 0.0
        ema_alpha = 0.1
        ema_fps = (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps if ema_fps > 0 else inst_fps

        if (now - last_stat_t) >= show_fps_every:
            self.fps_params["last_stat_t"] = now
        return ema_fps

    def _warm_up_model(self):
        """Wykonuje puste wnioskowanie aby rozgrzać GPU/model."""
        _ret, _warm = self.cap.read()
        if _ret:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with autocast(dtype=torch.float32, device_type=self.device.type):
                        _ = self.detector(_warm)
                else:
                    _ = self.detector(_warm)

    def _detect_objects(self, frame_bgr, imgsz: int = 640, run_detection=True):
        """Uruchamia detekcję osób na klatce."""
        if run_detection:
            iou = 0.7
            conf = 0.3
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with autocast(dtype=torch.float32, device_type=self.device.type):
                        detections = self.detector.track(
                            frame_bgr, persist=True,
                            device=self.device, verbose=False,
                            imgsz=imgsz, iou=iou, conf=conf
                        )
                else:
                    detections = self.detector.track(
                        frame_bgr, persist=True,
                        device=self.device, verbose=False,
                        imgsz=imgsz, iou=iou, conf=conf
                    )
        return detections[0]

    def _process_person(self, frame_bgr, track_id, boxes_xyxy, i, current_frame_idx):
        """Przetwarza wykrytą osobę - ubrania, klasyfikatory."""
        b_x1, b_y1, b_x2, b_y2 = map(int, boxes_xyxy[i].tolist())
        H_frame, W_frame = frame_bgr.shape[:2]
        b_x1, b_y1 = max(0, b_x1), max(0, b_y1)
        b_x2, b_y2 = min(W_frame, b_x2), min(H_frame, b_y2)

        if b_x2 <= b_x1 or b_y2 <= b_y1:
            return self.person_cache.get(track_id, {})

        person_crop = frame_bgr[b_y1:b_y2, b_x1:b_x2]
        cache_entry = self.person_cache.get(track_id, {"last_frame": -9999})

        if should_update_cache(self.person_cache, track_id, current_frame_idx, cache_ttl=100):
            clothes_data = process_clothes_detection(
                person_crop,
                self.clothes_detector,
                color_clf=self.color_clf if GPU_ENABLED else None,
                gpu_enabled=GPU_ENABLED
            )

            classifiers_result = {}
            try:
                pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                if GPU_ENABLED and False:  # Wyłączone
                    res_emo = self.emotion_classifier.process(pil_img)
                    classifiers_result["emotion"] = max(res_emo, key=res_emo.get)
                    res_gen = self.gender_classifier.process(pil_img)
                    classifiers_result["gender"] = max(res_gen, key=res_gen.get)
                    res_age = self.age_classifier.process(pil_img)
                    classifiers_result["age"] = max(res_age, key=res_age.get)
            except Exception as e:
                print(f"Classifier error: {e}")

            new_cache = {
                "last_frame": current_frame_idx,
                "clothes": clothes_data,
                "emotion": classifiers_result.get("emotion") or cache_entry.get("emotion"),
                "gender": classifiers_result.get("gender") or cache_entry.get("gender"),
                "age": classifiers_result.get("age") or cache_entry.get("age"),
                "lidar_data": cache_entry.get("lidar_data")
            }
            self.person_cache[track_id] = new_cache
            cache_entry = new_cache

        return cache_entry, (b_x1, b_y1, b_x2, b_y2)

    def _process_lidar_matching(self, boxes, labels, fov_deg):
        """Dopasowuje obiekty kamery do tracków Lidara."""
        l_tracks = read_lidar_tracks(self.lidar_path)
        people_indices = [idx for idx, lbl in enumerate(labels) if lbl == 0]
        person_angles = []
        for idx in people_indices:
            _x, _y, _w, _h = boxes[idx]
            _ang = calc_obj_angle((_x, _y), (_x + _w, _y + _h), self.imgsz, fov_deg=fov_deg)
            person_angles.append((idx, _ang))
        return match_camera_to_lidar(person_angles, l_tracks)

    def _handle_lidar_info(self, i, track_id, lidar_matches, x, y, frame_bgr, show_window):
        """Obsługuje informacje Lidara dla obiektu."""
        lidar_info = {}
        l_id = None
        l_dist = None

        if i in lidar_matches:
            lm = lidar_matches[i]
            l_id = lm.get("id", "?")
            l_dist = lm.get("last_position", [0, 0])[1]
            if track_id not in self.person_cache:
                self.person_cache[track_id] = {"last_frame": -9999}
            self.person_cache[track_id]["lidar_data"] = {"id": l_id, "dist": l_dist}
        elif track_id in self.person_cache and self.person_cache[track_id].get("lidar_data"):
            cached_l = self.person_cache[track_id]["lidar_data"]
            l_id = cached_l["id"]
            l_dist = cached_l["dist"]

        if l_id is not None:
            lidar_info = {"lidar_id": l_id, "distance": l_dist}
            if show_window:
                draw_lidar_info(frame_bgr, l_id, l_dist, (int(x), int(y) - 20))

        return lidar_info

    def _handle_weapon_detection(self, frame_bgr, detections, show_window):
        """Obsługuje detekcję broni."""
        guns, detected = process_weapon_detection(
            frame_bgr, self.guns_detector, self.frame_idx
        )
        if detected:
            self.last_detected_guns = guns

        if show_window:
            draw_weapon_boxes(frame_bgr, self.last_detected_guns)

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
                "lidar": {}
            })

    def run(
        self,
        save_video: bool = False,
        out_path: str = "output.mp4",
        show_window=True,
        det_stride: int = 1,
        show_fps: bool = True,
        verbose: bool = True,
        verbose_window: bool = True,
        fov_deg: int = 102,
        consolidate_with_lidar: bool = False,
    ):
        """
        Główna pętla przetwarzania wideo.
        
        Argumenty:
            save_video (bool): Czy zapisywać wideo.
            out_path (str): Ścieżka pliku wyjściowego wideo.
            show_window (bool): Czy pokazywać okno podglądu.
            det_stride (int): Co ile klatek uruchamiać detekcję.
            show_fps (bool): Czy pokazywać FPS.
            verbose (bool): Czy logować na konsolę.
            verbose_window (bool): Czy rysować szczegóły w oknie.
            fov_deg (int): Kąt widzenia kamery w stopniach.
            consolidate_with_lidar (bool): Czy łączyć dane z Lidarem.
        """
        save_video = self.init_recorder(out_path) if save_video else None
        if show_window is False or show_window is None:
            verbose_window = False

        if show_window:
            self.init_window()

        detections = {
            "objects": [],
            "countOfPeople": 0,
            "countOfObjects": 0,
            "suggested_mode": '',
            "brightness": 0.0,
        }
        self.frame_idx = 0
        mode = 'light'
        weapon_detection_enabled = False

        self._warm_up_model()

        try:
            self.fps_params["t_prev"] = time.time()

            while True:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    print("Koniec strumienia")
                    break

                detections["countOfPeople"] = 0
                detections["countOfObjects"] = 0
                detections["objects"] = []

                run_detection = (self.frame_idx % det_stride == 0)
                dets = self._detect_objects(frame_bgr, run_detection=run_detection)

                detections["brightness"] = calc_brightness(frame_bgr)
                detections["suggested_mode"] = suggest_mode(detections["brightness"], mode)

                if dets.boxes and dets.boxes.is_track:
                    boxes = dets.boxes.xywh.cpu()
                    boxes_xyxy = dets.boxes.xyxy.cpu()
                    track_ids = dets.boxes.id.int().cpu().tolist()
                    labels = dets.boxes.cls.int().cpu().tolist()

                    lidar_matches = {}
                    if consolidate_with_lidar:
                        lidar_matches = self._process_lidar_matching(boxes, labels, fov_deg)

                    if show_window:
                        frame_bgr = dets.plot()

                    for i, (box, track_id, label) in enumerate(zip(boxes, track_ids, labels)):
                        x, y, w, h = box
                        angle = calc_obj_angle((x, y), (x + w, y + h), self.imgsz, fov_deg=fov_deg)
                        add_info = []

                        if verbose_window:
                            self._actualize_tracks(frame_bgr, track_id, (x, y))

                        if label == 0:
                            result = self._process_person(frame_bgr, track_id, boxes_xyxy, i, self.frame_idx)
                            if result:
                                cache_entry, person_box = result

                                if show_window and "clothes" in cache_entry:
                                    draw_clothes_boxes(frame_bgr, cache_entry["clothes"], person_box)

                                if cache_entry.get("gender"):
                                    add_info.append({"gender": cache_entry["gender"]})
                                if cache_entry.get("age"):
                                    add_info.append({"age": cache_entry["age"]})
                                if cache_entry.get("emotion"):
                                    add_info.append({"emotion": cache_entry["emotion"]})
                                if cache_entry.get("clothes"):
                                    add_info.append({"clothes": cache_entry["clothes"]})

                        lidar_info = self._handle_lidar_info(
                            i, track_id, lidar_matches, x, y, frame_bgr, show_window
                        )

                        detections["objects"].append({
                            "id": track_id,
                            "type": self.class_names[label],
                            "left": x.item(),
                            "top": y.item(),
                            "width": w.item(),
                            "height": h.item(),
                            "isPerson": label == 0,
                            "angle": angle,
                            "additionalInfo": add_info,
                            "lidar": lidar_info
                        })
                        detections["countOfObjects"] += 1
                        detections["countOfPeople"] += (1 if label == 0 else 0)

                if weapon_detection_enabled:
                    self._handle_weapon_detection(frame_bgr, detections, show_window)

                ema_fps = self._calc_fps() if show_fps else 0
                if verbose_window:
                    draw_stats_overlay(frame_bgr, ema_fps, detections["brightness"], detections["suggested_mode"])

                if self.save_to_json is not None:
                    self.save_to_json("camera.jsonl", detections)

                if show_window:
                    cv2.imshow(self.window_name, frame_bgr)

                if save_video:
                    self.video_recorder.write(frame_bgr)

                self.frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord(ESCAPE_BUTTON):
                    break
                elif key == ord('w'):
                    weapon_detection_enabled = not weapon_detection_enabled
                    print(f"Weapon Detection: {'ON' if weapon_detection_enabled else 'OFF'}")

        except KeyboardInterrupt:
            print("Przerwano przez użytkownika.")
        finally:
            self.cap.release()
            if self.video_recorder is not None:
                self.video_recorder.release()
            if show_window:
                cv2.destroyAllWindows()
