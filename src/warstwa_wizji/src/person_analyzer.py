"""
Moduł analizy atrybutów osób.

Odpowiada za detekcję ubrań, klasyfikację koloru, emocji, płci i wieku
na wycinkach obrazu zawierających osoby. Zarządza cacheem wyników.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run() -> PersonAnalyzer.analyze()
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Any

GPU_ENABLED = True

if GPU_ENABLED:
    from warstwa_wizji.src.img_classifiers.image_classifier import (
        getClassifiers,
        getClothesClassifiers,
    )

# Interwały odświeżania poszczególnych grup cech (w klatkach).
# Różne wartości rozsuwają obliczenia w czasie, zapobiegając skokom GPU.
CLOTHES_DETECT_INTERVAL = 1000    # detekcja boxów ubrań
CLOTHES_COLOR_INTERVAL = 500     # klasyfikacja koloru ubrań
CLOTHES_TYPE_INTERVAL = 700      # klasyfikacja typu ubrań
CLOTHES_PATTERN_INTERVAL = 850   # klasyfikacja wzoru ubrań
EMOTION_UPDATE_INTERVAL = 2000   # klasyfikacja emocji



class PersonAnalyzer:
    """
    Analizuje atrybuty osób: ubrania, kolor, emocje, płeć, wiek.

    Zarządza cacheem wyników, aby nie powtarzać kosztownych operacji
    klasyfikacji w każdej klatce. Poszczególne klasyfikatory uruchamiane
    są w osobnych oknach czasowych, aby rozłożyć obciążenie GPU.

    Atrybuty:
        person_cache (dict): Cache atrybutów osób (track_id -> dane).
        emotion_classifier: Klasyfikator emocji.
        gender_classifier: Klasyfikator płci.
        age_classifier: Klasyfikator wieku.
        clothes_type_clf: Klasyfikator typu ubrań.
        clothes_pattern_clf: Klasyfikator wzoru ubrań.
        color_clf: Klasyfikator koloru.
    """

    def __init__(self):
        self.person_cache: dict = {}

        if GPU_ENABLED:
            self.emotion_classifier, self.gender_classifier, self.age_classifier = (
                getClassifiers()
            )
            self.clothes_type_clf, self.clothes_pattern_clf, self.color_clf = (
                getClothesClassifiers()
            )

    # ------------------------------------------------------------------

    def analyze_person(
        self,
        person_crop_bgr: np.ndarray,
        track_id: int,
        frame_idx: int,
        clothes_detector: Any,
    ) -> dict:
        """
        Analizuje ubrania i atrybuty osoby. Korzysta z cache.

        Płeć i wiek są klasyfikowane jednorazowo (gdy brak w cache).
        Emocje, detekcja ubrań, kolor, typ i wzór ubrań odświeżane
        cyklicznie w osobnych oknach czasowych, aby rozłożyć obciążenie GPU.

        Argumenty:
            person_crop_bgr: Wycinek klatki z osobą (BGR).
            track_id: ID trackera osoby.
            frame_idx: Aktualny indeks klatki.
            clothes_detector: Model YOLO do detekcji ubrań.

        Zwraca:
            Słownik cache_entry z kluczami:
            clothes, emotion, gender, age, lidar_data,
            last_clothes_detect_frame, last_color_frame,
            last_type_frame, last_pattern_frame, last_emotion_frame.
        """
        cache_entry = self.person_cache.get(track_id)

        if cache_entry is None:
            cache_entry = {
                "gender": None,
                "age": None,
                "emotion": None,
                "clothes": [],
                "lidar_data": None,
                "last_clothes_detect_frame": -9999,
                "last_color_frame": -9999,
                "last_type_frame": -9999,
                "last_pattern_frame": -9999,
                "last_emotion_frame": -9999,
            }

        # --- Płeć / wiek: jednorazowo (nie zmieniają się) ---
        if cache_entry["gender"] is None or cache_entry["age"] is None:
            self._classify_demographics(cache_entry, person_crop_bgr)

        # --- Detekcja ubrań (boxy): cyklicznie ---
        if (frame_idx - cache_entry["last_clothes_detect_frame"]) > CLOTHES_DETECT_INTERVAL:
            current_h, current_w = person_crop_bgr.shape[:2]
            cache_entry["clothes"] = self._detect_clothes(
                person_crop_bgr, clothes_detector, current_w, current_h,
            )
            cache_entry["last_clothes_detect_frame"] = frame_idx

        # --- Kolor ubrań: cyklicznie (osobne okno) ---
        if (frame_idx - cache_entry["last_color_frame"]) > CLOTHES_COLOR_INTERVAL:
            self._classify_clothes_color(cache_entry, person_crop_bgr)
            cache_entry["last_color_frame"] = frame_idx

        # --- Typ ubrań: cyklicznie (osobne okno) ---
        if (frame_idx - cache_entry["last_type_frame"]) > CLOTHES_TYPE_INTERVAL:
            self._classify_clothes_type(cache_entry, person_crop_bgr)
            cache_entry["last_type_frame"] = frame_idx

        # --- Wzór ubrań: cyklicznie (osobne okno) ---
        if (frame_idx - cache_entry["last_pattern_frame"]) > CLOTHES_PATTERN_INTERVAL:
            self._classify_clothes_pattern(cache_entry, person_crop_bgr)
            cache_entry["last_pattern_frame"] = frame_idx

        # --- Emocje: cyklicznie ---
        if (frame_idx - cache_entry["last_emotion_frame"]) > EMOTION_UPDATE_INTERVAL:
            self._classify_emotion(cache_entry, person_crop_bgr)
            cache_entry["last_emotion_frame"] = frame_idx

        self.person_cache[track_id] = cache_entry
        return cache_entry

    # ------------------------------------------------------------------

    def _classify_demographics(self, cache_entry: dict, crop_bgr: np.ndarray):
        """Klasyfikuje płeć i wiek — wywoływane jednorazowo per osoba."""
        if not GPU_ENABLED:
            return
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            res_gen = self.gender_classifier.process(pil_img)
            cache_entry["gender"] = max(res_gen, key=res_gen.get)
            res_age = self.age_classifier.process(pil_img)
            cache_entry["age"] = max(res_age, key=res_age.get)
        except Exception as e:
            print(f"Demographics classifier error: {e}")

    # ------------------------------------------------------------------

    def _classify_emotion(self, cache_entry: dict, crop_bgr: np.ndarray):
        """Klasyfikuje emocje — wywoływane cyklicznie."""
        if not GPU_ENABLED:
            return
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            res_emo = self.emotion_classifier.process(pil_img)
            cache_entry["emotion"] = max(res_emo, key=res_emo.get)
        except Exception as e:
            print(f"Emotion classifier error: {e}")

    # ------------------------------------------------------------------

    def _detect_clothes(
        self,
        person_crop_bgr: np.ndarray,
        clothes_detector: Any,
        crop_w: int,
        crop_h: int,
    ) -> list:
        """Wykrywa boxy ubrań na wycinku osoby (bez klasyfikacji atrybutów)."""
        results_clothes = clothes_detector.predict(person_crop_bgr, verbose=False)
        clothes_data = []

        for rc in results_clothes:
            c_boxes = rc.boxes.xyxy.cpu().numpy()
            c_clss = rc.boxes.cls.cpu().numpy()
            c_conf = rc.boxes.conf.cpu().numpy()
            c_names = rc.names

            for cb, cc, ccnf in zip(c_boxes, c_clss, c_conf):
                cx1, cy1, cx2, cy2 = cb
                norm_box = [
                    float(cx1) / crop_w,
                    float(cy1) / crop_h,
                    float(cx2) / crop_w,
                    float(cy2) / crop_h,
                ]
                clothes_data.append({
                    "box_norm": norm_box,
                    "label": c_names[int(cc)],
                    "class_num": int(cc),
                    "conf": float(ccnf),
                    "details": {},
                })

        return clothes_data

    # ------------------------------------------------------------------

    def _get_clothes_crops(
        self, cache_entry: dict, person_crop_bgr: np.ndarray
    ) -> list[tuple[dict, Image.Image]]:
        """Zwraca listę (item, pil_crop) dla ubrań typu clothing/shoe/bag."""
        crop_h, crop_w = person_crop_bgr.shape[:2]
        results = []
        for item in cache_entry.get("clothes", []):
            if item["label"] not in ("clothing", "shoe", "bag"):
                continue
            bx = item["box_norm"]
            x1 = max(0, int(bx[0] * crop_w))
            y1 = max(0, int(bx[1] * crop_h))
            x2 = min(crop_w, int(bx[2] * crop_w))
            y2 = min(crop_h, int(bx[3] * crop_h))
            if x2 > x1 and y2 > y1:
                crop = person_crop_bgr[y1:y2, x1:x2]
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                results.append((item, pil))
        return results

    # ------------------------------------------------------------------

    def _classify_clothes_color(self, cache_entry: dict, person_crop_bgr: np.ndarray):
        """Klasyfikuje dominujący kolor każdego elementu ubrania."""
        if not GPU_ENABLED:
            return
        try:
            for item, pil_img in self._get_clothes_crops(cache_entry, person_crop_bgr):
                dom_color = self.color_clf(np.asarray(pil_img))
                item["details"]["color"] = [int(x) for x in dom_color]
        except Exception as e:
            print(f"Clothes color error: {e}")

    # ------------------------------------------------------------------

    def _classify_clothes_type(self, cache_entry: dict, person_crop_bgr: np.ndarray):
        """Klasyfikuje typ każdego elementu ubrania."""
        if not GPU_ENABLED:
            return
        try:
            for item, pil_img in self._get_clothes_crops(cache_entry, person_crop_bgr):
                res = self.clothes_type_clf.process(pil_img)
                item["details"]["type"] = max(res, key=res.get)
        except Exception as e:
            print(f"Clothes type error: {e}")

    # ------------------------------------------------------------------

    def _classify_clothes_pattern(self, cache_entry: dict, person_crop_bgr: np.ndarray):
        """Klasyfikuje wzór każdego elementu ubrania."""
        if not GPU_ENABLED:
            return
        try:
            for item, pil_img in self._get_clothes_crops(cache_entry, person_crop_bgr):
                res = self.clothes_pattern_clf.process(pil_img)
                item["details"]["pattern"] = max(res, key=res.get)
        except Exception as e:
            print(f"Clothes pattern error: {e}")

    # ------------------------------------------------------------------

    @staticmethod
    def build_add_info(cache_entry: dict) -> list:
        """Buduje listę add_info z cache'u osoby."""
        add_info = []
        if cache_entry.get("gender"):
            add_info.append({"gender": cache_entry["gender"]})
        if cache_entry.get("age"):
            add_info.append({"age": cache_entry["age"]})
        if cache_entry.get("emotion"):
            add_info.append({"emotion": cache_entry["emotion"]})
        if cache_entry.get("clothes"):
            add_info.append({"clothes": cache_entry["clothes"]})
        return add_info
