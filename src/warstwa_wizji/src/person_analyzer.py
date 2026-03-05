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

# Liczba klatek przed odświeżeniem cache
CACHE_TTL_FRAMES = 100


class PersonAnalyzer:
    """
    Analizuje atrybuty osób: ubrania, kolor, emocje, płeć, wiek.

    Zarządza cacheem wyników, aby nie powtarzać kosztownych operacji
    klasyfikacji w każdej klatce.

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

        Argumenty:
            person_crop_bgr: Wycinek klatki z osobą (BGR).
            track_id: ID trackera osoby.
            frame_idx: Aktualny indeks klatki.
            clothes_detector: Model YOLO do detekcji ubrań.

        Zwraca:
            Słownik cache_entry z kluczami:
            clothes, emotion, gender, age, lidar_data, last_frame.
        """
        cache_entry = self.person_cache.get(track_id, {"last_frame": -9999})
        should_update = (frame_idx - cache_entry["last_frame"]) > CACHE_TTL_FRAMES

        if not should_update:
            return cache_entry

        current_h, current_w = person_crop_bgr.shape[:2]

        new_cache = {
            "last_frame": frame_idx,
            "clothes": [],
            "emotion": cache_entry.get("emotion"),
            "gender": cache_entry.get("gender"),
            "age": cache_entry.get("age"),
            "lidar_data": cache_entry.get("lidar_data"),
        }

        # 1. Detekcja ubrań
        clothes_data = self._detect_clothes(person_crop_bgr, clothes_detector, current_w, current_h)
        new_cache["clothes"] = clothes_data

        # 2. Klasyfikatory (emocje, płeć, wiek) — wyłączone flagą `and False`
        try:
            pil_img = Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))

            if GPU_ENABLED and False:
                res_emo = self.emotion_classifier.process(pil_img)
                new_cache["emotion"] = max(res_emo, key=res_emo.get)

                res_gen = self.gender_classifier.process(pil_img)
                new_cache["gender"] = max(res_gen, key=res_gen.get)

                res_age = self.age_classifier.process(pil_img)
                new_cache["age"] = max(res_age, key=res_age.get)
        except Exception as e:
            print(f"Classifier error: {e}")

        self.person_cache[track_id] = new_cache
        return new_cache

    # ------------------------------------------------------------------

    def _detect_clothes(
        self,
        person_crop_bgr: np.ndarray,
        clothes_detector: Any,
        crop_w: int,
        crop_h: int,
    ) -> list:
        """Wykrywa ubrania na wycinku osoby i klasyfikuje ich kolor."""
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

                details = {}
                label_name = c_names[int(cc)]

                # Wycinek konkretnego elementu ubrania
                icx1, icy1 = max(0, int(cx1)), max(0, int(cy1))
                icx2, icy2 = min(crop_w, int(cx2)), min(crop_h, int(cy2))

                if icx2 > icx1 and icy2 > icy1:
                    item_crop = person_crop_bgr[icy1:icy2, icx1:icx2]
                    try:
                        item_pil = Image.fromarray(
                            cv2.cvtColor(item_crop, cv2.COLOR_BGR2RGB)
                        )
                        if label_name in ["clothing", "shoe", "bag"] and GPU_ENABLED:
                            dom_color = self.color_clf(np.asarray(item_pil))
                            details["color"] = [int(x) for x in dom_color]
                    except Exception as e:
                        print(f"Clothes detail error: {e}")

                clothes_data.append({
                    "box_norm": norm_box,
                    "label": label_name,
                    "class_num": int(cc),
                    "conf": float(ccnf),
                    "details": details,
                })

        return clothes_data

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
