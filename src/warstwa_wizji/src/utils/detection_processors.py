"""
Moduł procesorów detekcji - przetwarzanie wykrytych obiektów.

Zawiera funkcje do przetwarzania detekcji ubrań, broni
oraz zarządzania pamięcią podręczną (cache) atrybutów osób.
Umożliwia ekstrakcję szczegółowych informacji o wykrytych obiektach.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent._process_person() 
        -> process_clothes_detection()
    warstwa_wizji/main.py -> CVAgent._handle_weapon_detection()
        -> process_weapon_detection()
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Any
import torch


# Stałe konfiguracyjne
DEFAULT_CACHE_TTL_FRAMES = 100  # Liczba klatek przed odświeżeniem cache


def process_clothes_detection(
    person_crop_bgr: np.ndarray,
    clothes_detector_model: Any,
    color_classifier: Optional[Any] = None,
    gpu_enabled: bool = True
) -> list:
    """
    Przetwarza detekcję ubrań na wycinku obrazu zawierającym osobę.
    
    Funkcja wykrywa elementy ubioru (ubrania, buty, torby) na wyciętym
    fragmencie obrazu i opcjonalnie klasyfikuje ich kolory.
    
    Argumenty:
        person_crop_bgr (np.ndarray): Wycinek obrazu z osobą w formacie BGR.
        clothes_detector_model: Model YOLO wytrenowany do detekcji ubrań.
        color_classifier: Opcjonalny klasyfikator do określania koloru ubrań.
        gpu_enabled (bool): Czy GPU jest dostępne do przetwarzania.
        
    Zwraca:
        list: Lista słowników z informacjami o wykrytych ubraniach:
              - "box_norm": Znormalizowane współrzędne [x1, y1, x2, y2] (0-1)
              - "label": Nazwa klasy ubrania (np. "clothing", "shoe")
              - "class_num": Numer klasy
              - "conf": Pewność detekcji (0-1)
              - "details": Słownik z dodatkowymi informacjami (np. kolor)
              
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_person()
            -> process_clothes_detection()
    """
    crop_height, crop_width = person_crop_bgr.shape[:2]
    detected_clothes = []
    
    # Uruchom detekcję ubrań
    detection_results = clothes_detector_model.predict(person_crop_bgr, verbose=False)
    
    for result in detection_results:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names
        
        for box, class_id, confidence in zip(boxes_xyxy, class_ids, confidences):
            x1, y1, x2, y2 = box
            
            # Normalizuj współrzędne do zakresu 0-1 względem wycinka
            normalized_box = [
                float(x1) / crop_width, 
                float(y1) / crop_height, 
                float(x2) / crop_width, 
                float(y2) / crop_height
            ]
            
            details = {}
            label_name = class_names[int(class_id)]
            
            # Wytnij fragment ubrania do analizy koloru
            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(crop_width, int(x2)), min(crop_height, int(y2))
            
            if x2_int > x1_int and y2_int > y1_int:
                clothing_crop = person_crop_bgr[y1_int:y2_int, x1_int:x2_int]
                
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2RGB))
                    
                    # Klasyfikuj kolor dla istotnych klas ubrań
                    if label_name in ["clothing", "shoe", "bag"]:
                        if gpu_enabled and color_classifier is not None:
                            dominant_color = color_classifier(np.asarray(pil_image))
                            details["color"] = [int(c) for c in dominant_color]
                            
                except Exception as error:
                    print(f"Błąd analizy szczegółów ubrania: {error}")
            
            detected_clothes.append({
                "box_norm": normalized_box,
                "label": label_name,
                "class_num": int(class_id),
                "conf": float(confidence),
                "details": details
            })
    
    return detected_clothes


def process_weapon_detection(
    frame_bgr: np.ndarray,
    weapon_detector_model: Any,
    frame_index: int,
    detection_interval: int = 10
) -> tuple[list, bool]:
    """
    Przetwarza detekcję broni na klatce wideo.
    
    Funkcja uruchamia detekcję tylko co określoną liczbę klatek
    dla optymalizacji wydajności.
    
    Argumenty:
        frame_bgr (np.ndarray): Klatka obrazu w formacie BGR.
        weapon_detector_model: Model YOLO wytrenowany do detekcji broni.
        frame_index (int): Numer bieżącej klatki wideo.
        detection_interval (int): Co ile klatek uruchamiać detekcję.
        
    Zwraca:
        tuple: (lista_wykrytej_broni, czy_uruchomiono_detekcje).
               - lista_wykrytej_broni: Lista słowników z danymi o broni
               - czy_uruchomiono_detekcje: True jeśli detekcja została wykonana
               
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._handle_weapon_detection()
            -> process_weapon_detection()
    """
    # Sprawdź czy to klatka do analizy
    if frame_index % detection_interval != 0:
        return [], False
    
    detected_weapons = []
    
    with torch.inference_mode():
        detection_results = weapon_detector_model(frame_bgr, verbose=False)
    
    for result in detection_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            weapon_label = result.names[class_id]
            
            detected_weapons.append({
                "box": [x1, y1, x2, y2],
                "conf": confidence,
                "label": weapon_label
            })
    
    return detected_weapons, True


def should_update_cache(
    person_cache: dict,
    track_id: int,
    frame_index: int,
    cache_ttl: int = DEFAULT_CACHE_TTL_FRAMES
) -> bool:
    """
    Sprawdza czy należy odświeżyć cache atrybutów dla danej osoby.
    
    Cache jest używany do unikania powtarzania kosztownych obliczeń
    (klasyfikacja ubrań, atrybutów) w każdej klatce.
    
    Argumenty:
        person_cache (dict): Słownik cache'u wszystkich śledzonych osób.
        track_id (int): ID trackera osoby (z modelu śledzenia).
        frame_index (int): Numer bieżącej klatki wideo.
        cache_ttl (int): Liczba klatek przed wymuszeniem odświeżenia.
        
    Zwraca:
        bool: True jeśli cache wymaga aktualizacji, False w przeciwnym razie.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_person()
            -> should_update_cache()
    """
    cache_entry = person_cache.get(track_id, {"last_frame": -9999})
    last_update_frame = cache_entry.get("last_frame", -9999)
    
    return (frame_index - last_update_frame) > cache_ttl


def update_person_cache(
    person_cache: dict,
    track_id: int,
    frame_index: int,
    clothes_data: list,
    classifier_data: Optional[dict] = None,
    lidar_data: Optional[dict] = None
) -> dict:
    """
    Aktualizuje cache atrybutów dla śledzionej osoby.
    
    Argumenty:
        person_cache (dict): Słownik cache'u wszystkich śledzonych osób.
        track_id (int): ID trackera osoby.
        frame_index (int): Numer bieżącej klatki.
        clothes_data (list): Lista wykrytych ubrań.
        classifier_data (dict): Opcjonalne dane z klasyfikatorów (emocje, płeć, wiek).
        lidar_data (dict): Opcjonalne dane z LiDAR (id, dystans).
        
    Zwraca:
        dict: Zaktualizowany wpis cache'u dla osoby.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent._process_person()
            -> update_person_cache()
    """
    previous_entry = person_cache.get(track_id, {})
    
    new_cache_entry = {
        "last_frame": frame_index,
        "clothes": clothes_data,
        "emotion": classifier_data.get("emotion") if classifier_data else previous_entry.get("emotion"),
        "gender": classifier_data.get("gender") if classifier_data else previous_entry.get("gender"),
        "age": classifier_data.get("age") if classifier_data else previous_entry.get("age"),
        "lidar_data": lidar_data or previous_entry.get("lidar_data")
    }
    
    person_cache[track_id] = new_cache_entry
    return new_cache_entry
