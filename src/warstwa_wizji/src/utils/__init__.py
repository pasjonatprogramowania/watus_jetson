"""
Pakiet cv_utils - narzędzia do przetwarzania obrazów i wizji komputerowej.

Zawiera moduły do:
- Obliczania kątów obiektów (angle)
- Analizy jasności (brightness)
- Integracji z LiDAR (lidar_integration)
- Przetwarzania detekcji (detection_processors)
- Wizualizacji (frame_overlay)
- Śledzenia obiektów (old_tracker, new_tracker)
- Wrappera modeli detekcji (cv_wrapper)

Hierarchia wywołań:
    warstwa_wizji/main.py -> warstwa_wizji/src/cv_utils/*
"""

from .angle import calc_obj_angle
from .brightness import calc_brightness, suggest_mode
from .old_tracker import IoUTracker as Tracker
from .new_tracker import nms_per_class
from .cv_wrapper import CVWrapper
from .lidar_integration import read_lidar_tracks, compute_lidar_angle, match_camera_to_lidar
from .detection_processors import (
    process_clothes_detection, 
    process_weapon_detection, 
    update_person_cache,
    should_update_cache
)
from .frame_overlay import (
    draw_clothes_boxes,
    draw_lidar_info,
    draw_stats_overlay,
    draw_weapon_boxes,
    draw_track_history
)