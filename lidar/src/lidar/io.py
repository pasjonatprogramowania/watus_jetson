from typing import Any, Dict, List
from pathlib import Path
import json

from .types import BeamCategory, CellType, CellDanger

def convert_beam_category_to_int(cat: BeamCategory) -> int:
    """
    Konwertuje kategorię wiązki LiDAR na wartość całkowitą dla JSON.
    
    Mapowanie: NONE->0, HUMAN->1, OBSTACLE->2.
    
    Hierarchia wywołań:
        lidar/src/lidar/io.py -> convert_beam_category_to_int()
    """
    if cat == BeamCategory.NONE:
        return 0
    if cat == BeamCategory.HUMAN:
        return 1
    if cat == BeamCategory.OBSTACLE:
        return 2
    return -1

def convert_cell_type_to_int(ct: CellType) -> int:
    """
    Konwertuje typ komórki na int dla JSON.
    
    Hierarchia wywołań:
        lidar/src/lidar/io.py -> convert_cell_type_to_int()
    """
    return int(ct)

def convert_cell_danger_to_int(cd: CellDanger) -> int:
    """
    Konwertuje poziom zagrożenia na int dla JSON.
    
    Hierarchia wywołań:
        lidar/src/lidar/io.py -> convert_cell_danger_to_int()
    """
    return int(cd)

def load_json_from_file(path: Path) -> Dict[str, Any]:
    """
    Wczytuje słownik z pliku JSON.
    
    Hierarchia wywołań:
        lidar/src/lidar/io.py -> load_json_from_file()
    """
    return json.loads(path.read_text(encoding="utf-8"))

def save_json_to_file(path: Path, data: Dict[str, Any]) -> None:
    """
    Zapisuje słownik do pliku JSON.
    
    Hierarchia wywołań:
        lidar/src/lidar/io.py -> save_json_to_file()
    """
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
