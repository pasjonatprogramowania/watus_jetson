# grid_vis.py
# Offline wizualizacja occupancy_grid:
#  - czyta wszystkie scan_XXXXX/grid.json z podanej sesji,
#  - rysuje mapę w osi XY (metry),
#  - koloruje: UNKNOWN / FREE / STATIC / HUMAN,
#  - nakłada warstwę DANGER,
#  - opcjonalnie zapisuje GIF z ewolucją mapy.

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .lidar.occupancy_grid import CellType, CellDanger
from .config import MAP_WIDTH_M, MAP_HEIGHT_M, CELL_SIZE_M


# ========== I/O GRID ==========

def load_grid_frames(session_dir: Path) -> List[Dict[str, Any]]:
    """
    Ładuje wszystkie grid.json z sesji:
    session_dir/scan_00000/grid.json
    session_dir/scan_00001/grid.json
    ...
    Zwraca listę obiektów dict (w kolejności po scan_xxxxx).
    """
    frames: List[Dict[str, Any]] = []

    scan_dirs = sorted(
        [p for p in session_dir.glob("scan_*") if p.is_dir()],
        key=lambda p: p.name,
    )

    for scan_dir in scan_dirs:
        grid_path = scan_dir / "grid.json"
        if not grid_path.exists():
            continue

        data = json.loads(grid_path.read_text(encoding="utf-8"))
        frames.append(data)

    return frames


# ========== MAPOWANIE NA KOLORY ==========

def grid_to_rgb(
    cell_type: np.ndarray,
    cell_danger: np.ndarray,
) -> np.ndarray:
    """
    Zamienia cell_type + cell_danger na obraz RGB [0..1].
    Konwencja:
      - UNKNOWN        -> ciemnoszary
      - FREE           -> jasnoszary
      - STATIC_OBSTACLE -> czarny
      - HUMAN          -> czerwony
      - DANGER         -> nałożona żółta "poświata"
    """
    h, w = cell_type.shape
    img = np.zeros((h, w, 3), dtype=float)

    # kolory bazowe
    UNKNOWN_COLOR = np.array([0.2, 0.2, 0.2])
    FREE_COLOR = np.array([0.85, 0.85, 0.85])
    STATIC_COLOR = np.array([0.0, 0.0, 0.0])
    HUMAN_COLOR = np.array([1.0, 0.0, 0.0])
    DANGER_COLOR = np.array([1.0, 1.0, 0.0])  # żółty

    img[:, :] = UNKNOWN_COLOR

    mask_free = (cell_type == int(CellType.FREE))
    mask_static = (cell_type == int(CellType.STATIC_OBSTACLE))
    mask_human = (cell_type == int(CellType.HUMAN))

    img[mask_free] = FREE_COLOR
    img[mask_static] = STATIC_COLOR
    img[mask_human] = HUMAN_COLOR

    # nakładamy "poświatę" DANGER
    mask_danger = (cell_danger == int(CellDanger.DANGER))
    if np.any(mask_danger):
        # mieszamy bazowy kolor z żółtym
        img[mask_danger] = 0.5 * img[mask_danger] + 0.5 * DANGER_COLOR

    return img


# ========== RYSOWANIE POJEDYNCZEJ KLATKI ==========

def draw_grid_frame(
    ax: plt.Axes,
    frame: Dict[str, Any],
):
    """
    Rysuje jedną klatkę occupancy grid na zadanej osi matplotlib.
    """
    cell_type = np.array(frame["cell_type"], dtype=np.int8)
    cell_danger = np.array(frame["cell_danger"], dtype=np.int8)

    map_width = float(frame.get("map_width_m", MAP_WIDTH_M))
    map_height = float(frame.get("map_height_m", MAP_HEIGHT_M))
    cell_size = float(frame.get("cell_size_m", CELL_SIZE_M))
    x_min = float(frame.get("x_min", -map_width / 2.0))
    y_min = float(frame.get("y_min", -map_height / 2.0))

    x_max = x_min + map_width
    y_max = y_min + map_height

    img = grid_to_rgb(cell_type, cell_danger)

    ax.clear()
    ax.set_title(
        f"OccupancyGrid  scan={frame.get('scan_id', '?')}  t={frame.get('timestamp', 0.0):.2f}s",
        color="white",
    )
    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")

    ax.imshow(
        img,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
    )

    # konfiguracja osi jak radar / XY
    ax.set_aspect("equal", "box")
    ax.set_facecolor("black")
    ax.grid(True, color="gray", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")

    # robot w (0,0)
    ax.scatter([0.0], [0.0], c="cyan", s=30, marker="x")
    ax.text(
        0.05,
        0.05,
        "(0,0)",
        color="cyan",
        fontsize=7,
        transform=ax.transAxes,
    )


# ========== ANIMACJA / GIF ==========

def play_session(
    session_dir: Path,
    save_gif: bool = False,
    gif_path: Path | None = None,
    fps: int = 10,
):
    frames = load_grid_frames(session_dir)
    if not frames:
        print("Brak plików grid.json w tej sesji.")
        return

    print(f"Załadowano {len(frames)} klatek grid.json")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))

    images_for_gif: List[np.ndarray] = []

    for i, frame in enumerate(frames):
        draw_grid_frame(ax, frame)
        plt.pause(0.001)
        plt.draw()

        if save_gif:
            # stabilne przechwycenie figury niezależnie od DPI/backendu
            fig.canvas.draw()
            raw_buf, (w, h) = fig.canvas.print_to_buffer()  # <-- kluczowa zmiana

            buf = np.frombuffer(raw_buf, dtype=np.uint8).reshape(h, w, 4)  # (H, W, ARGB)
            rgb = buf[:, :, 1:4]  # R,G,B
            images_for_gif.append(rgb.copy())

    if save_gif and images_for_gif:
        if gif_path is None:
            gif_path = session_dir / "occupancy_grid.gif"
        imageio.mimsave(gif_path, images_for_gif, duration=1.0 / fps)
        print(f"Zapisano GIF: {gif_path}")

    print("Zamykam okno...")
    plt.show()


# ========== CLI ==========

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline wizualizacja occupancy_grid (grid.json) z jednej sesji."
    )
    parser.add_argument(
        "session_dir",
        type=str,
        help="Ścieżka do folderu sesji (np. data/processed/lidar/session_20251211_120000)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Jeśli podane, zapisuje GIF z ewolucją mapy.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS GIF-a (domyślnie 10).",
    )

    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        print(f"Folder sesji nie istnieje: {session_dir}")
        return

    gif_path = None
    if args.gif:
        gif_path = session_dir / "occupancy_grid.gif"

    play_session(
        session_dir=session_dir,
        save_gif=args.gif,
        gif_path=gif_path,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
