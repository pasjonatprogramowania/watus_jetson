# src/live_vis_v3.py
# Offline vis: zbiera N_SCANS skanów, potem renderuje GIF (bez laga).
# Wersja V3:
#  - ślady obiektów,
#  - mała strzałka przewidywanego ruchu,
#  - legenda z boku z kolorem, ID i prędkością [m/s].

import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .config import (
    LIDAR_PORT,
    R_MAX_M,
    MAP_WIDTH_M,
    MAP_HEIGHT_M,
    CELL_SIZE_M,
)
from .hardware.lidar_driver import initialize_lidar, acquire_complete_scan
from .lidar.system import AiwataLidarSystem
from .lidar.types import Pose2D, BeamResult, BeamCategory
from .lidar.io import convert_beam_category_to_int

# parametry zbierania i wizualizacji
N_SCANS = 200            # liczba skanów do nagrania
OUTPUT_GIF = "vis_v3_tracking.gif"
FRAME_DIR = "vis_v3_frames"
GIF_FPS = 10
FULL_SCAN_PACKETS = 60

# strefa, w której rozpoznajemy obiekty ruchome (dopasuj do segmentacji!)
MOVING_DET_MIN_R = 0.7   # m
MOVING_DET_MAX_R = 3.5   # m


def convert_beams_to_cartesian(beams: List[BeamResult]) -> Tuple[List[float], List[float]]:
    """
    Konwertuje listę wyników wiązek LiDAR na współrzędne kartezjańskie.
    
    Funkcja przekształca współrzędne biegunowe (r, theta) na kartezjańskie (x, y),
    pomijając wiązki bez kategorii (NONE).
    
    Argumenty:
        beams (List[BeamResult]): Lista obiektów BeamResult z polami r, theta, category.
    
    Zwraca:
        Tuple[List[float], List[float]]: Krotka (xs, ys) z listami współrzędnych X i Y.
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> record_scan_sequence() -> convert_beams_to_cartesian()
    """
    xs: List[float] = []
    ys: List[float] = []
    for b in beams:
        if b.category == BeamCategory.NONE:
            continue
        x = b.r * np.cos(b.theta)
        y = b.r * np.sin(b.theta)
        xs.append(x)
        ys.append(y)
    return xs, ys


def configure_radar_plot_axes(ax, max_range: float) -> None:
    """
    Konfiguruje osie wykresu radarowego dla wizualizacji LiDAR.
    
    Argumenty:
        ax (plt.Axes): Obiekt osi matplotlib do konfiguracji.
        max_range (float): Maksymalny zasięg w metrach do wyświetlenia.
    
    Zwraca:
        None
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> render_all_frames_to_png() -> configure_radar_plot_axes()
    """
    ax.clear()
    ax.set_title("LIDAR", color="white")
    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal", "box")

    ax.set_facecolor("black")
    ax.grid(True, color="gray", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")

    # standardowe okręgi zasięgu
    for r in np.linspace(max_range / 4, max_range, 4):
        circle = plt.Circle(
            (0, 0),
            r,
            color="gray",
            alpha=0.2,
            fill=False,
            linestyle="--",
        )
        ax.add_patch(circle)

    # strefa detekcji ruchu: od MOVING_DET_MIN_R do MOVING_DET_MAX_R
    if MOVING_DET_MIN_R > 0.0:
        inner_circle = plt.Circle(
            (0, 0),
            MOVING_DET_MIN_R,
            color="yellow",
            alpha=0.25,
            fill=False,
            linestyle=":",
            linewidth=1.0,
        )
        ax.add_patch(inner_circle)

    if MOVING_DET_MAX_R < max_range:
        outer_circle = plt.Circle(
            (0, 0),
            MOVING_DET_MAX_R,
            color="yellow",
            alpha=0.35,
            fill=False,
            linestyle="-.",
            linewidth=1.2,
        )
        ax.add_patch(outer_circle)

        ax.text(
            MOVING_DET_MAX_R,
            0.1 * MOVING_DET_MAX_R,
            "strefa ruchu",
            color="yellow",
            fontsize=7,
            ha="left",
            va="center",
            alpha=0.6,
        )

    # lidar w środku
    ax.scatter(0.0, 0.0, c="red", marker="x", s=80)

    # subtelny wskaźnik kierunku robota: +X = przód, +Y = lewo
    arrow_len = max_range * 0.18
    line_width = max_range * 0.004
    alpha = 0.45

    ax.arrow(
        0.0,
        0.0,
        arrow_len,
        0.0,
        width=line_width,
        length_includes_head=True,
        color="yellow",
        alpha=alpha,
    )
    ax.text(
        arrow_len * 0.95,
        0.0,
        "przód",
        color="yellow",
        fontsize=7,
        ha="left",
        va="center",
        alpha=alpha,
    )

    ax.arrow(
        0.0,
        0.0,
        0.0,
        arrow_len,
        width=line_width,
        length_includes_head=True,
        color="yellow",
        alpha=alpha,
    )
    ax.text(
        0.0,
        arrow_len * 0.95,
        "lewo",
        color="yellow",
        fontsize=7,
        ha="center",
        va="bottom",
        alpha=alpha,
    )


def record_scan_sequence() -> List[Dict[str, Any]]:
    """
    Nagrywa serię skanów LiDAR do pamięci.
    
    Funkcja inicjalizuje LiDAR, tworzy system przetwarzania i zbiera N_SCANS skanów.
    
    Argumenty:
        Brak
    
    Zwraca:
        List[Dict[str, Any]]: Lista klatek, każda zawiera klucze xs, ys, tracks.
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> main() -> record_scan_sequence()
    """
    print(f"Próbuję otworzyć lidar na porcie: {LIDAR_PORT}")
    initialize_lidar(port=LIDAR_PORT)
    print("Lidar podłączony i port otwarty.")

    system = AiwataLidarSystem(
        map_width_m=MAP_WIDTH_M,
        map_height_m=MAP_HEIGHT_M,
        cell_size_m=CELL_SIZE_M,
    )

    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    t0 = time.time()

    frames: List[Dict[str, Any]] = []

    print(f"Nagrywam {N_SCANS} skanów bez wizualizacji...")
    for i in range(N_SCANS):
        r, theta = acquire_complete_scan(num_packets=FULL_SCAN_PACKETS)
        t = time.time() - t0

        result = system.process_complete_lidar_scan(r, theta, pose, t)

        xs, ys = convert_beams_to_cartesian(result.beams)

        tracks_info: List[Dict[str, Any]] = []
        for tr in result.human_tracks:
            vx = float(getattr(tr, "vx", 0.0))
            vy = float(getattr(tr, "vy", 0.0))
            speed = float(getattr(tr, "speed", np.hypot(vx, vy)))

            tracks_info.append({
                "id": tr.id,
                "x": float(tr.x),
                "y": float(tr.y),
                "vx": vx,
                "vy": vy,
                "speed": speed,
            })

        frames.append({
            "xs": xs,
            "ys": ys,
            "tracks": tracks_info,
        })

        if (i + 1) % 100 == 0:
            print(f"Zebrano skanów: {i + 1}/{N_SCANS}")

    print("Zakończono nagrywanie skanów.")
    return frames


def ensure_directory_empty(path: Path) -> None:
    """
    Tworzy pusty katalog lub czyści istniejący.
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> render_all_frames_to_png() -> ensure_directory_empty()
    """
    if path.exists():
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                for q in p.rglob("*"):
                    if q.is_file():
                        q.unlink()
                p.rmdir()
    else:
        path.mkdir(parents=True, exist_ok=True)


def render_all_frames_to_png(frames: List[Dict[str, Any]], frame_dir: Path) -> List[Path]:
    """
    Renderuje wszystkie klatki LiDAR do plików PNG.
    
    Argumenty:
        frames (List[Dict[str, Any]]): Lista klatek z record_scan_sequence().
        frame_dir (Path): Katalog na pliki PNG.
    
    Zwraca:
        List[Path]: Lista ścieżek do utworzonych plików PNG.
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> main() -> render_all_frames_to_png()
    """
    ensure_directory_empty(frame_dir)

    max_range = min(R_MAX_M, MAP_WIDTH_M / 2.0, MAP_HEIGHT_M / 2.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    configure_radar_plot_axes(ax, max_range)

    beams_scatter = ax.scatter([], [], s=3, c="white")
    track_scatters: List[Any] = []
    track_texts: List[Any] = []
    trail_lines: List[Any] = []
    pred_arrows: List[Any] = []
    legend_texts: List[Any] = []

    # ślady: historia pozycji dla każdego ID
    track_trails: Dict[str, List[Tuple[float, float]]] = {}

    # stałe kolory per ID
    cmap = plt.get_cmap("tab10")
    n_colors = 10

    frame_paths: List[Path] = []

    print("Renderuję klatki do PNG...")
    for i, frame in enumerate(frames):
        xs = frame["xs"]
        ys = frame["ys"]
        tracks = frame["tracks"]

        if xs:
            data = np.column_stack((xs, ys))
        else:
            data = np.empty((0, 2))
        beams_scatter.set_offsets(data)

        # usuwamy stare obiekty tracków z poprzedniej klatki
        for sc in track_scatters:
            sc.remove()
        for txt in track_texts:
            txt.remove()
        for line in trail_lines:
            line.remove()
        for arr in pred_arrows:
            arr.remove()
        for ltxt in legend_texts:
            ltxt.remove()

        track_scatters.clear()
        track_texts.clear()
        trail_lines.clear()
        pred_arrows.clear()
        legend_texts.clear()

        MAX_TRAIL_LEN = 40          # maksymalna długość śladu
        PRED_DT = 0.6               # horyzont predykcji dla małej strzałki [s]
        MIN_ARROW_SPEED = 0.05      # próg rysowania strzałki [m/s]

        # --- rysowanie tracków i śladów ---
        for tr in tracks:
            track_id = tr["id"] # uuid str
            x = tr["x"]
            y = tr["y"]
            vx = tr["vx"]
            vy = tr["vy"]
            speed = tr["speed"]

            # filtr: pokazujemy tylko obiekty w "strefie ruchu"
            dist = float(np.hypot(x, y))
            if dist < MOVING_DET_MIN_R or dist > MOVING_DET_MAX_R:
                continue
            
            color_idx = hash(track_id) % n_colors
            color = cmap(color_idx)

            # aktualizacja śladu
            trail = track_trails.get(track_id)
            if trail is None:
                trail = []
                track_trails[track_id] = trail
            trail.append((x, y))
            if len(trail) > MAX_TRAIL_LEN:
                trail.pop(0)

            # cienka kreska śladu
            if len(trail) >= 2:
                tx = [p[0] for p in trail]
                ty = [p[1] for p in trail]
                line = ax.plot(tx, ty, linewidth=0.7, color=color)[0]
                trail_lines.append(line)

            # aktualna pozycja (kropka)
            sc = ax.scatter([x], [y], s=40, c=[color])
            track_scatters.append(sc)

            # ID nad punktem
            short_id = str(track_id)[:4]
            txt = ax.text(
                x,
                y,
                short_id,
                color="white",
                fontsize=8,
                ha="center",
                va="bottom",
            )
            track_texts.append(txt)

            # mała strzałka przewidywanego ruchu
            if speed >= MIN_ARROW_SPEED:
                dx = vx * PRED_DT
                dy = vy * PRED_DT
                arr = ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    width=0.02,
                    length_includes_head=True,
                    color=color,
                    alpha=0.9,
                )
                pred_arrows.append(arr)

        # --- legenda z boku ---
        tracks_sorted = sorted(tracks, key=lambda t: t["id"])
        max_entries = 8
        legend_y_start = 0.95
        legend_dy = 0.05

        for j, tr in enumerate(tracks_sorted[:max_entries]):
            track_id = tr["id"]
            short_id = str(track_id)[:4]
            speed = tr["speed"]
            x = tr["x"]
            y = tr["y"]
            dist = float(np.hypot(x, y))
            color_idx = hash(track_id) % n_colors
            color = cmap(color_idx)

            if dist < MOVING_DET_MIN_R or dist > MOVING_DET_MAX_R:
                continue

            text = f"ID {short_id}: {speed:.2f} m/s, {dist:.2f} m"
            ltxt = ax.text(
                0.98,
                legend_y_start - j * legend_dy,
                text,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
                color=color,
            )
            legend_texts.append(ltxt)

        fig.canvas.draw()
        frame_path = frame_dir / f"frame_{i:05d}.png"
        fig.savefig(frame_path, dpi=100)
        frame_paths.append(frame_path)

        if (i + 1) % 100 == 0:
            print(f"Zapisano klatek: {i + 1}/{len(frames)}")

    plt.close(fig)
    print("Renderowanie PNG zakończone.")
    return frame_paths


def create_gif_from_png_images(frame_paths: List[Path], output_path: Path) -> None:
    """
    Tworzy animowany GIF z listy plików PNG.
    
    Hierarchia wywołań:
        lidar/src/Live_Vis_v3.py -> main() -> create_gif_from_png_images()
    """
    print(f"Tworzę GIF: {output_path}")
    images = []
    for p in frame_paths:
        images.append(imageio.imread(p))
    imageio.mimsave(output_path, images, fps=GIF_FPS)
    print("GIF zapisany.")


def main() -> None:
    """
    Główna funkcja offline wizualizacji LiDAR.
    
    Orkiestruje cały proces:
      1. Nagrywanie skanów z LiDAR
      2. Renderowanie klatek do PNG
      3. Tworzenie animowanego GIF
    
    Hierarchia wywołań:
        lidar/create_gif.py -> main()
        lidar/run_vis.py -> main()
        lidar/src/Live_Vis_v3.py -> __main__ -> main()
    """
    frames = record_scan_sequence()

    frame_dir = Path(FRAME_DIR)
    frame_paths = render_all_frames_to_png(frames, frame_dir)

    output_path = Path(OUTPUT_GIF)
    create_gif_from_png_images(frame_paths, output_path)

    print("Gotowe.")


if __name__ == "__main__":
    main()
