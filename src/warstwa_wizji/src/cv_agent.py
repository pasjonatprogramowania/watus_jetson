"""
Główny agent wizyjny — lekki orkiestrator.

Deleguje odpowiedzialności do wyspecjalizowanych klas:
- ModelManager: ładowanie i zarządzanie modelami YOLO
- VideoIO: obsługa kamery i nagrywania
- PersonAnalyzer: analiza atrybutów osób
- DetectionPipeline: przetwarzanie klatek i budowanie wyników

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.run()
"""

import cv2

from warstwa_wizji.src.model_manager import ModelManager
from warstwa_wizji.src.video_io import VideoIO, ESCAPE_BUTTON
from warstwa_wizji.src.person_analyzer import PersonAnalyzer
from warstwa_wizji.src.detection_pipeline import DetectionPipeline


class CVAgent:
    """
    Orkiestrator warstwy wizyjnej — łączy modele, wideo i pipeline detekcji.

    Argumenty __init__:
        weights_path: Ścieżka do wag YOLO.
        imgsz: Rozmiar wejściowy obrazu.
        source: Indeks kamery lub ścieżka do pliku wideo.
        cap: Opcjonalny, gotowy obiekt VideoCapture.
        json_save_func: Callback do zapisu wyników do JSONL.
        use_net_stream: Czy korzystać z GStreamer.
        export_to_engine: Czy eksportować modele do TensorRT.
    """

    def __init__(
        self,
        weights_path: str = "yolo12s.pt",
        imgsz: int = 640,
        source: int | str = 0,
        cap=None,
        json_save_func=None,
        use_net_stream: bool = False,
        export_to_engine: bool = False,
        use_yoloe: bool = False,
    ):
        self.save_to_json = json_save_func
        self.imgsz = imgsz

        # --- Komponenty ---
        self.models = ModelManager(
            weights_path=weights_path,
            imgsz=imgsz,
            export_to_engine=export_to_engine,
            use_yoloe=use_yoloe
        )
        self.video = VideoIO(
            source=source,
            cap=cap,
            use_net_stream=use_net_stream,
        )
        self.person_analyzer = PersonAnalyzer()
        self.pipeline = DetectionPipeline()

    # ------------------------------------------------------------------

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
        """Główna pętla detekcji i wizualizacji."""
        if save_video:
            save_video = self.video.init_recorder(out_path)
        if show_window is False or show_window is None:
            verbose_window = False
        if show_window:
            self.video.init_window()

        # Rozgrzewanie modelu
        ret, warm_frame = self.video.read_frame()
        if ret:
            self.models.warm_up(warm_frame)

        frame_idx = 0

        try:
            self.pipeline.fps_params["t_prev"] = __import__("time").time()

            while True:
                ret, frame_bgr = self.video.read_frame()
                if not ret:
                    print("Koniec strumienia")
                    break

                run_detection = (frame_idx % det_stride == 0)
                dets = self.models.detect_objects(
                    frame_bgr, imgsz=self.imgsz, run_detection=run_detection,
                )

                detections, frame_bgr_vis = self.pipeline.process_frame(
                    frame_bgr=frame_bgr,
                    dets=dets,
                    frame_idx=frame_idx,
                    imgsz=self.imgsz,
                    fov_deg=fov_deg,
                    class_names=self.models.class_names,
                    person_analyzer=self.person_analyzer,
                    model_manager=self.models,
                    consolidate_with_lidar=consolidate_with_lidar,
                    show_window=show_window,
                    verbose_window=verbose_window,
                    show_fps=show_fps,
                )

                # Zapis JSON
                if self.save_to_json is not None:
                    self.save_to_json("camera.jsonl", detections)

                # Wyświetlanie / nagrywanie
                if show_window:
                    self.video.show_frame(frame_bgr_vis)
                if save_video:
                    self.video.write_frame(frame_bgr_vis)

                frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord(ESCAPE_BUTTON):
                    break
                elif key == ord("w"):
                    self.pipeline.toggle_weapon_detection()

        except KeyboardInterrupt:
            print("Przerwano przez użytkownika.")
        finally:
            self.video.release()


if __name__ == "__main__":
    agent = CVAgent()
    agent.run(save_video=True, show_window=True, consolidate_with_lidar=False)
