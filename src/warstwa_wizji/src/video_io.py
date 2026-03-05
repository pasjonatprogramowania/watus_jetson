"""
Moduł obsługi wejścia/wyjścia wideo.

Odpowiada za otwieranie kamery (lokalnej lub GStreamer), odczyt klatek,
nagrywanie wideo i zarządzanie oknem OpenCV.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.__init__() -> VideoIO()
"""

import cv2


ESCAPE_BUTTON = "q"


class VideoIO:
    """
    Obsługa kamery, nagrywania i okna wyświetlania.

    Atrybuty:
        cap (cv2.VideoCapture): Źródło wideo.
        video_recorder (cv2.VideoWriter | None): Nagrywarka wideo.
        window_name (str): Nazwa okna OpenCV.
    """

    def __init__(
        self,
        source: int | str = 0,
        cap=None,
        use_net_stream: bool = False,
    ):
        if cap is not None:
            self.cap = cap
        elif use_net_stream:
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
            raise RuntimeError("Nie mogę otworzyć kamery")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_recorder = None
        self.window_name = f"YOLOv12 – naciśnij '{ESCAPE_BUTTON}' aby wyjść"

    # ------------------------------------------------------------------

    def init_recorder(self, out_path: str) -> bool:
        """Inicjalizuje nagrywarkę wideo MP4."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        fps_output = float(fps_cap) if fps_cap and fps_cap > 1.0 else 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_recorder = cv2.VideoWriter(out_path, fourcc, fps_output, (width, height))
        return self.video_recorder is not None

    def init_window(self):
        """Tworzy okno OpenCV do podglądu."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def read_frame(self):
        """Odczytuje jedną klatkę. Zwraca (ret, frame_bgr)."""
        return self.cap.read()

    def show_frame(self, frame_bgr):
        """Wyświetla klatkę w oknie OpenCV."""
        cv2.imshow(self.window_name, frame_bgr)

    def write_frame(self, frame_bgr):
        """Zapisuje klatkę do pliku wideo (jeśli recorder aktywny)."""
        if self.video_recorder is not None:
            self.video_recorder.write(frame_bgr)

    def release(self):
        """Zwalnia zasoby kamery, nagrywarki i okien."""
        self.cap.release()
        if self.video_recorder is not None:
            self.video_recorder.release()
        cv2.destroyAllWindows()
