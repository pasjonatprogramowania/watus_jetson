"""Testy klasy VideoIO z mockowanym VideoCapture."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestVideoIO(unittest.TestCase):

    @patch("warstwa_wizji.src.video_io.cv2.VideoCapture")
    def test_creation_with_source(self, mock_cap_cls):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap
        vio = VideoIO(source=0)
        mock_cap_cls.assert_called_once_with(0)
        self.assertIsNotNone(vio.cap)

    @patch("warstwa_wizji.src.video_io.cv2.VideoCapture")
    def test_creation_fails_when_not_opened(self, mock_cap_cls):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap
        with self.assertRaises(RuntimeError):
            VideoIO(source=0)

    def test_creation_with_existing_cap(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        self.assertIs(vio.cap, mock_cap)

    def test_read_frame_delegates(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        vio = VideoIO(cap=mock_cap)
        ret, f = vio.read_frame()
        self.assertTrue(ret)
        np.testing.assert_array_equal(f, frame)

    def test_release_delegates(self):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        vio.release()
        mock_cap.release.assert_called_once()

    @patch("warstwa_wizji.src.video_io.cv2.imshow")
    def test_show_frame_calls_imshow(self, mock_imshow):
        from warstwa_wizji import VideoIO
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        vio = VideoIO(cap=mock_cap)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vio.show_frame(frame)
        mock_imshow.assert_called_once()
