"""
Testy dostępności sprzętowej LiDAR.

Wykonywane TYLKO na systemie Linux — na Windowsie są pomijane,
ponieważ LiDAR (np. RPLidar / LD06) wymaga urządzeń /dev/ttyUSB*.
"""

import os
import sys
import glob
import unittest

IS_LINUX = sys.platform.startswith("linux")
SKIP_REASON = "Testy LiDAR dostępności wymagają systemu Linux"

# Typowe ścieżki urządzeń LiDAR
LIDAR_DEV_PATTERNS = ["/dev/ttyUSB*", "/dev/ttyACM*"]
LIDAR_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lidar", "data", "lidar.json")
)


def _find_lidar_devices() -> list:
    """Szuka dostępnych urządzeń seryjnych pasujących do LiDAR."""
    devices = []
    for pattern in LIDAR_DEV_PATTERNS:
        devices.extend(glob.glob(pattern))
    return devices


@unittest.skipUnless(IS_LINUX, SKIP_REASON)
class TestLidarHardwareAvailability(unittest.TestCase):
    """Testy dostępności sprzętowej i pliku danych LiDAR (Linux-only)."""

    def test_serial_ports_exist(self):
        """System Linux posiada urządzenia /dev/ttyUSB* lub /dev/ttyACM*."""
        devices = _find_lidar_devices()
        self.assertGreater(
            len(devices), 0,
            f"Nie znaleziono urządzeń LiDAR w {LIDAR_DEV_PATTERNS}"
        )

    def test_serial_port_readable(self):
        """Pierwsze znalezione urządzenie LiDAR jest do odczytu."""
        devices = _find_lidar_devices()
        if not devices:
            self.skipTest("Brak urządzeń LiDAR — pomijam")
        first_dev = devices[0]
        self.assertTrue(
            os.access(first_dev, os.R_OK),
            f"Urządzenie {first_dev} nie ma uprawnień do odczytu"
        )

    def test_lidar_data_file_exists(self):
        """Plik lidar.json istnieje (strumień danych aktywny)."""
        if not os.path.isfile(LIDAR_DATA_PATH):
            self.skipTest(f"Plik {LIDAR_DATA_PATH} nie istnieje — LiDAR nie aktywny")
        self.assertTrue(os.path.isfile(LIDAR_DATA_PATH))

    def test_lidar_data_file_nonempty(self):
        """Plik lidar.json nie jest pusty."""
        if not os.path.isfile(LIDAR_DATA_PATH):
            self.skipTest(f"Plik {LIDAR_DATA_PATH} nie istnieje — LiDAR nie aktywny")
        size = os.path.getsize(LIDAR_DATA_PATH)
        self.assertGreater(size, 0, "Plik lidar.json jest pusty")

    def test_read_lidar_tracks_from_real_file(self):
        """Odczyt tracków z prawdziwego pliku lidar.json."""
        if not os.path.isfile(LIDAR_DATA_PATH):
            self.skipTest(f"Plik {LIDAR_DATA_PATH} nie istnieje — LiDAR nie aktywny")
        from warstwa_wizji import read_lidar_tracks
        tracks = read_lidar_tracks(LIDAR_DATA_PATH)
        self.assertIsInstance(tracks, list)
        # Tracki mogą być puste, ale nie powinien wystąpić wyjątek


@unittest.skipUnless(IS_LINUX, SKIP_REASON)
class TestLidarDevicePermissions(unittest.TestCase):
    """Testy uprawnień do urządzeń LiDAR (Linux-only)."""

    def test_dev_tty_group_membership(self):
        """Użytkownik ma uprawnienia do grupy dialout/uucp (dostęp do /dev/tty*)."""
        import grp
        username = os.environ.get("USER", "")
        if not username:
            self.skipTest("Nie można ustalić nazwy użytkownika")

        lidar_groups = ["dialout", "uucp", "tty"]
        user_in_any_group = False
        for group_name in lidar_groups:
            try:
                group_info = grp.getgrnam(group_name)
                if username in group_info.gr_mem:
                    user_in_any_group = True
                    break
            except KeyError:
                continue

        # Informacyjny — nie blokujemy jeśli root
        if os.geteuid() == 0:
            user_in_any_group = True

        self.assertTrue(
            user_in_any_group,
            f"Użytkownik '{username}' nie należy do grup {lidar_groups}. "
            f"Uruchom: sudo usermod -aG dialout {username}"
        )
