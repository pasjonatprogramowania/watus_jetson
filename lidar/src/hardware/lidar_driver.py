from __future__ import annotations
from typing import Tuple, List

import struct
import numpy as np
import serial  # pip install pyserial

from src.config import LIDAR_PORT, LIDAR_BAUDRATE, LIDAR_TIMEOUT

HEADER_BYTE = 0x54  # bajt nagłówka ramki

_ser: serial.Serial | None = None


def init_lidar(
    port: str = LIDAR_PORT,
    baudrate: int = LIDAR_BAUDRATE,
    timeout: float = LIDAR_TIMEOUT,
) -> None:
    """
    Inicjalizuje połączenie z LiDAR przez port szeregowy.
    
    Funkcja otwiera port szeregowy do komunikacji z LiDAR. Jest idempotentna -
    jeśli port jest już otwarty, nie wykonuje żadnej akcji.
    
    Argumenty:
        port (str): Nazwa portu szeregowego (np. "/dev/ttyUSB0"). Domyślnie LIDAR_PORT.
        baudrate (int): Prędkość transmisji w bps. Domyślnie LIDAR_BAUDRATE (230400).
        timeout (float): Timeout odczytu w sekundach. Domyślnie LIDAR_TIMEOUT.
    
    Zwraca:
        None
    
    Wyjątki:
        serial.SerialException: Gdy nie można otworzyć portu.
    
    Hierarchia wywołań:
        run_live.py -> main() -> init_lidar()
        Live_Vis_v3.py -> record_scans() -> init_lidar()
        check_lidar.py -> main() -> init_lidar()
    """
    # Otwiera port szeregowy do lidaru (idempotentnie)
    global _ser

    if _ser is not None and _ser.is_open:
        return

    _ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
    )


def _read_exact(num_bytes: int) -> bytes:
    """
    Odczytuje dokładnie podaną liczbę bajtów z portu szeregowego.
    
    Funkcja pomocnicza zapewniająca odczyt pełnej ilości danych,
    nawet jeśli przychodzą w kilku porcjach.
    
    Argumenty:
        num_bytes (int): Liczba bajtów do odczytania.
    
    Zwraca:
        bytes: Odczytane bajty o długości num_bytes.
    
    Wyjątki:
        IOError: Gdy wystąpi timeout przed odczytaniem wszystkich bajtów.
    
    Hierarchia wywołań:
        lidar_driver.py -> _read_one_packet() -> _read_exact()
    """
    # Czyta dokładnie num_bytes lub rzuca wyjątek przy timeout
    assert _ser is not None
    data = b""
    while len(data) < num_bytes:
        chunk = _ser.read(num_bytes - len(data))
        if not chunk:
            raise IOError("Timeout podczas czytania z lidara")
        data += chunk
    return data


def _read_one_packet() -> bytes:
    """
    Odczytuje jeden pełny pakiet danych z LiDAR.
    
    Funkcja synchronizuje się z nagłówkiem pakietu (0x54), odczytuje
    informacje o liczbie punktów i pobiera resztę danych.
    
    Argumenty:
        Brak
    
    Zwraca:
        bytes: Kompletny pakiet danych (11 + 3*N bajtów, gdzie N = liczba punktów).
    
    Wyjątki:
        IOError: Gdy wystąpi timeout oczekiwania na dane.
    
    Hierarchia wywołań:
        lidar_driver.py -> get_next_scan() -> _read_one_packet() -> _read_exact()
    """
    # Czyta jeden pełny pakiet danych z lidara
    assert _ser is not None

    while True:
        first = _ser.read(1)
        if not first:
            raise IOError("Timeout oczekiwania na bajt nagłówka")

        if first[0] != HEADER_BYTE:
            # nieprawidłowy nagłówek – szukaj dalej
            continue

        ver_len_bytes = _read_exact(1)
        ver_len = ver_len_bytes[0]
        num_points = ver_len & 0x1F  # dolne 5 bitów = liczba punktów

        if num_points <= 0 or num_points > 32:
            # zły pakiet – szukaj kolejnego nagłówka
            continue

        packet_len = 11 + 3 * num_points
        rest = _read_exact(packet_len - 2)
        packet = first + ver_len_bytes + rest

        if len(packet) != packet_len:
            # niekompletny pakiet – spróbuj ponownie
            continue

        return packet


def _parse_packet(packet: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parsuje surowy pakiet LiDAR do tablic odległości i kątów.
    
    Funkcja dekoduje strukturę pakietu zgodnie z protokołem LiDAR:
      - bajt 0: nagłówek (0x54)
      - bajt 1: wersja + liczba punktów (5 dolnych bitów)
      - bajty 2-3: prędkość
      - bajty 4-5: kąt startowy
      - bajty 6 do N*3+5: dane punktów (odległość 2B + intensywność 1B)
      - ostatnie bajty: kąt końcowy, timestamp, CRC
    
    Argumenty:
        packet (bytes): Surowy pakiet danych z _read_one_packet().
    
    Zwraca:
        Tuple[np.ndarray, np.ndarray]: Krotka (distances_mm, angles_deg):
            - distances_mm: odległości w milimetrach
            - angles_deg: kąty w stopniach [0, 360)
    
    Wyjątki:
        ValueError: Gdy pakiet ma nieprawidłową strukturę.
    
    Hierarchia wywołań:
        lidar_driver.py -> get_next_scan() -> _parse_packet()
    """
    # Parsuje pakiet do tablic distances_mm i angles_deg
    if len(packet) < 11:
        raise ValueError(f"Za krótki pakiet: {len(packet)} B")

    if packet[0] != HEADER_BYTE:
        raise ValueError("Nieprawidłowy byte nagłówka")

    ver_len = packet[1]
    num_points = ver_len & 0x1F
    expected_len = 11 + 3 * num_points
    if len(packet) != expected_len:
        raise ValueError(f"Zły rozmiar pakietu: {len(packet)} zamiast {expected_len}")

    # speed, start_angle (aktualnie nieużywane)
    speed_raw, start_angle_raw = struct.unpack_from("<HH", packet, offset=2)

    distances_mm: List[int] = []
    offset = 6
    for _ in range(num_points):
        dist_raw, intensity_raw = struct.unpack_from("<HB", packet, offset=offset)
        distances_mm.append(dist_raw)
        offset += 3

    end_angle_raw, timestamp = struct.unpack_from("<HH", packet, offset=offset)
    # ostatni bajt CRC pomijamy

    start_deg = (start_angle_raw % 36000) / 100.0
    end_deg = (end_angle_raw % 36000) / 100.0

    angle_span = (end_deg - start_deg) % 360.0
    if num_points > 1:
        step = angle_span / (num_points - 1)
    else:
        step = 0.0

    angles_deg = np.array(
        [(start_deg + i * step) % 360.0 for i in range(num_points)],
        dtype=float,
    )
    distances_mm_arr = np.array(distances_mm, dtype=float)

    return distances_mm_arr, angles_deg


def get_next_scan() -> Tuple[np.ndarray, np.ndarray]:
    """
    Pobiera jeden pakiet danych z LiDAR.
    
    Funkcja odczytuje i parsuje pojedynczy pakiet zwracając tablice
    odległości w metrach i kątów w radianach.
    
    Argumenty:
        Brak
    
    Zwraca:
        Tuple[np.ndarray, np.ndarray]: Krotka (r, theta):
            - r: odległości w metrach
            - theta: kąty w radianach
    
    Wyjątki:
        RuntimeError: Gdy LiDAR nie został zainicjalizowany.
    
    Hierarchia wywołań:
        check_lidar.py -> main() -> get_next_scan()
        lidar_driver.py -> get_full_scan() -> get_next_scan()
    """
    # Zwraca jeden pakiet: r [m], theta [rad]
    if _ser is None or not _ser.is_open:
        raise RuntimeError("Najpierw wywołaj init_lidar().")

    packet = _read_one_packet()
    distances_mm, angles_deg = _parse_packet(packet)

    r = distances_mm / 1000.0      # mm -> m
    theta = np.deg2rad(angles_deg)  # deg -> rad

    return r, theta


def get_full_scan(num_packets: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Łączy wiele pakietów LiDAR w jeden pełny skan 360°.
    
    Funkcja pobiera num_packets pakietów, łączy ich dane i sortuje
    według kąta, tworząc pełny obraz otoczenia.
    
    Argumenty:
        num_packets (int): Liczba pakietów do połączenia. Domyślnie 30.
            Więcej pakietów = wyższa rozdzielczość kątowa.
    
    Zwraca:
        Tuple[np.ndarray, np.ndarray]: Krotka (r, theta) posortowana według kąta:
            - r: odległości w metrach
            - theta: kąty w radianach [0, 2π)
    
    Hierarchia wywołań:
        run_live.py -> main() -> get_full_scan() -> get_next_scan()
        Live_Vis_v3.py -> record_scans() -> get_full_scan() -> get_next_scan()
    """
    # Skleja num_packets pakietów w jeden posortowany skan
    all_r: List[float] = []
    all_theta: List[float] = []

    for _ in range(num_packets):
        r, theta = get_next_scan()
        all_r.extend(r.tolist())
        all_theta.extend(theta.tolist())

    r_arr = np.array(all_r, dtype=float)
    theta_arr = np.array(all_theta, dtype=float)

    order = np.argsort(theta_arr)
    return r_arr[order], theta_arr[order]
