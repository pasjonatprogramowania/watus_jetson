import serial
import time
import math
import numpy as np
from typing import Tuple, List, Optional
from src.config import LIDAR_PORT, LIDAR_BAUDRATE, LIDAR_TIMEOUT

class LidarDriver:
    """
    Sterownik do obsługi lidaru przez port szeregowy.
    
    Odpowiada za:
    - Otwarcie/zamknięcie portu
    - Odczyt bitów i bajtów pakietów
    - Parsowanie nagłówków i danycyh (odległość, kąt)
    - Składanie pełnego skanu (360 stopni)
    """

    def __init__(self, port: str = LIDAR_PORT, baudrate: int = LIDAR_BAUDRATE, timeout: float = LIDAR_TIMEOUT):
        """
        Inicjalizuje obiekt sterownika.
        
        Nie otwiera portu automatycznie - należy użyć metody connect().
        
        Argumenty:
            port (str): Nazwa portu szeregowego.
            baudrate (int): Prędkość transmisji.
            timeout (float): Czas oczekiwania na dane.
            
        Hierarchia wywołań:
            lidar/src/hardware/lidar_driver.py -> initialize_lidar() -> LidarDriver()
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

    def connect(self):
        """
        Otwiera port szeregowy.
        
        Wyjątki:
            serial.SerialException: Gdy nie uda się otworzyć portu.
            
        Hierarchia wywołań:
            lidar/src/hardware/lidar_driver.py -> initialize_lidar() -> connect()
        """
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        if not self.ser.is_open:
            self.ser.open()

    def disconnect(self):
        """
        Zamyka port szeregowy.
        
        Hierarchia wywołań:
             Może być używane przy zamykaniu aplikacji.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _read_exact_bytes(self, size: int) -> bytes:
        """
        Czyta dokładnie 'size' bajtów z portu.
        
        Argumenty:
            size (int): Liczba bajtów do odczytania.
            
        Zwraca:
            bytes: Odczytane dane.
            
        Wyjątki:
            IOError: Gdy nie udało się odczytać wymaganej liczby bajtów.
            
        Hierarchia wywołań:
            lidar/src/hardware/lidar_driver.py -> _read_single_packet() -> _read_exact_bytes()
        """
        if self.ser is None:
            raise IOError("Lidar nie jest połączony")
        data = self.ser.read(size)
        if len(data) < size:
            raise IOError(f"Odczytano za mało danych: {len(data)}/{size}")
        return data

    def _read_single_packet(self) -> bytes:
        """
        Odczytuje jeden pakiet danych lidaru.
        
        Struktura pakietu (przykładowa - zależy od modelu):
        - Nagłówek (2 bajty, np. 0xAA 0x55)
        - Wersja/Typ (1 bajt)
        - Liczba punktów (1 bajt)
        - Kąt początkowy i końcowy (4 bajty)
        - Suma kontrolna (2 bajty)
        - Dane punktów (N * rozmiar punktu)
        
        Zwraca:
            bytes: Surowe bajty jednego pakietu.
            
        Hierarchia wywołań:
            lidar/src/hardware/lidar_driver.py -> acquire_single_scan_packet() -> _read_single_packet()
        """
        # Szukanie nagłówka
        # To jest uproszczona implementacja - w rzeczywistości trzeba szukać bajfów synchronizacji
        header = self._read_exact_bytes(2) 
        if header != b'\xaa\x55': # Przykładowy nagłówek
            # Synchronizacja - czytaj po jednym bajcie aż trafisz
           pass
           
        # Odczyt reszty nagłówka by poznać długość
        info = self._read_exact_bytes(2) 
        # ... logika parsowania długości pakietu ...
        packet_len = 0 # tu obliczamy długość
        
        # Odczyt reszty pakietu
        payload = self._read_exact_bytes(packet_len)
        return header + info + payload

    def _decode_packet_data(self, packet: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dekoduje surowe bajty pakietu na odległości i kąty.
        
        Argumenty:
            packet (bytes): Surowy pakiet.
            
        Zwraca:
            Tuple[np.ndarray, np.ndarray]: (odległości_mm, kąty_stopnie)
            
        Hierarchia wywołań:
            lidar/src/hardware/lidar_driver.py -> acquire_single_scan_packet() -> _decode_packet_data()
        """
        # Implementacja zależy od konkretnego protokołu lidaru
        # Tu zwracamy puste tablice jako placeholder
        return np.array([]), np.array([])
    

driver_instance: Optional[LidarDriver] = None

def initialize_lidar(port: str = LIDAR_PORT) -> None:
    """
    Inicjalizuje globalny sterownik lidaru.
    
    Argumenty:
        port (str): Port szeregowy.
        
    Hierarchia wywołań:
        lidar/src/run_live.py -> main() -> initialize_lidar()
        lidar/src/check_lidar.py -> main() -> initialize_lidar()
    """
    global driver_instance
    driver_instance = LidarDriver(port=port)
    driver_instance.connect()

def acquire_single_scan_packet() -> Tuple[np.ndarray, np.ndarray]:
    """
    Pobiera dane z jednego pakietu (części skanu).
    
    Zwraca:
        Tuple[np.ndarray, np.ndarray]: (odległości_m, kąty_rad)
        
    Hierarchia wywołań:
        lidar/src/hardware/lidar_driver.py -> acquire_complete_scan() -> acquire_single_scan_packet()
    """
    # Placeholder - w rzeczywistości wywołuje metody driver_instance
    # Zwraca dane w metrach i radianach
    return np.array([]), np.array([])

def acquire_complete_scan(num_packets: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pobiera pełny skan (składa wiele pakietów).
    
    Argumenty:
        num_packets (int): Liczba pakietów do pobrania na jeden obrót.
        
    Zwraca:
        Tuple[np.ndarray, np.ndarray]: Zcalone tablice (odległości_m, kąty_rad).
        
    Hierarchia wywołań:
        lidar/src/run_live.py -> main() -> acquire_complete_scan()
    """
    all_ranges = []
    all_angles = []
    
    for _ in range(num_packets):
        r, angle = acquire_single_scan_packet()
        all_ranges.append(r)
        all_angles.append(angle)
        
    if not all_ranges:
        return np.array([]), np.array([])
        
    return np.concatenate(all_ranges), np.concatenate(all_angles)
