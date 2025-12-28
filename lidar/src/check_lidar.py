from src.hardware.lidar_driver import init_lidar, get_next_scan
from src.config import LIDAR_PORT


def main():
    """
    Testuje połączenie z LiDAR i odczytuje jeden skan.
    
    Funkcja diagnostyczna sprawdzająca:
      1. Czy port szeregowy można otworzyć
      2. Czy LiDAR przesyła poprawne dane
    Wyświetla odczytane wartości r (odległość) i theta (kąt).
    
    Argumenty:
        Brak
    
    Zwraca:
        None
    
    Hierarchia wywołań:
        __main__ -> main() -> init_lidar(), get_next_scan()
    """
    print(f"Próbuję otworzyć port: {LIDAR_PORT}")
    try:
        init_lidar(port=LIDAR_PORT)
        print("Port otwarty poprawnie.")
    except Exception as e:
        print("Błąd przy otwieraniu portu:")
        print(repr(e))
        return

    print("Próbuję odczytać jeden mini-skan (12 punktów)...")
    try:
        r, theta = get_next_scan()
        print("Odczyt danych zakończony powodzeniem.")
        print("r (m):      ", r)
        print("theta (rad):", theta)
    except Exception as e:
        print("Port otwarty, ale wystąpił błąd przy czytaniu danych:")
        print(repr(e))


if __name__ == "__main__":
    main()
